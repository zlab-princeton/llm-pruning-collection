import pdb
import sys
import torch
import torch.nn as nn
import numpy as np
import random
import inspect
import argparse

from tqdm import tqdm


from fms.models import get_model
from fms.models.llama import LLaMA
from fms.models.hf import to_hf_api
from fms_fsdp.utils.config_utils import get_model_config

from transformers import AutoTokenizer, AutoModelForCausalLM

# import sys; sys.path.append('/work/nvme/bdhh/yxu21/pruning/fms-wanda/training')
# from eval import get_ppl

def set_seed(seed):
    random.seed(seed)                      # Python random
    np.random.seed(seed)                   # NumPy random
    torch.manual_seed(seed)                # CPU RNG
    torch.cuda.manual_seed(seed)           # GPU RNG (single-GPU)
    torch.cuda.manual_seed_all(seed)       # GPU RNG (multi-GPU)

    torch.backends.cudnn.deterministic = True  # Force cuDNN to be deterministic
    torch.backends.cudnn.benchmark = False 
    torch.use_deterministic_algorithms(True, warn_only=True)
    
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(False)


def compare_hf_fms_weights(hf_model, fms_model, rtol=1e-12, atol=1e-12):
    
    def compare_weights(param1, param2, name):
        if not torch.allclose(param1.data, param2.data.to(param1.dtype).to(param1.device), rtol=rtol, atol=atol):
            max_diff = (param1.data - param2.data.to(param1.device)).abs().max().item()
            print(f"{name} mismatch with max diff {max_diff}")
        else:
            print(f"{name} weight match within tolerance")
            # import pdb; pdb.set_trace()
    
    compare_weights(hf_model.model.embed_tokens.weight, fms_model.shared.emb.weight, 'embedding')
    compare_weights(hf_model.model.norm.weight, fms_model.dec_norm.weight, 'norm')
    compare_weights(hf_model.lm_head.weight, fms_model.shared.head.weight, 'lm_head')
    
    # import pdb; pdb.set_trace()
    
    head_dim = hf_model.config.head_dim
    num_heads = hf_model.config.num_attention_heads
    kv_heads = hf_model.config.num_key_value_heads
    ffn_dim = hf_model.config.intermediate_size
    qkv_splits = [head_dim * num_heads, kv_heads * head_dim, kv_heads * head_dim]
    wg1_splits = [ffn_dim, ffn_dim]
    
    for i in range(hf_model.config.num_hidden_layers):
        
        # compare input_layernorm
        compare_weights(
            hf_model.model.layers[i].input_layernorm.weight,
            fms_model.layers[i].ln.weight,
            f'layer {i} input_layernorm'
        )
        
        # compare attn
        # import pdb; pdb.set_trace()
        q, k, v = torch.split(fms_model.layers[i].attn.in_proj.qkv_fused.weight, qkv_splits, dim=0)
        compare_weights(hf_model.model.layers[i].self_attn.q_proj.weight, q, f'layer {i} q_proj')
        compare_weights(hf_model.model.layers[i].self_attn.k_proj.weight, k, f'layer {i} k_proj')
        compare_weights(hf_model.model.layers[i].self_attn.v_proj.weight, v, f'layer {i} v_proj')
        compare_weights(hf_model.model.layers[i].self_attn.o_proj.weight, fms_model.layers[i].attn.dense.weight, f'layer {i} o_proj')
        
        # compare post_attn_layernorm
        compare_weights(
            hf_model.model.layers[i].post_attention_layernorm.weight,
            fms_model.layers[i].ff_ln.weight,
            f'layer {i} post_attn_layernorm'
        )
        
        # compare mlp
        wg, w1 = torch.split(fms_model.layers[i].ff_sub_layer.wg1_fused.weight, wg1_splits, dim=0)
        compare_weights(hf_model.model.layers[i].mlp.gate_proj.weight, wg, f'layer {i} gate_proj')
        compare_weights(hf_model.model.layers[i].mlp.up_proj.weight, w1, f'layer {i} up_proj')
        compare_weights(hf_model.model.layers[i].mlp.down_proj.weight, fms_model.layers[i].ff_sub_layer.w2.weight, f'layer {i} down_proj')
        
def compare_model_outputs(hf_model, fms_model, tokenizer, device='cuda', atol=1e-12, rtol=1e-12):
    
    hf_model.eval()
    fms_model.eval()
    
    prompt = "The quick brown fox jumps over the lazy dog."
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        # hf_out = hf_model(input_ids, output_hidden_states=True)
        # hf_hidden = hf_out.hidden_states[-1]  
        
        # fms_out = fms_model._helper(input_ids)
        # fms_hidden = fms_out[0]
        
        hf_out = hf_model(input_ids, output_hidden_states=True)
        hf_hidden = hf_out.logits
        
        fms_hidden = fms_model(input_ids) # logits actually; written as hidden for convenience
    
    if not torch.allclose(hf_hidden, fms_hidden, rtol=rtol, atol=atol):
        diff = (hf_hidden - fms_hidden).abs().max().item()
        print(f"Hidden states mismatch: max abs diff = {diff}")
    else:
        print("Hidden states match within tolerance.")
    
    
def compare_module_outputs(hf_model, fms_model, device='cuda', atol=1e-12, rtol=1e-12):
    
    class LinearFromFused(nn.Module):
        def __init__(self, fused_linear, start, end):
            super().__init__()
            self.fused_linear = fused_linear
            self.start = start
            self.end = end

        def forward(self, x):
            return nn.functional.linear(
                x, 
                self.fused_linear.weight[self.start:self.end],
                self.fused_linear.bias[self.start:self.end] if self.fused_linear.bias is not None else None
            )
            
    input_shape = (1, 16)  # batch size 1, seq_len 16
    vocab_size = hf_model.config.vocab_size
    input_ids = torch.randint(0, vocab_size, input_shape).to(device)
    position_ids = torch.arange(0, input_shape[1], dtype=torch.long, device=device).unsqueeze(0)
    
    def compare_outs(module1, module2, inputs, name):
        if 'position_ids' in inspect.signature(module1.forward).parameters:
            out1 = module1(inputs, position_ids=position_ids)
        else:
            out1 = module1(inputs)
        if isinstance(out1, tuple):
            out1 = out1[0]
        out2 = module2(inputs)
        if isinstance(out2, tuple):
            out2 = out2[0]
        out2 = out2.to(out1.dtype).to(out1.device)
        if not torch.allclose(out1, out2, rtol=rtol, atol=atol):
            max_diff = (out1 - out2).abs().max().item()
            print(f"{name} output mismatch with max diff {max_diff}")
        else:
            print(f"{name} output passed")
    
    hf_model.eval()
    fms_model.eval()
    
    head_dim = hf_model.config.head_dim
    num_heads = hf_model.config.num_attention_heads
    kv_heads = hf_model.config.num_key_value_heads
    ffn_dim = hf_model.config.intermediate_size
    qkv_splits = [head_dim * num_heads, kv_heads * head_dim, kv_heads * head_dim]
    wg1_splits = [ffn_dim, ffn_dim]

    with torch.no_grad():
        input_emb = hf_model.model.embed_tokens(input_ids)
        compare_outs(hf_model.model.embed_tokens, fms_model.shared.emb, input_ids, 'embedding')
        compare_outs(hf_model.model.norm, fms_model.dec_norm, input_emb, 'norm')
        # pdb.set_trace()
        compare_outs(hf_model.lm_head, fms_model.shared.head, input_emb, 'head')
        
        for i in range(hf_model.config.num_hidden_layers):
        
            # compare input_layernorm
            compare_outs(
                hf_model.model.layers[i].input_layernorm,
                fms_model.layers[i].ln,
                input_emb,
                f'layer {i} input_layernorm'
            )
            
            # compare attn
            compare_outs(
                hf_model.model.layers[i].self_attn,
                fms_model.layers[i].attn,
                input_emb,
                f'layer {i} attn'
            )
            
            # compare attn projs
            q_proj = LinearFromFused(fms_model.layers[i].attn.in_proj.qkv_fused, 0, qkv_splits[0])
            k_proj = LinearFromFused(fms_model.layers[i].attn.in_proj.qkv_fused, qkv_splits[0], qkv_splits[0] + qkv_splits[1])
            v_proj = LinearFromFused(fms_model.layers[i].attn.in_proj.qkv_fused, qkv_splits[0] + qkv_splits[1], sum(qkv_splits))
            compare_outs(hf_model.model.layers[i].self_attn.q_proj, q_proj, input_emb, f'layer {i} attn q_proj')
            compare_outs(hf_model.model.layers[i].self_attn.k_proj, k_proj, input_emb, f'layer {i} attn k_proj')
            compare_outs(hf_model.model.layers[i].self_attn.v_proj, v_proj, input_emb, f'layer {i} attn v_proj')
            compare_outs(
                hf_model.model.layers[i].self_attn.o_proj, fms_model.layers[i].attn.dense, input_emb, f'layer {i} attn o_proj'
            )
            
            # compare post_attn_layernorm
            compare_outs(
                hf_model.model.layers[i].post_attention_layernorm,
                fms_model.layers[i].ff_ln,
                input_emb,
                f'layer {i} post_attn_layernorm'
            )
            
            # compare mlp
            compare_outs(
                hf_model.model.layers[i].mlp,
                fms_model.layers[i].ff_sub_layer,
                input_emb,
                f'layer {i} mlp'
            )
            
            # compare mlp projs
            wg = LinearFromFused(fms_model.layers[i].ff_sub_layer.wg1_fused, 0, wg1_splits[0])
            w1 = LinearFromFused(fms_model.layers[i].ff_sub_layer.wg1_fused, wg1_splits[0], wg1_splits[0] + wg1_splits[1])
            compare_outs(hf_model.model.layers[i].mlp.gate_proj, wg, input_emb, f'layer {i} mlp gate_proj')
            compare_outs(hf_model.model.layers[i].mlp.up_proj, w1, input_emb, f'layer {i} mlp up_proj')
            # compare_outs(
            #     hf_model.model.layers[i].mlp.down_proj,
            #     fms_model.layers[i].ff_sub_layer.w2,
            #     input_emb,
            #     f'layer {i} mlp down_proj'
            # )
            # w1, g = torch.split(fms_model.layers[i].ff_sub_layer.wg1_fused.weight, wg1_splits, dim=0)
            # compare_weights(hf_model.model.layers[i].mlp.up_proj.weight, w1, f'layer {i} up_proj')
            # compare_weights(hf_model.model.layers[i].mlp.gate_proj.weight, g, f'layer {i} gate_proj')
            # compare_weights(hf_model.model.layers[i].mlp.down_proj.weight, fms_model.layers[i].ff_sub_layer.w2.weight, f'layer {i} down_proj')

        
    
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Compare Hugging Face and FMS model weights and outputs")
    parser.add_argument('--model_variant', type=str, required=True, help='Model variant (e.g., llama3_4b_width)')
    parser.add_argument('--hf_path', type=str, required=True, help='Path to the Hugging Face model')
    parser.add_argument('--fms_path', type=str, required=True, help='Path to the FMS model')
    args = parser.parse_args()
    
    
    dtype = torch.float16
    tokenizer = AutoTokenizer.from_pretrained(args.hf_path)

    hf_model = AutoModelForCausalLM.from_pretrained(args.hf_path, torch_dtype=dtype, device_map='cuda').to(dtype)
    
    llama_config = get_model_config(args.model_variant)
    fms_model = LLaMA(llama_config)
    fms_model.to_empty(device="cpu")
    state_dict = torch.load(args.fms_path, weights_only=False, map_location="cpu")
    model_state_dict = {}
    for k, v in state_dict['model_state'].items():
        if k.startswith("_orig_mod."):
            newk = k[len("_orig_mod."):]
        else:
            newk = k
        model_state_dict[newk] = v.to(dtype)
    fms_model.load_state_dict(model_state_dict)
    fms_model = fms_model.to(dtype).to('cuda')
    
    compare_hf_fms_weights(hf_model, fms_model)
    
    compare_model_outputs(hf_model, fms_model, tokenizer)
    
    compare_module_outputs(hf_model, fms_model)