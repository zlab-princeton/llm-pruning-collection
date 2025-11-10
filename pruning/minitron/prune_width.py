import pdb
import torch
import torch.nn.functional as F

def get_idx(importance, size, layer_idx=0):
    _, idx = torch.sort(importance[layer_idx, :], descending=True)
    return idx[:size]
    
def prune_linear_module(module, in_idx=None, out_idx=None):
    if in_idx is not None:
        module.weight.data = module.weight.data[:, in_idx]
        module.in_features = in_idx.size(0)
    if out_idx is not None:
        module.weight.data = module.weight.data[out_idx, :]
        module.out_features = out_idx.size(0)
        if module.bias is not None:
            module.bias.data = module.bias.data[out_idx]

def width_prune(model, tokenizer, scorer, args):
    
    device = next(model.parameters()).device
    intermediate_size = model.config.intermediate_size
    hidden_size_importance = torch.zeros(1, model.config.hidden_size, device=device)
    ffn_importance = torch.zeros(model.config.num_hidden_layers, intermediate_size, device=device)
    attn_importance = torch.zeros(model.config.num_hidden_layers, model.config.hidden_size, device=device)
    
    def get_ffn_hook(layer_idx):
        def ffn_hook(module, inputs, outputs):
            ffn_input = inputs[0]
            activations = ffn_input.abs().mean(dim=0) 
            activations = activations.pow(2).sum(dim=0) # ffn_hidden_size
            ffn_importance[layer_idx, :] += activations
        return ffn_hook
    
    def LN_hook(module, inputs, outputs):
        activations = outputs.abs().mean(dim=0)
        activations = activations.pow(2).sum(dim=0) # hidden_size
        hidden_size_importance[0, :] += activations
    
    hooks = []
    for i, layer in enumerate(model.model.layers):
        hooks.append(layer.input_layernorm.register_forward_hook(LN_hook))
        hooks.append(layer.post_attention_layernorm.register_forward_hook(LN_hook))
        hooks.append(layer.mlp.down_proj.register_forward_hook(get_ffn_hook(i)))
    
    _ = scorer(model, tokenizer)
    
    # print(hidden_size_importance)
    # print(ffn_importance)
    # pdb.set_trace()
        
    hidden_idx = get_idx(importance=hidden_size_importance, size=args.hidden_size)
    for i, layer in enumerate(model.model.layers):
        # NOTE sort by importance
        ffn_idx = get_idx(importance=ffn_importance, size=args.ffn_hidden_size, layer_idx=i).view(-1)
        
        # NOTE ATTN pruning
        layer.input_layernorm.weight.data = layer.input_layernorm.weight.data[hidden_idx]
        prune_linear_module(module=layer.self_attn.q_proj, in_idx=hidden_idx)
        prune_linear_module(module=layer.self_attn.k_proj, in_idx=hidden_idx)
        prune_linear_module(module=layer.self_attn.v_proj, in_idx=hidden_idx)
        prune_linear_module(module=layer.self_attn.o_proj, out_idx=hidden_idx)
        
        # NOTE MLP pruning
        layer.post_attention_layernorm.weight.data = layer.post_attention_layernorm.weight.data[hidden_idx]
        prune_linear_module(module=layer.mlp.up_proj, in_idx=hidden_idx, out_idx=ffn_idx)
        prune_linear_module(module=layer.mlp.gate_proj, in_idx=hidden_idx, out_idx=ffn_idx)
        layer.mlp.intermediate_size = args.ffn_hidden_size
        
        prune_linear_module(module=layer.mlp.down_proj, in_idx=ffn_idx, out_idx=hidden_idx)
    
    # NOTE prune embedding   
    model.model.embed_tokens.weight.data = model.model.embed_tokens.weight.data[:, hidden_idx]
    model.model.embed_tokens.embedding_dim = args.hidden_size
    # NOTE prune model norm and lm_head
    model.model.norm.weight.data = model.model.norm.weight.data[hidden_idx]
    prune_linear_module(module=model.lm_head, in_idx=hidden_idx)
        
    for hook in hooks:
        hook.remove()
        
    model.config.hidden_size = args.hidden_size
    model.config.intermediate_size = args.ffn_hidden_size