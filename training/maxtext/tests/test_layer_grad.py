# -*- coding: utf-8 -*-
"""
Compare HF vs MaxText on ONLY layer-0:
- Forward to layer-0 output
- Make a simple MSE loss on that hidden state
- Backprop and print per-module grads for layer-0

Assumptions:
- Your HF checkpoint is LLaMA-like (LlamaForCausalLM or compatible) with modules:
    model.embed_tokens, model.layers[0].self_attn.{q_proj,k_proj,v_proj,o_proj},
    model.layers[0].input_layernorm, model.layers[0].post_attention_layernorm,
    model.layers[0].mlp.{gate_proj,up_proj,down_proj}
- Your MaxText model exposes:
    Transformer(config, mesh, quant), and after bind() you can access:
      bound.decoder.shared_embedding  (or bound.shared_embedding depending on version)
      bound.decoder.decoder_layer[0]  (a single DecoderLayer)
    DecoderLayer signature matches your pasted code:
      layer0(inputs, decoder_segment_ids, decoder_positions, deterministic, model_mode, ...)
- We DO NOT try to align logits head here; loss is MSE on hidden state to avoid pulling in extra params.
"""
import os
import sys
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

import jax
import jax.numpy as jnp

# ---- Import your MaxText deps (adjust these names to your tree) ----
from MaxText import maxtext_utils
from MaxText import pyconfig
from MaxText.layers import models
from MaxText.layers import quantizations
from MaxText.common_types import MODEL_MODE_TRAIN, DECODING_ACTIVE_SEQUENCE_INDICATOR


def str2bool(v):
    if isinstance(v, bool):
        return v
    v = v.lower()
    if v in ("yes", "true", "t", "1"):
        return True
    elif v in ("no", "false", "f", "0"):
        return False
    else:
        raise ValueError(f"Invalid boolean value: {v}")
    

def _to_torch(x):
    return torch.tensor(np.array(x), dtype=torch.float32)


def _to_jnp(x):
    return jnp.asarray(x)


def build_attention_mask_torch(input_ids, pad_token_id):
    # HF attention mask: 1 for tokens that are not masked, 0 for masked
    return (input_ids != pad_token_id).long()


def run_hf_layer0_and_grads(hf_model, tokenizer, prompt: str, device="cpu"):
    hf_model.train().to(device)

    # Tokenize
    batch = tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    )
    input_ids = batch["input_ids"].to(device)
    attn_mask = build_attention_mask_torch(input_ids, tokenizer.pad_token_id).to(device)

    # Freeze all layers except layer-0 to localize grads
    for n, p in hf_model.named_parameters():
        p.requires_grad_(False)
    for p in hf_model.model.layers[0].parameters():
        p.requires_grad_(True)

    # ---- Forward to layer-0 output via a hook ----
    layer0_out = {}

    def hook_layer0_out(module, inp, out):
        # out is a tuple for some archs; for LLaMA it is usually hidden_states
        layer0_out["hidden"] = out[0] if isinstance(out, tuple) else out

    handle = hf_model.model.layers[0].register_forward_hook(hook_layer0_out)

    # Forward whole model (cheapest to reuse HF plumbing); only layer-0 has requires_grad=True
    with torch.autocast(device_type="cuda" if device.startswith("cuda") else "cpu", dtype=torch.bfloat16, enabled=(hf_model.dtype==torch.bfloat16)):
        outputs = hf_model(input_ids=input_ids, attention_mask=attn_mask, output_hidden_states=False)

    handle.remove()

    hidden0 = layer0_out["hidden"].float()  # [B, T, C]

    # ---- Construct a simple MSE loss on hidden0 to keep graph local to layer-0 ----
    # Use a zero target so we don't pull other params (like lm_head) into the loss.
    loss = (hidden0 ** 2).mean()

    # Backprop
    hf_model.zero_grad(set_to_none=True)
    loss.backward()

    # ---- Collect grad stats for layer-0 modules ----
    stats = {}

    def stat_of(p):
        if p.grad is None:
            return None
        g = p.grad.detach().float().reshape(-1)
        return dict(mean=float(g.mean()), std=float(g.std()), max=float(g.abs().max()))

    with torch.no_grad():
        embed = hf_model.model.embed_tokens
        # Attention Q,K,V,O
        q = hf_model.model.layers[0].self_attn.q_proj
        k = hf_model.model.layers[0].self_attn.k_proj
        v = hf_model.model.layers[0].self_attn.v_proj
        o = hf_model.model.layers[0].self_attn.o_proj
        # Norms
        ln_in = hf_model.model.layers[0].input_layernorm
        ln_post = hf_model.model.layers[0].post_attention_layernorm
        # MLP
        gate = hf_model.model.layers[0].mlp.gate_proj
        up = hf_model.model.layers[0].mlp.up_proj
        down = hf_model.model.layers[0].mlp.down_proj

        stats["HF model.embed_tokens.weight"] = stat_of(embed.weight)
        stats["HF L0 q_proj.weight"] = stat_of(q.weight)
        stats["HF L0 k_proj.weight"] = stat_of(k.weight)
        stats["HF L0 v_proj.weight"] = stat_of(v.weight)
        stats["HF L0 o_proj.weight"] = stat_of(o.weight)
        stats["HF L0 input_layernorm.weight"] = stat_of(ln_in.weight)
        stats["HF L0 post_attention_layernorm.weight"] = stat_of(ln_post.weight)
        stats["HF L0 mlp.gate_proj.weight"] = stat_of(gate.weight)
        stats["HF L0 mlp.up_proj.weight"] = stat_of(up.weight)
        stats["HF L0 mlp.down_proj.weight"] = stat_of(down.weight)

    return dict(
        loss=float(loss.item()),
        hidden_shape=tuple(hidden0.shape),
        stats=stats,
        batch={"input_ids": input_ids.cpu(), "attn_mask": attn_mask.cpu()},
    )


def run_maxtext_layer0_and_grads(config, prompt: str):
    # Build tokenizer consistent with HF path in config
    # （如果你已有外部 tokenizer，就直接传入 ids）
    # 这里我们假设用 HF tokenizer 生成 ids，再喂给 MaxText
    # —— 为了复用 run_hf_layer0_and_grads 的 batch，我们在 main() 里传过来

    # Create model/state
    init_rng = jax.random.PRNGKey(config.init_weights_seed)
    init_rng, rng1 = jax.random.split(init_rng)
    devices_array = maxtext_utils.create_device_mesh(config)
    mesh = jax.sharding.Mesh(devices_array, config.mesh_axes)
    quant = quantizations.configure_quantization(config)
    mt_model = models.Transformer(config, mesh, quant=quant)
    mt_state, _ = maxtext_utils.setup_decode_state(mt_model, config, rng1, mesh, None)

    # Bind variables so we can call submodules
    bound = mt_model.bind(mt_state.params)

    # Helper to run embedding only (calls Decoder._apply_embedding)
    def embed_only(input_ids_jnp, positions_jnp, segment_ids_jnp):
        # Depending on your MaxText version, shared_embedding may be at:
        #   bound.shared_embedding  OR  bound.decoder.shared_embedding
        # We will access via decoder to match your pasted code.
        cfg = config
        y = bound.decoder._apply_embedding(
            input_ids_jnp, positions_jnp, deterministic=True, model_mode=MODEL_MODE_TRAIN,
            image_embeddings=None, bidirectional_mask=None
        )
        return y.astype(cfg.dtype)

    # Build token ids/positions/segments (we’ll construct in main and pass in)
    def loss_on_layer0(params, ids, pos, seg):
        # Re-bind with new params for grad
        b = mt_model.bind(params)
        y = b.decoder._apply_embedding(ids, pos, deterministic=True, model_mode=MODEL_MODE_TRAIN,
                                       image_embeddings=None, bidirectional_mask=None)
        # Call ONLY layer-0
        layer0 = b.decoder.decoder_layer[0]
        y0, _ = layer0(
            y,
            seg,
            pos,
            previous_chunk=None,
            slot=None,
            page_state=None,
        )
        # MSE loss on y0, to keep grads inside layer-0 (plus norms inside it)
        return jnp.mean(jnp.square(y0))

    # We return callable loss so main() can pass aligned ids from HF
    return mt_model, mt_state, loss_on_layer0


def pretty_print_stats(title, stats_dict):
    print(f"\n=== {title} ===")
    for k, v in stats_dict.items():
        if v is None:
            print(f"{k:<55} grad: None")
        else:
            print(f"{k:<55} grad: mean={v['mean']:+.3e}, std={v['std']:.3e}, max={v['max']:.3e}")


def main(config, args):

    # ---- Load tokenizer & HF model ----
    tokenizer = AutoTokenizer.from_pretrained(args.hf_model_path)
    tokenizer.pad_token = tokenizer.eos_token
    hf_model = AutoModelForCausalLM.from_pretrained(args.hf_model_path, torch_dtype=torch.bfloat16)

    args.prompt = "Hello, how are you!"

    # ---- HF: run to layer-0 and backprop ----
    hf_res = run_hf_layer0_and_grads(hf_model, tokenizer, args.prompt)
    print(f"\nHF layer-0 MSE loss: {hf_res['loss']:.6f}, hidden shape: {hf_res['hidden_shape']}")
    pretty_print_stats("HF layer-0 grad stats", hf_res["stats"])

    # ---- MaxText: construct model + single-layer loss ----
    #   （注意：JAX/Flax/Orbax 这部分如果有 API 不同，你需要参照你的版本稍微调整，
    #     我已经尽量贴近你上面贴出来的 MaxText 代码）

    mt_model, mt_state, mt_loss_fn = run_maxtext_layer0_and_grads(cfg, args.prompt)

    # Reuse HF batch ids to ensure same tokenization
    input_ids = hf_res["batch"]["input_ids"].numpy()           # [B, T]
    B, T = input_ids.shape
    ids_j = jnp.asarray(input_ids, dtype=jnp.int32)

    # Positions & segments for MaxText
    pos = jnp.tile(jnp.arange(T, dtype=jnp.int32)[None, :], (B, 1))
    seg = jnp.zeros_like(ids_j, dtype=jnp.int32) + DECODING_ACTIVE_SEQUENCE_INDICATOR

    # ---- Compute loss & grads on MaxText layer-0 ----
    loss_val, grads = jax.value_and_grad(mt_loss_fn)(mt_state.params, ids_j, pos, seg)

    # Flatten with names
    flat = jax.tree_util.tree_flatten_with_path(grads)[0]
    mt_stats = {}

    def reduce_stats(arr):
        arr = arr.reshape(-1)
        return dict(mean=float(arr.mean()), std=float(arr.std()), max=float(jnp.abs(arr).max()))

    # Collect only layer-0 related params
    for path, g in flat:
        key = ".".join(p.key for p in path)
        if ("decoder.layers_0" in key) or ("decoder.layers.layers_0" in key):
            mt_stats[f"MaxText {key}"] = None if g is None else reduce_stats(g)

    print(f"\nMaxText layer-0 MSE loss: {float(loss_val):.6f}")
    pretty_print_stats("MaxText layer-0 grad stats", mt_stats)

    # ---- Highlight the key mappings for attention ----
    print("\n# Attention mapping to check side-by-side:")
    print("HF: model.layers.0.self_attn.q_proj.weight    <-> MaxText: params.decoder.layers_0.self_attention.query.kernel")
    print("HF: model.layers.0.self_attn.k_proj.weight    <-> MaxText: params.decoder.layers_0.self_attention.key.kernel")
    print("HF: model.layers.0.self_attn.v_proj.weight    <-> MaxText: params.decoder.layers_0.self_attention.value.kernel")
    print("HF: model.layers.0.self_attn.o_proj.weight    <-> MaxText: params.decoder.layers_0.self_attention.out.kernel")
    print("HF: model.layers.0.input_layernorm.weight     <-> MaxText: params.decoder.layers_0.pre_self_attention_layer_norm.scale")
    print("HF: model.layers.0.post_attention_layernorm.weight <-> MaxText: params.decoder.layers_0.post_self_attention_layer_norm.scale")
    print("HF: model.layers.0.mlp.gate_proj.weight       <-> MaxText: params.decoder.layers_0.mlp.wi_0.kernel")
    print("HF: model.layers.0.mlp.up_proj.weight         <-> MaxText: params.decoder.layers_0.mlp.wi_1.kernel")
    print("HF: model.layers.0.mlp.down_proj.weight       <-> MaxText: params.decoder.layers_0.mlp.wo.kernel")

    print("\nTip: 如果你看到 MaxText 的 query.kernel 的 grad std/ max 明显是 HF q_proj 的 ~√head_dim 倍，几乎可以确定是 pre-scale Q 导致的。把转换里的 scale_query 关掉，或确认前向没有再除一次 √d。")


if __name__ == "__main__":
    jax.config.update("jax_default_prng_impl", "unsafe_rbg")
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"

    parser = argparse.ArgumentParser()
    parser.add_argument("--atol", type=float, required=False, default=0.1)
    parser.add_argument("--rtol", type=float, required=False, default=0.1)
    parser.add_argument("--token_size", type=int, required=False)
    parser.add_argument("--max_kl_div", type=float, required=False, default=0.1)
    parser.add_argument("--golden_logits_path", type=str, required=False, default="")
    parser.add_argument("--hf_model_path", type=str, required=False, default="")
    parser.add_argument("--run_hf_model", type=str2bool, required=False, default=False)
    test_args, _ = parser.parse_known_args()

    # Remove args defined in this test file to avoid error from pyconfig
    model_args = sys.argv
    to_remove_args = [
        "--atol",
        "--rtol",
        "--token_size",
        "--max_kl_div",
        "--golden_logits_path",
        "--hf_model_path",
        "--run_hf_model",
    ]
    for arg in to_remove_args:
        model_args = [s for s in model_args if not s.startswith(arg)]

    cfg = pyconfig.initialize(model_args)
    main(cfg, test_args)