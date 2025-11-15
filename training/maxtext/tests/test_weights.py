import os
import jax
import jax.numpy as jnp
import numpy as np
import sys
import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer

from MaxText import maxtext_utils
from MaxText import pyconfig
from MaxText.layers import models
from MaxText.layers import quantizations

def unpermute_from_match_maxtext_rope(arr):
  """
  Function to get the RoPE values in correct ordering
  """
  evens = arr[..., ::2]
  odds = arr[..., 1::2]
  return np.concatenate((evens, odds), axis=arr.ndim - 1)

def compare_hf_model_weights(hf_model_1, hf_model_2):
    sd1 = hf_model_1.state_dict()
    sd2 = hf_model_2.state_dict()

    # Verify keys match
    keys1 = set(sd1.keys())
    keys2 = set(sd2.keys())
    if keys1 != keys2:
        print("Key mismatch!")
        print("In model1 not model2:", keys1 - keys2)
        print("In model2 not model1:", keys2 - keys1)

    # Compare parameters
    tolerance = 1e-9  # you can tighten/loosen this
    all_close = True
    for k in sd1.keys():
        if sd1[k].shape != sd2[k].shape:
            print(f"Shape mismatch at {k}: {sd1[k].shape} vs {sd2[k].shape}")
            all_close = False
            continue
        diff = (sd1[k] - sd2[k]).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        if max_diff > tolerance:
            print(f"Parameter {k}: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")
            all_close = False

    if all_close:
        print("✅ All parameters are close within tolerance.")
    else:
        print("⚠️ Some parameters differ beyond tolerance.")
        
def compare_hf_orbax_model_weights(hf_model, orbax_state, config, atol=1e-3, rtol=1e-3, max_print=10):
    # 1. Extract HF parameters into a dict
    hf_params = dict(hf_model.named_parameters())

    # 2. Flatten Orbax model parameters
    orbax_params = jax.tree_util.tree_flatten_with_path(orbax_state.params)[0]

    # 3. Define key mapping logic
    def map_orbax_to_hf_key(orbax_key: str) -> str:
        """
        Convert Orbax param path (dot string) to Hugging Face param key.
        This is not exhaustive — expand as needed.
        """
        key = orbax_key
        key = key.replace("params.decoder.decoder_norm.scale", "model.norm.weight")
        key = key.replace("params.token_embedder.embedding", "model.embed_tokens.weight")
        key = key.replace("params.decoder.logits_dense.kernel", "lm_head.weight")

        key = key.replace("params.decoder.layers_", "model.layers.")
        key = key.replace(".self_attention.query.kernel", ".self_attn.q_proj.weight")
        key = key.replace(".self_attention.key.kernel",   ".self_attn.k_proj.weight")
        key = key.replace(".self_attention.value.kernel", ".self_attn.v_proj.weight")
        key = key.replace(".self_attention.out.kernel",   ".self_attn.o_proj.weight")
        key = key.replace(".mlp.wi_0.kernel",             ".mlp.gate_proj.weight")
        key = key.replace(".mlp.wi_1.kernel",             ".mlp.up_proj.weight")
        key = key.replace(".mlp.wo.kernel",               ".mlp.down_proj.weight")
        key = key.replace(".pre_self_attention_layer_norm.scale",  ".input_layernorm.weight")
        key = key.replace(".post_self_attention_layer_norm.scale", ".post_attention_layernorm.weight")

        return key        
    
    def reshape_orbax_weight(key, value):
        # Self-attention projections
        if "self_attention.query.kernel" in key:
            # From (hidden_dim, num_heads, head_dim) -> (hidden_dim, hidden_dim)
            value = unpermute_from_match_maxtext_rope(value)
            return value.reshape((value.shape[0], -1)).transpose()
        
        elif "self_attention.key.kernel" in key:
            value = unpermute_from_match_maxtext_rope(value)
            return value.reshape((value.shape[0], -1)).transpose()
        
        elif "self_attention.value.kernel" in key:
            return value.reshape((value.shape[0], -1)).transpose()
        
        elif "self_attention.out.kernel" in key:
            # From (num_heads, head_dim, hidden_dim) -> (hidden_dim, hidden_dim)
            return value.transpose(0, 1, 2).reshape((-1, value.shape[-1])).transpose()

        elif "mlp" in key:
            return value.T

        else:
            return value  # No reshape needed

    matched = 0
    mismatched = 0
    hidden_dim = 4096

    for i, (path, orbax_value) in enumerate(orbax_params):
        orbax_key = ".".join(p.key for p in path)
        hf_key = map_orbax_to_hf_key(orbax_key)

        if hf_key not in hf_params:
            print(f"❌ {orbax_key} → {hf_key} not found in HF model")
            continue

        hf_value = hf_params[hf_key].detach().cpu().numpy()
        
        # Temporary patch for q and k: reshape HF to Orbax format
        # if "self_attention.query.kernel" in orbax_key:
        #     print('orbax query original shape', orbax_value)
        #     # HF shape: (hidden_dim, hidden_dim) → Orbax: (hidden_dim, n_q_heads, head_dim)
        #     reshaped = hf_value.T.reshape((hidden_dim, config.base_num_query_heads, config.head_dim))
        #     print(f"🛠️  Reshaping HF q_proj to match Orbax for {orbax_key}")
        #     orbax_value = reshaped
        #     print('orbax query after shape', orbax_value)

        # elif "self_attention.key.kernel" in orbax_key:
        #     print('orbax key original shape', orbax_value)
        #     reshaped = hf_value.T.reshape((hidden_dim, config.base_num_kv_heads, config.head_dim))
        #     print(f"🛠️  Reshaping HF k_proj to match Orbax for {orbax_key}")
        #     orbax_value = reshaped
        #     print('orbax key after shape', orbax_value)

        # Reshape
        orbax_value = reshape_orbax_weight(orbax_key, orbax_value)
        orbax_value = np.asarray(orbax_value)


        if orbax_value.shape != hf_value.shape:
            print(f"❌ Shape mismatch for {orbax_key} → {hf_key}: Orbax {orbax_value.shape}, HF {hf_value.shape}")
            mismatched += 1
            continue

        # Compare values
        abs_diff = np.abs(orbax_value - hf_value)
        max_diff = abs_diff.max()
        mean_diff = abs_diff.mean()

        # if max_diff > atol + rtol * np.abs(hf_value).max():
        print(f"⚠️  Mismatch in {orbax_key} → {hf_key}: max diff = {max_diff:.4e}, mean diff = {mean_diff:.4e}")
        mismatched += 1
        # else:
        #     print(f"✅ Match: {orbax_key} → {hf_key}")
        #     matched += 1

        if i + 1 >= max_print:
            break

    print(f"\nSummary: Matched = {matched}, Mismatched = {mismatched}, Total Checked = {i+1}")
    
import numpy as np

def patch_orbax_weights(hf_model, orbax_state, config, limit=1000):
    hf_params = dict(hf_model.named_parameters())
    orbax_param_tuples = jax.tree_util.tree_flatten_with_path(orbax_state.params)[0]

    patched = 0

    for path, _ in orbax_param_tuples:
        orbax_key = ".".join(p.key for p in path)

        if all(
            k not in orbax_key for k in [
                "self_attention.query.kernel",
                "self_attention.key.kernel",
                "self_attention.value.kernel",
                "self_attention.out.kernel",
            ]
        ):
            continue

        hf_key = orbax_key
        hf_key = hf_key.replace("params.decoder.layers_", "model.layers.")
        hf_key = hf_key.replace(".self_attention.query.kernel", ".self_attn.q_proj.weight")
        hf_key = hf_key.replace(".self_attention.key.kernel", ".self_attn.k_proj.weight")
        hf_key = hf_key.replace(".self_attention.value.kernel", ".self_attn.v_proj.weight")
        hf_key = hf_key.replace(".self_attention.out.kernel", ".self_attn.o_proj.weight")

        if hf_key not in hf_params:
            print(f"⚠️  HF param not found for {orbax_key} → {hf_key}")
            continue

        hf_tensor = hf_params[hf_key].detach().cpu().numpy()
        hidden_dim = hf_tensor.shape[1]  # q_proj/k_proj weight shape: [in_dim, out_dim] → [4096, 4096]

        # print("HF tensor shape", hf_tensor.shape)

        if "query.kernel" in orbax_key:
            # OK!
            reshaped = hf_tensor.T.reshape((hidden_dim, config.base_num_query_heads, config.head_dim)) #/ (np.sqrt(config.head_dim).astype(np.float32))  # pylint: disable=E1137
        elif "key.kernel" in orbax_key:
            # OK!
            reshaped = hf_tensor.T.reshape((hidden_dim, config.base_num_kv_heads,    config.head_dim))
        elif "value.kernel" in orbax_key:
            # OK!
            reshaped = hf_tensor.T.reshape((hidden_dim, config.base_num_kv_heads,    config.head_dim))
        elif "out.kernel" in orbax_key:
            # OK!
            reshaped = hf_tensor.reshape((hidden_dim, config.base_num_query_heads,    config.head_dim)).transpose(1, 2, 0)
        else:
            continue

        # Traverse tree and update in place
        subtree = orbax_state.params
        for key in path[:-1]:
            subtree = subtree[key.key]
        last_key = path[-1].key
        
        before_value = subtree[last_key]
        after_value = reshaped
        
        subtree[last_key] = reshaped  # In-place update

        print(f"✅ Patched {orbax_key} from HF {hf_key}")

        print(f"before shape: {before_value.shape}, values: {before_value.flatten()[:5]}")
        print(f"after  shape: {after_value.shape}, values: {after_value.flatten()[:5]}")
        patched += 1
        
        if patched >= limit:
            break

    print(f"\n✅ Done. Patched {patched} Orbax weights in-place.")
    
def main(config, test_args):
    
    hf_model_1 = AutoModelForCausalLM.from_pretrained(
        # '/home/zephyr/gcs-bucket/model_ckpts/Llama-3.1-8B',
        test_args.hf_model_path,
        torch_dtype=torch.float16,
    )
    hf_model_2 = AutoModelForCausalLM.from_pretrained(
        '/home/zephyr/gcs-bucket/model_ckpts/hf/llama3.1_1b_scratch_back',
        torch_dtype=torch.float16,
    )
    
    init_rng = jax.random.PRNGKey(config.init_weights_seed)
    init_rng, rng1 = jax.random.split(init_rng)
    devices_array = maxtext_utils.create_device_mesh(config)
    mesh = jax.sharding.Mesh(devices_array, config.mesh_axes)
    quant = quantizations.configure_quantization(config)
    orbax_model = models.Transformer(config, mesh, quant=quant)
    orbax_state, _ = maxtext_utils.setup_decode_state(orbax_model, config, rng1, mesh, None)
    
    compare_hf_model_weights(hf_model_1, hf_model_2)
    
    # patch_orbax_weights(hf_model_1, orbax_state, config, limit=4)
    
    compare_hf_orbax_model_weights(hf_model_1, orbax_state, config)

if __name__ == "__main__":
    jax.config.update("jax_default_prng_impl", "unsafe_rbg")
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"

    parser = argparse.ArgumentParser()
    parser.add_argument("--atol", type=float, required=False, default=0.1)
    parser.add_argument("--rtol", type=float, required=False, default=0.1)
    parser.add_argument("--token_size", type=int, required=False)
    parser.add_argument("--max_kl_div", type=float, required=False, default=None)
    parser.add_argument("--golden_logits_path", type=str, required=False, default="")
    parser.add_argument("--hf_model_path", type=str, required=False, default="")
    parser.add_argument("--run_hf_model", type=bool, required=False, default=False)
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