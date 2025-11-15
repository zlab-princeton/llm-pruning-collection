# tests/test_grads.py
import os
import sys
import jax
import jax.numpy as jnp
import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer

from MaxText import maxtext_utils
from MaxText import pyconfig
from MaxText.layers import models
from MaxText.layers import quantizations
from MaxText.common_types import DECODING_ACTIVE_SEQUENCE_INDICATOR

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

def main(config, test_args):
    if test_args.hf_model_path == "":
        raise ValueError("You must pass a HF model path")

    # ---------------- HF model ----------------
    hf_model = AutoModelForCausalLM.from_pretrained(
        test_args.hf_model_path, torch_dtype=torch.bfloat16
    )
    tokenizer = AutoTokenizer.from_pretrained(test_args.hf_model_path)
    tokenizer.pad_token = tokenizer.eos_token

    # ---------------- MaxText model ----------------
    init_rng = jax.random.PRNGKey(config.init_weights_seed)
    init_rng, rng1 = jax.random.split(init_rng)
    devices_array = maxtext_utils.create_device_mesh(config)
    mesh = jax.sharding.Mesh(devices_array, config.mesh_axes)
    quant = quantizations.configure_quantization(config)
    maxtext_model = models.Transformer(config, mesh, quant=quant)
    maxtext_state, _ = maxtext_utils.setup_decode_state(
        maxtext_model, config, rng1, mesh, None
    )

    # ---------------- Prompts ----------------
    prompts = ["I love to", "Today is a", "What is the"]

    for input_text in prompts:
        print(f"\n--- Prompt: {input_text} ---")

        # HF tokenize
        inputs = tokenizer(
            input_text,
            return_tensors="pt",
            padding=True,
            max_length=config.max_target_length,
            truncation=True,
        )
        actual_seq_len = inputs["input_ids"].shape[1]
        labels = inputs["input_ids"].clone()

        # MaxText tokenize
        mt_ids = jnp.asarray(inputs["input_ids"], dtype=jnp.int32)
        if mt_ids.shape[0] != config.global_batch_size_to_train_on:
            mt_ids = jnp.repeat(
                mt_ids,
                config.global_batch_size_to_train_on // mt_ids.shape[0],
                axis=0,
            )

        s = (config.global_batch_size_to_train_on, config.max_target_length)
        mt_decoder_segment_ids_full = (
            jnp.zeros(s, dtype=jnp.int32) + DECODING_ACTIVE_SEQUENCE_INDICATOR
        )
        mt_decoder_segment_ids = mt_decoder_segment_ids_full[:, :actual_seq_len]

        mt_decoder_positions_full = jnp.stack(
            [
                jnp.arange(config.max_target_length, dtype=jnp.int32)
                for _ in range(config.global_batch_size_to_train_on)
            ]
        )
        mt_decoder_positions = mt_decoder_positions_full[:, :actual_seq_len]

        # ---------------- HF Forward + Grad ----------------
        hf_model.train()
        for p in hf_model.parameters():
            if p.grad is not None:
                p.grad.zero_()

        outputs = hf_model(**inputs, labels=labels)
        loss_hf = outputs.loss
        loss_hf.backward()

        print(f"HF loss: {loss_hf.item():.6f}")
        for name, param in hf_model.named_parameters():
            if any('layers.{i}' in name for i in range(30)):
                continue
            if param.grad is not None:
                g = param.grad.detach().cpu().float().flatten()
                print(
                    f"[HF] {name:<40} grad: mean={g.mean():+.3e}, std={g.std():.3e}, max={g.abs().max():.3e}"
                )
                # break  # remove to dump all grads

        # ---------------- MaxText Forward + Grad ----------------
        def loss_fn(params, mt_ids, mt_positions, mt_segments, labels):
            logits = maxtext_model.apply(
                params,
                mt_ids,
                mt_positions,
                mt_segments,
                enable_dropout=False,
                rngs={"aqt": init_rng},
            )
            logits = logits[:, :actual_seq_len, :]  # [batch, seq, vocab]
            log_probs = jax.nn.log_softmax(logits, axis=-1)
            nll = -jnp.take_along_axis(
                log_probs, labels[..., None], axis=-1
            ).squeeze(-1)
            return nll.mean()

        loss_mt, grads = jax.value_and_grad(loss_fn)(
            maxtext_state.params,
            mt_ids,
            mt_decoder_positions,
            mt_decoder_segment_ids,
            mt_ids,
        )
        print(f"MaxText loss: {float(loss_mt):.6f}")

        flat_grads = jax.tree_util.tree_flatten_with_path(grads)[0]
        for path, g in flat_grads:
            grad_key = ".".join(p.key for p in path)
            if any('layers_{i}' in grad_key for i in range(30)):
                continue
            g = g.reshape(-1)
            print(
                f"[MaxText] {grad_key:60s} grad: mean={g.mean():+.3e}, "
                f"std={g.std():.3e}, max={jnp.abs(g).max():.3e}"
            )
            # break  # remove to dump all grads


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