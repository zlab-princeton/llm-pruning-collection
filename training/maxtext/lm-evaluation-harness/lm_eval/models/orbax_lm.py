import random

import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm
from functools import partial
from transformers import AutoTokenizer

from lm_eval.api.model import LM
from lm_eval.api.registry import register_model

from MaxText import maxtext_utils
from MaxText import pyconfig
from MaxText.layers import models
from MaxText.layers import quantizations
from MaxText.utils.ckpt_conversion.utils.hf_utils import (
    convert_jax_weight_to_torch,
)
from MaxText import maxengine

from jax.experimental import mesh_utils
from jax.sharding import Mesh
from jax.experimental.pjit import pjit
from flax.linen import partitioning as nn_partitioning


def loss_fn(model, config, data, dropout_rng, params, is_train=True):
  """loss_fn for both train and eval.

  Args:
    model: A nn.Module
    config: Config of parameters
    data: Batch of data to apply to the model
    dropout_rng: A key to use to generate rng for dropout
    params: Model params
    is_train: True for train_step and False for eval_step

  Returns:
    loss: average loss
    aux: a dictionary including intermediate_outputs, total_loss, and total_weights
  """
  # inputs, targets, segments, positions = apply_args
  rng1, aqt_rng = jax.random.split(dropout_rng)

  # decimate proportion of data when per_device_batch_size<1
  if is_train:
    for k, v in data.items():
      data[k] = v[: config.micro_batch_size_to_train_on, :]
  else:
    for k, v in data.items():
      data[k] = v[: config.micro_batch_size_to_eval_on, :]
  mutable_collections = ["intermediates"]
  if config.mtp_num_layers > 0 and is_train:
    # The single model.apply call now triggers the entire chain if MTP is enabled:
    # Decoder runs -> returns hidden_state -> MTPBlock uses it -> MTPBlock sows losses -> we reap them here.
    mutable_collections.append("mtp_losses")

  # During evaluation, if the acceptance rate test is enabled, we must
  # make its specific collection mutable so the MTPBlock can sow into it.
  if config.mtp_eval_target_module > 0 and not is_train:
    mutable_collections.append("mtp_acceptance")

  logits, intermediate_outputs = model.apply(
      params,
      data["inputs"],
      data["inputs_position"],
      decoder_segment_ids=data["inputs_segmentation"],
      encoder_images=data["images"] if config.use_multimodal else None,
      enable_dropout=config.enable_dropout if is_train else False,
      rngs={"dropout": rng1, "params": aqt_rng},
      mutable=mutable_collections,
      decoder_target_tokens=data["targets"],
      decoder_target_mask=data["targets_segmentation"],
  )
  one_hot_targets = jax.nn.one_hot(data["targets"], config.vocab_size)
  xent, _ = max_utils.cross_entropy_with_logits(logits, one_hot_targets, 0.0)
  xent = nn.with_logical_constraint(xent, ("activation_embed_and_logits_batch", "activation_length"))
  # Mask out paddings at the end of each example.
  xent = xent * (data["targets_segmentation"] != 0)
  total_loss = jnp.sum(xent)
  total_weights = jnp.sum(data["targets_segmentation"] != 0)
  loss = total_loss / (total_weights + EPS)

  # Calculate and Add MTP Loss
  mtp_loss = 0.0
  if config.mtp_num_layers > 0 and is_train:
    mtp_loss = calculate_mtp_loss(intermediate_outputs, config)
    loss += mtp_loss

  # get moe load balance loss
  moe_lb_loss = 0.0
  if config.num_experts > 1:
    nested_key = ("intermediates", "decoder", "layers", "moe_lb_loss")
    total_moe_lb_loss = maxtext_utils.get_nested_value(intermediate_outputs, nested_key, 0.0)
    moe_lb_loss = jnp.mean(jnp.array(total_moe_lb_loss))
    loss += moe_lb_loss

  # Add the model's primary output to the intermediates dict so it can be used
  # by the acceptance rate calculation in eval_step.
  intermediate_outputs["logits"] = logits

  aux = {
      "intermediate_outputs": intermediate_outputs,
      "total_loss": total_loss,
      "total_weights": total_weights,
      "moe_lb_loss": moe_lb_loss,
      "mtp_loss": mtp_loss,
  }
  return loss, aux

@partial(jax.jit, static_argnames=["model"])
def forward_jax(model, params, input_ids, positions, segment_ids):
    return model.apply(
        params,
        input_ids,
        positions,
        segment_ids,
        enable_dropout=False,
        rngs={"aqt": jax.random.PRNGKey(0)},
    )

@register_model("orbax_lm")
class OrbaxLM(LM):
    def __init__(self, model, state, tokenizer, config, state_mesh_shardings, mesh) -> None:
        super().__init__()
        self.model = model
        self.state = state
        self.tokenizer = tokenizer
        self.config = config
        self.state_mesh_shardings = state_mesh_shardings
        self.mesh = mesh

        self._compiled_forward = self._create_fast_forward()
        
    def _create_fast_forward(self):
        @partial(pjit,
                 in_shardings=(self.state_mesh_shardings.params, None, None, None, None),
                 out_shardings=None)
        def fast_forward(params, input_ids, positions, segment_ids, decoder_target_mask):
            return self.model.apply(
                params,
                input_ids,
                positions,
                segment_ids,
                decoder_target_mask=decoder_target_mask,
                enable_dropout=False,
                rngs={"aqt": jax.random.PRNGKey(0)},
            )
        return fast_forward
       
    def forward(self, input_ids, **kwargs):
        input_ids_np = input_ids.cpu().numpy()
        input_ids_jax = jnp.asarray(input_ids_np, dtype=jnp.int32)

        batch_size, seq_len = input_ids_jax.shape
        segment_ids = jnp.ones((batch_size, seq_len), dtype=jnp.int32)
        positions = jnp.tile(jnp.arange(seq_len, dtype=jnp.int32), (batch_size, 1))

        # with self.mesh, nn_partitioning.axis_rules(self.config.logical_axis_rules):
        jax_logits = self._compiled_forward(
            self.state.params,
            input_ids_jax,
            positions,
            segment_ids,
        )

        class Output:
            def __init__(self, logits):
                self.logits = convert_jax_weight_to_torch(logits)

        return Output(jax_logits)

    @classmethod
    def create_from_arg_string(cls, arg_string, additional_config=None):
        return cls

    def _tokenize(self, text):
        enc = self.tokenizer(text, return_tensors="pt", padding=True, max_length=self.config.max_target_length, truncation=True)
        return enc["input_ids"].numpy(), enc["attention_mask"].numpy()

    def tok_encode(
        self,
        string: str,
        left_truncate_len: int | None = None,
        add_special_tokens: bool | None = None,
    ) -> list[int]:
        """ """
        # default for None - empty dict, use predefined tokenizer param
        # used for all models except for CausalLM or predefined value
        special_tokens_kwargs = {}

        # by default for CausalLM - false or self.add_bos_token is set
        # if add_special_tokens is None:
        #     if self.backend == "causal":
        #         special_tokens_kwargs = {
        #             "add_special_tokens": False or self.add_bos_token
        #         }
        # # otherwise the method explicitly defines the value
        # else:
        # special_tokens_kwargs = {"add_special_tokens": add_special_tokens}

        encoding = self.tokenizer.encode(string)

        # left-truncate the encoded context to be at most `left_truncate_len` tokens long
        if left_truncate_len:
            encoding = encoding[-left_truncate_len:]

        return encoding

    def loglikelihood(
        self, requests: list["Instance"], disable_tqdm: bool = False
    ) -> list[tuple[float, bool]]:
        new_reqs = []
        for context, continuation in [req.args for req in requests]:
            if context == "":
                # BOS or EOS as context
                context_enc, continuation_enc = (
                    [self.prefix_token_id],
                    self.tok_encode(continuation),
                )
            else:
                context_enc, continuation_enc = self._encode_pair(context, continuation)

            new_reqs.append(((context, continuation), context_enc, continuation_enc))

        return self._loglikelihood_tokens(new_reqs, disable_tqdm=disable_tqdm)

    def _loglikelihood_tokens(
        self,
        requests: list[tuple[tuple[str, str], list[int], list[int]]],
        disable_tqdm: bool = False,
        override_bs: int | None = None,
    ) -> list[tuple[float, bool]]:
        results = []

        max_len = max(len(ctx) + len(cont) for _, ctx, cont in requests)
        pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        def pad(seq):
            return seq + [pad_id] * (max_len - len(seq))

        with self.mesh, nn_partitioning.axis_rules(self.config.logical_axis_rules):
            for (_, context_enc, continuation_enc) in tqdm(requests, disable=disable_tqdm):

                seq = context_enc + continuation_enc
                seq_len = len(seq)
                seq = pad(seq)
                input_ids = jnp.asarray([seq], dtype=jnp.int32)

                L = input_ids.shape[1]
                positions = jnp.tile(jnp.arange(L, dtype=jnp.int32), (1, 1))
                segment_ids = jnp.ones((1, L), dtype=jnp.int32)
                
                decoder_target_mask = (input_ids != pad_id)

                logits = self._compiled_forward(
                    self.state.params, 
                    input_ids, 
                    positions, 
                    segment_ids, 
                    decoder_target_mask,
                )
                logits = jnp.where(decoder_target_mask[..., None], logits, -1e30)
                logits = jax.nn.log_softmax(logits, axis=-1)

                cont_len = len(continuation_enc)
                cont_start = seq_len - cont_len

                idxs = jnp.arange(cont_start - 1, cont_start - 1 + cont_len)
                tok_ids = jnp.asarray(continuation_enc)

                log_probs = logits[0, idxs, tok_ids]
                ll = float(jnp.sum(log_probs))

                preds = jnp.argmax(logits[0, idxs], axis=-1)
                match = bool(jnp.all(preds == tok_ids))

                results.append((ll, match))

        return results
    
    def _encode_pair(
        self, context: str, continuation: str
    ) -> tuple[list[int], list[int]]:
        import transformers

        n_spaces = len(context) - len(context.rstrip())
        if n_spaces > 0:
            continuation = context[-n_spaces:] + continuation
            context = context[:-n_spaces]

        model_class = getattr(self, "AUTO_MODEL_CLASS", None)

        if model_class == transformers.AutoModelForSeq2SeqLM:
            context_enc = self.tok_encode(context)
            continuation_enc = self.tok_encode(continuation, add_special_tokens=False)
        else:
            whole_enc = self.tok_encode(context + continuation)
            context_enc = self.tok_encode(context)

            context_enc_len = len(context_enc)
            continuation_enc = whole_enc[context_enc_len:]

        return context_enc, continuation_enc

    def generate_until(self, requests, disable_tqdm: bool = False):
        raise NotImplementedError
        
        # res = []

        # for request in tqdm(requests, disable=disable_tqdm):
        #     res.append("lol")
        #     assert request.arguments[0].strip() != ""

        # return res

    def loglikelihood_rolling(self, requests, disable_tqdm: bool = False):
        raise NotImplementedError
    
        results = []
        for text in tqdm(requests, disable=disable_tqdm):
            input_ids, _ = self._tokenize(text)
            mt_ids = jnp.asarray(input_ids, dtype=jnp.int32)
            actual_seq_len = mt_ids.shape[1]

            mt_decoder_segment_ids = jnp.zeros((mt_ids.shape[0], actual_seq_len), dtype=jnp.int32) + 1
            mt_decoder_positions = jnp.stack([jnp.arange(actual_seq_len, dtype=jnp.int32)] * mt_ids.shape[0])

            logits = self.model.apply(
                self.state.params,
                mt_ids,
                mt_decoder_positions,
                mt_decoder_segment_ids,
                enable_dropout=False,
                rngs={"aqt": jax.random.PRNGKey(0)},
            )

            # rolling logp placeholder
            loglikelihood = -1.0  # TODO: compute rolling log-likelihood from logits
            results.append(loglikelihood)

        return results
