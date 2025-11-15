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
class OrbaxAdapter(LM):
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
                 in_shardings=(self.state_mesh_shardings.params, None, None, None),
                 out_shardings=None)
        def fast_forward(params, input_ids, positions, segment_ids):
            return self.model.apply(
                params,
                input_ids,
                positions,
                segment_ids,
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

        with self.mesh, nn_partitioning.axis_rules(self.config.logical_axis_rules):
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

        for (_text, context_enc, continuation_enc) in tqdm(requests, disable=disable_tqdm):
            input_ids = context_enc + continuation_enc
            input_ids = jnp.asarray([input_ids], dtype=jnp.int32)

            batch_size, seq_len = input_ids.shape
            segment_ids = jnp.ones((batch_size, seq_len), dtype=jnp.int32)
            positions = jnp.tile(jnp.arange(seq_len, dtype=jnp.int32), (batch_size, 1))

            logits = self._compiled_forward(
                self.state.params,
                input_ids,
                positions,
                segment_ids,
            )

            # logits: [1, seq_len, vocab_size]
            # compute log-probs over continuation tokens
            logits = jax.nn.log_softmax(logits, axis=-1)
            cont_len = len(continuation_enc)
            cont_start = input_ids.shape[1] - cont_len

            log_probs = []
            match = True
            for i, tok in enumerate(continuation_enc):
                logp = logits[0, cont_start + i - 1, tok]  # use previous token's logits
                log_probs.append(logp)
                pred = int(jnp.argmax(logits[0, cont_start + i - 1]))
                if pred != tok:
                    match = False

            loglikelihood = float(jnp.sum(jnp.stack(log_probs)))
            results.append((loglikelihood, match))

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
