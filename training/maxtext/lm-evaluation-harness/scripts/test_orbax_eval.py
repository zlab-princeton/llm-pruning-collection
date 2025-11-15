# import hydra
# requirements: 
# pip install sacrebleu accelerate peft 
import os
import json
import jax
import jax.numpy as jnp
import numpy as np
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import argparse

from tqdm import tqdm
from functools import partial
from datasets import load_dataset
from omegaconf import DictConfig, OmegaConf
from transformers import AutoModelForCausalLM, AutoTokenizer
from lm_eval import evaluator
from lm_eval.models.orbax_lm import OrbaxLM

from MaxText import maxtext_utils
from MaxText import pyconfig
from MaxText.layers import models
from MaxText.layers import quantizations

from jax.sharding import Mesh
from jax.experimental import mesh_utils

import math

def str2bool(v):
    if isinstance(v, bool):
        return v
    val = v.lower()
    if val in ("yes", "true", "t", "y", "1"):
        return True
    elif val in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected (yes/no/true/false)")

PPL_TASKS = [
    "c4",
    "wikitext",
    "wikitext2",
    "cnn_dailymail",
    "dclm"
]

ACC_TASKS = [
    {
        "name": "winogrande",
        "num_fewshot": 0,
        "acc_key": "acc,none",
    },
    {
        "name": "winogrande",
        "num_fewshot": 5,
        "acc_key": "acc,none",
    },
    {
        "name": "arc_easy",
        "num_fewshot": 0,
        "acc_key": "acc_norm,none",
    },
    {
        "name": "arc_challenge",
        "num_fewshot": 0,
        "acc_key": "acc_norm,none",
    },
    {
        "name": "arc_challenge",
        "num_fewshot": 25,
        "acc_key": "acc_norm,none",
    },
    {
        "name": "hellaswag",
        "num_fewshot": 0,
        "acc_key": "acc_norm,none",
    },
    {
        "name": "hellaswag",        
        "num_fewshot": 10,
        "acc_key": "acc_norm,none",
    },
    {
        "name": "truthfulqa_mc1",
        "num_fewshot": 0,
        "acc_key": "acc,none",
    },
    {
        "name": "truthfulqa_mc2",
        "num_fewshot": 0,
        "acc_key": "acc,none",
    },
    {
        "name": "piqa",
        "num_fewshot": 0,
        "acc_key": "acc_norm,none",
    },
    {
        "name": "sciq",
        "num_fewshot": 0,
        "acc_key": "acc,none",
    },
    {
        "name": "boolq",
        "num_fewshot": 0,
        "acc_key": "acc,none",
    },
    {
        "name": "anli_r1",
        "num_fewshot": 0,
        "acc_key": None,
    },
    {
        "name": "anli_r2",
        "num_fewshot": 0,
        "acc_key": None,
    },
    {
        "name": "anli_r3",
        "num_fewshot": 0,
        "acc_key": None,
    },
    {
        "name": "openbookqa",
        "num_fewshot": 0,
        "acc_key": None,
    },
    {
        "name": "rte",
        "num_fewshot": 0,
        "acc_key": None,
    },
    {
        "name": "mmlu",
        "num_fewshot": 0,
        "acc_key": None,
    },
    {
        "name": "mmlu",
        "num_fewshot": 5,
        "acc_key": None,
    },
    {
        "name": "record",
        "num_fewshot": 0,
        "acc_key": None,
    },
]

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_ppl_enc(task, tokenizer, add_special_tokens: bool = True):
    if task == 'wikitext':
        dataset = load_dataset("wikitext", "wikitext-103-v1", split="train", trust_remote_code=True)
        text_column = "text"
        testenc = tokenizer.encode("\n\n".join(dataset[:32768][text_column]), return_tensors='pt', add_special_tokens=add_special_tokens)
    elif task == 'wikitext2':
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train", trust_remote_code=True)
        text_column = "text"
        testenc = tokenizer.encode("\n\n".join(dataset[:32768][text_column]), return_tensors='pt', add_special_tokens=add_special_tokens)
    elif task == 'cnn_dailymail':
        dataset = load_dataset("cnn_dailymail", "3.0.0", split="train", trust_remote_code=True)
        text_column = "article"
        testenc = tokenizer.encode(" ".join(dataset[:16384][text_column]), return_tensors='pt', add_special_tokens=add_special_tokens)
    elif task == 'c4':
        dataset = load_dataset(
            "allenai/c4", 
            data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, 
            split="train", 
            verification_mode="no_checks",
            trust_remote_code=True
        )
        text_column = "text"
        testenc = tokenizer.encode(" ".join(dataset[:8192][text_column]), return_tensors='pt', add_special_tokens=add_special_tokens)
    elif task == 'dclm':
        data_path = "/home/zephyr/gcs-bucket/datasets/dclm/dclm_baseline_1.0.val.jsonl"
        dataset = load_dataset(
            "json",
            data_files={"train": data_path},
            split="train",
            verification_mode="no_checks"
        )
        text_column = "text"
        testenc = tokenizer.encode(" ".join(dataset[:8192][text_column]), return_tensors='pt', add_special_tokens=add_special_tokens)
    else:
        raise NotImplementedError(f"Unsupported task: {task}")
    return testenc

def get_ppl(
    model, 
    tokenizer, 
    tasks,
    batch_size: int = 1,
    calib_size: int = 256,
    max_length: int = 8192,
    add_special_tokens: bool = True,
    task_range: list = []
):
    # devices_in_data_fsdp = model.devices_in_data_fsdp
    # if batch_size % devices_in_data_fsdp != 0:
    #     print(f"🔁 Adjusting batch_size {batch_size} → {devices_in_data_fsdp * ((batch_size + devices_in_data_fsdp - 1) // devices_in_data_fsdp)} for device mesh compatibility.")
    #     batch_size = devices_in_data_fsdp * ((batch_size + devices_in_data_fsdp - 1) // devices_in_data_fsdp)
    if task_range:
        tasks = [t for t in tasks if t in task_range]
    
    ppl_res = {}
    for task in tasks:
        testenc = get_ppl_enc(task, tokenizer, add_special_tokens=add_special_tokens)
        tot_loss = 0
        tot_tokens = 0
        bs = batch_size
        seq_len = max_length
        nsamples = min(testenc.numel() // seq_len, calib_size)
        with torch.no_grad():
            for i in tqdm(range(0, nsamples, bs), desc=f"Evaluating PPL for {task}"):
                j = min(i + bs, nsamples)
                inputs = testenc[:,(i * seq_len):(j * seq_len)]
                inputs = inputs.reshape(j - i, seq_len)
                # import pdb; pdb.set_trace()
                
                outputs = model.forward(inputs)
                if hasattr(outputs, "logits"):
                    lm_logits = outputs.logits
                else:
                    lm_logits = outputs
                
                shift_logits = lm_logits[:, :-1, :].contiguous()
                shift_labels = inputs[:, 1:]
                
                loss_fct = nn.CrossEntropyLoss().to(shift_logits.device)
                loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1))
                
                tot_loss += loss.item() * seq_len * (j - i)
                tot_tokens += seq_len * (j - i)
                
            ppl_res[task] = torch.exp(torch.tensor(tot_loss / tot_tokens)).item()
            print(task, ppl_res[task])
            if task == "dclm":
                print("dclm val loss", math.log(ppl_res[task]))
                
    return ppl_res

def get_acc(model, tokenizer, tasks, task_range=[], limit=1000000):
    # lm_eval_model = models.orbax_lm.HFLM(
    #     pretrained=model, 
    #     tokenizer=tokenizer,
    #     generation_kwargs={
    #         "do_sample": True,
    #         "temperature": 0.2,
    #         "top_p": 0.95,
    #     }
    # )
    if task_range:
        tasks = [cfg for cfg in tasks if cfg["name"] in task_range]
    
    print("tasks to evaluate:")
    print(json.dumps(tasks, indent=2))
    acc_res = {}
    for cfg in tasks:
        task = cfg["name"]
        res = evaluator.simple_evaluate(
            model=model,
            tasks=[task],
            num_fewshot=cfg["num_fewshot"],
            max_batch_size=32,
            log_samples=True,
            # task_kwargs={"limit": 256}, 
            confirm_run_unsafe_code=True,
            limit=limit
        )
        
        print(res['results'][task])
        acc_key = cfg["acc_key"]
        if acc_key is not None:
            acc_res[task] = res['results'][task][acc_key]

    return acc_res

def cast_orbax_state_to_bf16(orbax_state):
    casted_params = jax.tree_util.tree_map(
        lambda x: x.astype(jnp.bfloat16) if hasattr(x, "dtype") and x.dtype == jnp.float32 else x,
        orbax_state.params
    )
    orbax_state = orbax_state.replace(params=casted_params)
    return orbax_state

def main(config, test_args):
    tokenizer = AutoTokenizer.from_pretrained(test_args.hf_model_path)
    
    init_rng = jax.random.PRNGKey(config.init_weights_seed)
    init_rng, rng1 = jax.random.split(init_rng)
    devices_array = maxtext_utils.create_device_mesh(config)
    mesh = jax.sharding.Mesh(devices_array, config.mesh_axes)
    quant = quantizations.configure_quantization(config)
    orbax_model = models.Transformer(config, mesh, quant=quant)
    orbax_state, _ = maxtext_utils.setup_decode_state(orbax_model, config, rng1, mesh, None)
    
    orbax_state = cast_orbax_state_to_bf16(orbax_state)
    
    _, _, state_mesh_shardings = maxtext_utils.get_abstract_state(
        orbax_model, None, config, rng1, mesh, is_training=False
    )

    model = OrbaxLM(orbax_model, orbax_state, tokenizer, config, state_mesh_shardings, mesh)
    
    ppl_res = get_ppl(
        model, 
        tokenizer, 
        # batch_size=config.global_batch_size_to_train_on, 
        batch_size=1,
        calib_size=min(256, test_args.limit),
        max_length=config.max_target_length, 
        tasks=PPL_TASKS,
        add_special_tokens=test_args.add_special_tokens,
        task_range=test_args.tasks,
    )
    print(ppl_res)

    acc_res = get_acc(
        model,
        tokenizer,
        tasks=ACC_TASKS,
        task_range=test_args.tasks,
        limit=test_args.limit
    )
    print(acc_res)
    
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
    parser.add_argument('--add_special_tokens', type=str2bool, default=True)
    parser.add_argument("--limit", type=int, default=1000000)
    parser.add_argument("--tasks", type=lambda x: [] if not x else x.split(","), default=[])
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
        "--add_special_tokens",
        "--limit",
        "--tasks"
    ]
    for arg in to_remove_args:
        model_args = [s for s in model_args if not s.startswith(arg)]

    cfg = pyconfig.initialize(model_args)
    main(cfg, test_args)