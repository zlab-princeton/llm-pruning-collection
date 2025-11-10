import os
import sys

import json
import torch
import torch.nn as nn
import numpy as np 
import random

from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from lm_eval import evaluator, tasks, models

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_ppl_enc(task, tokenizer):
    if task == 'wikitext':
        dataset = load_dataset(
            "wikitext", 
            "wikitext-103-v1", 
            split="train", 
        )
        text_column = "text"
        testenc = tokenizer.encode("\n\n".join(dataset[:131072][text_column]), return_tensors='pt')
    elif task == 'wikitext2':
        dataset = load_dataset(
            "wikitext", 
            "wikitext-2-raw-v1", 
            split="train", 
        )
        text_column = "text"
        testenc = tokenizer.encode("\n\n".join(dataset[:131072][text_column]), return_tensors='pt')
    elif task == 'cnn_dailymail':
        dataset = load_dataset(
            "cnn_dailymail", 
            "3.0.0", 
            split="train", 
        )
        text_column = "article"
        testenc = tokenizer.encode(" ".join(dataset[:8000][text_column]), return_tensors='pt')
    elif task == 'c4':
        dataset = load_dataset(
            "allenai/c4", 
            data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, 
            split="train", 
            verification_mode="no_checks", 
        )
        text_column = "text"
        testenc = tokenizer.encode(" ".join(dataset[:8000][text_column]), return_tensors='pt')
    elif task == 'pg19':
        dataset = load_dataset(
            "emozilla/pg19", 
            split="validation", 
            
        )
        text_column = "text"
        testenc = tokenizer.encode(" ".join(dataset[:8000][text_column]), return_tensors='pt')
    else:
        raise NotImplementedError(f"Unsupported task: {task}")
    return testenc

def get_ppl(
    model, 
    tokenizer, 
    task,
    batch_size: int = 1,
    calib_size: int = 256,
    max_length: int = 8192
):
    testenc = get_ppl_enc(task, tokenizer)
    model.eval()
    tot_loss = 0
    tot_tokens = 0
    bs = batch_size
    seq_len = max_length
    nsamples = min(testenc.numel() // seq_len, calib_size)
    device = model.device
    with torch.no_grad():
        for i in tqdm(range(0, nsamples, bs), desc=f"Evaluating PPL for {task}"):
            j = min(i + bs, nsamples)
            inputs = testenc[:,(i * seq_len):(j * seq_len)].to(device)
            inputs = inputs.reshape(j - i, seq_len)
            
            outputs = model(inputs)
            if hasattr(outputs, "logits"):
                lm_logits = outputs.logits
            else:
                lm_logits = outputs
            
            shift_logits = lm_logits[:, :-1, :].contiguous()
            shift_labels = inputs[:, 1:]
            
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1))
            
            tot_loss += loss.item() * seq_len * (j - i)
            tot_tokens += seq_len * (j - i)
            
        ppl = torch.exp(torch.tensor(tot_loss / tot_tokens)).item()
    
    # print(f"{task} ppl: {ppl}")
    return ppl
        
def get_acc(
    model, 
    tokenizer, 
    task, 
    acc_key,
    num_fewshot=0,
    limit=None,
):
    lm_eval_model = models.huggingface.HFLM(
        pretrained=model, 
        tokenizer=tokenizer,
        # generation_kwargs={
        #     "do_sample": True,
        #     "temperature": 0.2,
        #     "top_p": 0.95,
        # }
    )

    res = evaluator.simple_evaluate(
        model=lm_eval_model,
        tasks=[task],
        num_fewshot=num_fewshot,
        max_batch_size=64,
        log_samples=True,
        confirm_run_unsafe_code=True,
        limit=limit,
    )
    
    acc = res['results'][task][acc_key]
    return acc
    
if __name__ == '__main__':
    pass