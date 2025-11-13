# Sheared Llama

This is an organized version of the officisal implementation of Sheared Llama by Xia et al. Specifically, we fixed the [incompatibility with flash-attn2](https://github.com/princeton-nlp/LLM-Shearing?tab=readme-ov-file#install-requirements) and [loss NaN issue](https://github.com/princeton-nlp/LLM-Shearing/issues/34) in the original release.

## Installation 
```bash
bash scripts/install.sh
```

## Prepare Composer Model
```bash
bash scripts/hf2composer.sh
```

## Data Preparation
Please download the data from [here](https://huggingface.co/datasets/Zephyr271828/redpajama-for-prune) to `llmshearing/data/red_pajama/for_prune`. You should have file layout as follows. `redpajama/for_prune` is what you need to prune the model and update the weights.
```
.
├── count_tokens.py
├── get_all_jsonl.py
├── llama_tokenizer.py
├── merge_data.py
├── README.md
├── redpajama
│   └── for_prune
│       ├── arxiv
│       ├── book
│       ├── c4-rp
│       ├── cc
│       ├── eval
│       ├── eval_merge
│       ├── github
│       ├── stackexchange
│       └── wiki
├── sample_all_domains.sh
├── sample.py
├── sample_redpajama
│   └── ...
├── split_jsonl.py
├── tokenize_all_files.sh
├── tokenizer.model
└── tokenize_single_file.py
```
For a detailed and customized data preparation guide, please see [here](llmshearing/data/README.md).

## Pruning
```bash
bash scripts/prune_llama2-2.7b.sh
bash scripts/prune_llama2-1.3b.sh
bash scripts/prune_llama2-370m.sh
```

## Convert back to HF
```bash
bash scripts/composer2hf.sh
```

## Results

To reproduce the results after pruning and before continual training, we downloadd the officially released [1.3B](https://huggingface.co/princeton-nlp/Sheared-LLaMA-1.3B-Pruned) and [2.7B](https://huggingface.co/princeton-nlp/Sheared-LLaMA-2.7B-Pruned) checkpoints from HF and evaluated the checkpoints with lm-eval-harness.
| Size | Source   | BoolQ | PIQA | Winogrande | ARC-C | ARC-E | Hellaswag | 
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| 2.7B | Released | 84.5 | 66.4 | 53.2 | 26.5 | 49.9 | 47.1 |
| 2.7B | Tested   | 84.2 | 66.2 | 55.9 | 28.2 | 52.8 | 46.9 |
| 1.3B | Released | 77.5 | 62.6 | 50.3 | 19.5 | 41.0 | 34.8 |
| 1.3B | Tested   | 77.8 | 60.5 | 51.0 | 18.4 | 41.8 | 34.1 |