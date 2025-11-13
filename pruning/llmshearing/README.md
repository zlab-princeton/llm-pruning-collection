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
├── merge_data.py
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
├── sample.py
├── sample_redpajama
│   ├── ...
├── split_jsonl.py
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

Llama-2-2.7B
| Source | |
|:--:|:--:|
| | |

Llama-2-1.3B
| Source | |
|:--:|:--:|
| | |