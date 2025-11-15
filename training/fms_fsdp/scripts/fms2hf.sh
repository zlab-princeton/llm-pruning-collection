#!/bin/bash

set -euo pipefail

PROJ_DIR=$(pwd)

source $(conda info --base)/etc/profile.d/conda.sh
conda activate fms_fsdp
export PYTHONPATH=$PROJ_DIR/..:${PYTHONPATH:-''}

PROJ_DIR=$(pwd)
ckpt_dir=$PROJ_DIR/../../checkpoints
model_variant='llama3_8b'
load_path="${ckpt_dir}/fms/${model_variant}_fms.pth"
tokenizer_path="/n/fs/vision-mix/yx1168/model_ckpts/Llama-3.1-8B"
save_dir="${ckpt_dir}/hf"
mkdir -p ${save_dir}

python training/fms2hf.py \
    --model_variant ${model_variant} \
    --load_path ${load_path} \
    --save_path ${save_dir}/${model_variant}_hf \
    --tokenizer_name_or_path ${tokenizer_path}