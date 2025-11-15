#!/bin/bash

set -euo pipefail

PROJ_DIR=$(pwd)

source $(conda info --base)/etc/profile.d/conda.sh
conda activate fms_fsdp
export PYTHONPATH=$PROJ_DIR/..:${PYTHONPATH:-''}

ckpt_dir=$PROJ_DIR/../../checkpoints
model_variant='llama3_8b'
fms_Path="${ckpt_dir}/fms/${model_variant}_fms.pth"
hf_path="${ckpt_dir}/hf/${model_variant}_hf"

python training/eq_test.py \
    --model_variant ${model_variant} \
    --hf_path ${hf_path} \
    --fms_path ${fms_Path} 