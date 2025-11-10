#!/bin/bash

set -euo pipefail

PROJ_DIR="$(dirname $(realpath "${BASH_SOURCE[0]}"))"/..

# model_path=meta-llama/Llama-3.1-8B
model_path=/n/fs/vision-mix/yx1168/model_ckpts/Llama-3.1-8B
save_dir=${PROJ_DIR}/../../checkpoints
log_dir=${PROJ_DIR}/logs
calib_size=32

python $PROJ_DIR/main.py bi \
    --model_path ${model_path} \
    --save_dir ${save_dir} \
    --log_dir ${log_dir} \
    --prune_task wikitext \
    --num_layers 16 \
    --calib_size ${calib_size}

python $PROJ_DIR/main.py depth \
    --model_path ${model_path} \
    --save_dir ${save_dir} \
    --log_dir ${log_dir} \
    --prune_task wikitext \
    --num_layers 16 \
    --calib_size ${calib_size}

python $PROJ_DIR/main.py width \
    --model_path ${model_path} \
    --save_dir ${save_dir} \
    --log_dir ${log_dir} \
    --prune_task wikitext \
    --hidden_size 3072 \
    --ffn_hidden_size 9216 \
    --calib_size ${calib_size}