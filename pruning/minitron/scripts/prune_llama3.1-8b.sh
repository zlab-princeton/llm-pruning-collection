#!/bin/bash

#SBATCH --job-name=prune_llama3.1-8b_%j
#SBATCH --output=logs/prune_llama3.1-8b_%j.out
#SBATCH --error=logs/prune_llama3.1-8b_%j.err

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

#SBATCH --cpus-per-task=16
#SBATCH --mem=128GB
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:1

#SBATCH --mail-type=all
#SBATCH --mail-user=yx1168@princeton.edu

set -euo pipefail

source /usr/local/anaconda3/2024.02/etc/profile.d/conda.sh
conda activate minitron

PROJ_DIR="$(dirname $(realpath "${BASH_SOURCE[0]}"))/.."

model_path=meta-llama/Llama-3.1-8B
# model_path=/n/fs/vision-mix/yx1168/model_ckpts/Llama-3.1-8B
save_dir=${PROJ_DIR}/../../checkpoints
log_dir=${PROJ_DIR}/outputs
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

python $PROJ_DIR/main.py depth \
    --model_path ${model_path} \
    --save_dir ${save_dir} \
    --log_dir ${log_dir} \
    --prune_task winogrande \
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