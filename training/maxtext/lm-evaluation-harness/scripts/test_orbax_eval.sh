#!/bin/bash

set +x
set -eo pipefail

# export MODEL='llama3.1-8b'
export MODEL='llama3.1-1.5b-depth'
export bucket_name=llm_pruning_us_central2_b
export BASE_OUTPUT_DIRECTORY="gs://$bucket_name/model_ckpts/maxtext"
export DIRECT_PARAMETER_CHECKPOINT_RUN="direct_generate_param_only_checkpoint_${MODEL}"

export HF_MODEL_PATH='/home/zephyr/gcs-bucket/model_ckpts/Llama-3.1-8B'

export CONVERTED_CHECKPOINT_PATH="gs://$bucket_name/model_ckpts/maxtext/${MODEL}"

export CONVERTED_CHECKPOINT="gs://$bucket_name/model_ckpts/maxtext/llama3.1-1.5b-depth_S50_seqlen_8192_bs_4_grad_accum_4_lr_3.e-4_min_lr_ratio_0.1_warmup_ratio_0.05_test2/checkpoints/12499/items"

export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export JAX_DISABLE_MOST_OPTIMIZATIONS=False

export DIRECT_PARAMETER_CHECKPOINT_RUN="direct_generate_param_only_checkpoint_${MODEL}_lr_3e-4_no_q_scale"
export UNSCANNED_CKPT_PATH="${BASE_OUTPUT_DIRECTORY}/${DIRECT_PARAMETER_CHECKPOINT_RUN}/checkpoints/0/items"

cd ..
export PYTHONPATH=$(pwd):$PYTHONPATH
python3 -m MaxText.generate_param_only_checkpoint \
    MaxText/configs/base.yml \
    checkpoint_dir=${BASE_OUTPUT_DIRECTORY} \
    base_output_directory=${BASE_OUTPUT_DIRECTORY} \
    load_parameters_path=${CONVERTED_CHECKPOINT} \
    run_name=${DIRECT_PARAMETER_CHECKPOINT_RUN} \
    model_name=$MODEL \
    force_unroll=true

cd lm-evaluation-harness
export PYTHONPATH=$(pwd):$PYTHONPATH
python3 -u scripts/test_orbax_eval.py \
    ../MaxText/configs/base.yml \
    load_parameters_path=${UNSCANNED_CKPT_PATH} \
    run_name=forward_pass_test \
    per_device_batch_size=1 \
    model_name=${MODEL} \
    max_prefill_predict_length=4 \
    max_target_length=8192 \
    dataset_type=synthetic \
    dtype=bfloat16 \
    scan_layers=false \
    attention="dot_product" \
    --hf_model_path=${HF_MODEL_PATH} \
    --add_special_tokens=False

# decode example
# idx=0
# TOKENIZER='/home/zephyr/maxtext/assets/tokenizer_llama3.tiktoken'
# python3 -m MaxText.decode \
#     /home/zephyr/gcs-bucket/maxtext/MaxText/configs/base.yml \
#     load_parameters_path=${UNSCANNED_CKPT_PATH} \
#     tokenizer_type=tiktoken \
#     tokenizer_path=$TOKENIZER \
#     per_device_batch_size=1 \
#     run_name=runner_$(date +%Y-%m-%d-%H-%M) \
#     max_prefill_predict_length=4 \
#     max_target_length=16 \
#     model_name=$MODEL \
#     dataset_type=synthetic \
#     async_checkpointing=false \
#     scan_layers=false \
#     attention=dot_product \
#     prompt="I love to" 

# python3 -m MaxText.decode MaxText/configs/base.yml tokenizer_path=assets/tokenizer_llama3.tiktoken tokenizer_type=tiktoken load_parameters_path=${UNSCANNED_CKPT_PATH} per_device_batch_size=1 run_name=runner_$(date +%Y-%m-%d-%H-%M) max_prefill_predict_length=4 max_target_length=16 dataset_type=synthetic async_checkpointing=false scan_layers=false model_name=${MODEL_VARIATION} attention=dot_product prompt="I love to"
