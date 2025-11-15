#!/bin/bash

set +x
set -eo pipefail

export bucket_name=llm_pruning_us_central2_b

export DATASET_PATH='/home/zephyr/gcs-bucket/datasets/'

# export MODEL='llama3.1-8b'
# export HF_MODEL_PATH='/home/zephyr/gcs-bucket/model_ckpts/Llama-3.1-8B'

# export MODEL='llama2-7b'
export MODEL="llama3.1-4b-width"
# export HF_MODEL_PATH='/home/zephyr/gcs-bucket/model_ckpts/Llama-2-7b-hf'
export HF_MODEL_PATH="/home/zephyr/gcs-bucket/model_ckpts/maxtext/llama3.1_minitron_width_HF"

export BASE_OUTPUT_DIRECTORY="gs://$bucket_name/model_ckpts/maxtext"
export PYTHONPATH='/home/zephyr/maxtext':$PYTHONPATH

# export CONVERTED_CHECKPOINT_PATH="gs://$bucket_name/model_ckpts/maxtext/${MODEL}"
export CONVERTED_CHECKPOINT_PATH="/home/zephyr/gcs-bucket/model_ckpts/maxtext/llama3.1-4b-width-orbax"
export CONVERTED_CHECKPOINT="${CONVERTED_CHECKPOINT_PATH}/0/items"
export DIRECT_PARAMETER_CHECKPOINT_RUN="direct_generate_param_only_checkpoint_${MODEL}"
export UNSCANNED_CKPT_PATH="${BASE_OUTPUT_DIRECTORY}/${DIRECT_PARAMETER_CHECKPOINT_RUN}/checkpoints/0/items"


# HF TO ORBAX
# python3 -m MaxText.llama_or_mistral_ckpt \
#     --base-model-path ${HF_MODEL_PATH} \
#     --huggingface-checkpoint True \
#     --model-size $MODEL \
#     --maxtext-model-path ${CONVERTED_CHECKPOINT_PATH}

python3 -m MaxText.generate_param_only_checkpoint \
    MaxText/configs/base.yml \
    checkpoint_dir=${BASE_OUTPUT_DIRECTORY} \
    base_output_directory=${BASE_OUTPUT_DIRECTORY} \
    load_parameters_path=${CONVERTED_CHECKPOINT} \
    run_name=${DIRECT_PARAMETER_CHECKPOINT_RUN} \
    model_name=$MODEL \
    force_unroll=true

# ORBAX TO HF
# JAX_PLATFORMS=cpu python3 -m MaxText.llama_mistral_mixtral_orbax_to_hf \
#     MaxText/configs/base.yml \
#     base_output_directory=${BASE_OUTPUT_DIRECTORY} \
#     load_parameters_path=${CONVERTED_CHECKPOINT} \
#     run_name=convert_to_hf \
#     model_name=${MODEL} \
#     hf_model_path=/home/zephyr/gcs-bucket/model_ckpts/${MODEL}-hf

# LOGITS OUT TEST
JAX_PLATFORMS=tpu python3 -u tests/test_eq.py \
    MaxText/configs/base.yml \
    load_parameters_path=${UNSCANNED_CKPT_PATH} \
    run_name=forward_pass_test \
    per_device_batch_size=1 \
    model_name=${MODEL} \
    max_prefill_predict_length=4 \
    max_target_length=4 \
    dataset_type=synthetic \
    dtype=bfloat16 \
    scan_layers=false \
    --run_hf_model=True \
    --hf_model_path=${HF_MODEL_PATH}

