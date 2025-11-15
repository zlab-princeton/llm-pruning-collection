#!/bin/bash

#!/bin/bash

set +x
set -eo pipefail

export bucket_name=llm_pruning_us_central2_b

export DATASET_PATH='/home/zephyr/gcs-bucket/datasets/'

# export MODEL='llama3.1-8b'
# export HF_MODEL_PATH='/home/zephyr/gcs-bucket/model_ckpts/Llama-3.1-8B'

export MODEL='llama2-7b'
export HF_MODEL_PATH='/home/zephyr/gcs-bucket/model_ckpts/Llama-2-7b-hf'

export BASE_OUTPUT_DIRECTORY="gs://$bucket_name/model_ckpts/maxtext"
export PYTHONPATH='/home/zephyr/maxtext':$PYTHONPATH

export CONVERTED_CHECKPOINT_PATH="gs://$bucket_name/model_ckpts/maxtext/${MODEL}"
export CONVERTED_CHECKPOINT="${CONVERTED_CHECKPOINT_PATH}/0/items"
export DIRECT_PARAMETER_CHECKPOINT_RUN="direct_generate_param_only_checkpoint_${MODEL}"
export UNSCANNED_CKPT_PATH="${BASE_OUTPUT_DIRECTORY}/${DIRECT_PARAMETER_CHECKPOINT_RUN}/checkpoints/0/items"

export PYTHONPATH='/home/zephyr/maxtext':$PYTHONPATH
python3 -u tests/test_weights.py \
    MaxText/configs/base.yml \
    load_parameters_path=${UNSCANNED_CKPT_PATH} \
    run_name=forward_pass_test per_device_batch_size=1 \
    model_name=${MODEL} \
    max_prefill_predict_length=4 \
    max_target_length=4 \
    dataset_type=synthetic \
    dtype=bfloat16 \
    scan_layers=false \
    --hf_model_path=${HF_MODEL_PATH}
