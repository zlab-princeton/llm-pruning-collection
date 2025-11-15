#!/bin/bash

# export TPU_PREFIX=llm-pruning-v6e
required_vars=(
    "BUCKET_NAME"
    "TPU_PREFIX"
)
for var in "${required_vars[@]}"; do
  if [[ -z "${!var:-}" ]]; then
    echo "[ERROR] $var is not set"
    exit 1
  fi
done


export MODEL_NAME='llama3-8b'
export NUM_STEPS=50000
export SEQ_LEN=8192
export BATCH_SIZE=4
export GRAD_ACCUM=1
export LR=3.e-5
export MIN_LR_RATIO=0.1
export WARMUP_RATIO=0.05
export ASYNC_CHECKPOINTING=false
export BASE_OUTPUT_DIRECTORY="gs://$BUCKET_NAME/model_ckpts/maxtext"
export DATA_FILES='/home/zephyr/gcs-bucket/datasets/dclm/llama3_64_arrayrecord/*.array_record'

export RUN_NAME="${MODEL_NAME}_L200_seqlen_${SEQ_LEN}_bs_${BATCH_SIZE}_grad_accum_${GRAD_ACCUM}_lr_${LR}_min_lr_ratio_${MIN_LR_RATIO}_warmup_ratio_${WARMUP_RATIO}"

python -u multihost_runner.py \
    --TPU_PREFIX=$TPU_PREFIX \
    --INTERNAL_IP=true \
    --COMMAND="
    export TPU_LOG_DIR=/home/zephyr/tpu_logs
    sudo docker run \
        --privileged \
        --network=host \
        -v /home/zephyr:/home/zephyr \
        -v /home/zephyr/.config/gcloud:/root/.config/gcloud \
        -v /dev:/dev \
        -v /run:/run \
        -w /home/zephyr/maxtext \
        -e PYTHONPATH=/home/zephyr/maxtext \
        yx3038/maxtext_base_image:latest \
        bash -c \"
        export PYTHONPATH=/home/zephyr/maxtext:\$PYTHONPATH
        python3.10 -u -m MaxText.train MaxText/configs/base.yml \
            run_name=${RUN_NAME} \
            base_output_directory=${BASE_OUTPUT_DIRECTORY} \
            dataset_type=grain \
            grain_train_files=${DATA_FILES} \
            grain_file_type='arrayrecord' \
            grain_worker_count=8 \
            enable_data_shuffling=False \
            grain_worker_count_eval=1 \
            tokenize_train_data=False \
            tokenize_eval_data=False \
            max_target_length=${SEQ_LEN} \
            max_position_embeddings=${SEQ_LEN} \
            original_max_position_embeddings=${SEQ_LEN} \
            async_checkpointing=${ASYNC_CHECKPOINTING} \
            model_name=${MODEL_NAME} \
            steps=${NUM_STEPS} \
            per_device_batch_size=${BATCH_SIZE} \
            gradient_accumulation_steps=${GRAD_ACCUM} \
            learning_rate=${LR} \
            cosine_learning_rate_final_fraction=${MIN_LR_RATIO} \
            warmup_steps_fraction=${WARMUP_RATIO} \
            checkpoint_period=500 \
            checkpoint_max_to_keep=1 \
            use_wandb=True \
            wandb_project=llm_pruning \
            wandb_run_name=${TPU_PREFIX}_${RUN_NAME} \
            packing=false \
            jax_distributed_initialization_timeout=900
        \"
    "

# metrics_file=\"/home/zephyr/maxtext/logs/${RUN_NAME}.log\" \

# python3 -m MaxText.train \
#     MaxText/configs/base.yml \
#     run_name=runner_pretraining_${idx}\
#      base_output_directory=${BASE_OUTPUT_DIRECTORY} \
#      dataset_path=${DATASET_PATH} \
#      async_checkpointing=${ASYNC_CHECKPOINTING} \
#      per_device_batch_size=1 \
#      model_name='llama2-7b' \
#      ici_context_parallelism=4 \
#      steps=10 \
#      per_device_batch_size=1 \
#      packing=false
