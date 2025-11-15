#!/bin/bash

set +x
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <mode> [--model=MODEL] [--orbax_ckpt_path=ORBAX_CKPT_NAME] [--step=STEP] [--hf_model_path=HF_MODEL_NAME] [--direct_run_name=DIRECT_RUN_NAME]"
  echo "Modes: hf_to_orbax | gen_param_ckpt | orbax_to_hf | logits_test | eval"
  echo "Run $0 help for more details."
  exit 1
fi
MODE=$1
shift
for arg in "$@"; do
  case $arg in
    --model=*) MODEL="${arg#*=}" ;;
    --orbax_ckpt_name=*) ORBAX_CKPT_NAME="${arg#*=}" ;;
    --step=*) STEP="${arg#*=}" ;;
    --hf_model_name=*) HF_MODEL_NAME="${arg#*=}" ;;
    --direct_run_name=*) DIRECT_RUN_NAME="${arg#*=}" ;;
    --tasks=*) TASKS="${arg#*=}" ;;
    *) echo "[WARN] Unknown arg $arg" ;;
  esac
done

if [[ $MODE == "help" ]]; then
  echo "$0 hf_to_orbax     --model=MODEL --orbax_ckpt_name=ORBAX_CKPT_NAME --hf_model_name=HF_MODEL_NAME"
  echo "$0 gen_param_ckpt  --model=MODEL --orbax_ckpt_name=ORBAX_CKPT_NAME --step=STEP                    --direct_run_name=DIRECT_RUN_NAME"
  echo "$0 orbax_to_hf     --model=MODEL --orbax_ckpt_name=ORBAX_CKPT_NAME --step=STEP                    --hf_model_name=HF_MODEL_NAME"
  echo "$0 logits_test     --model=MODEL --direct_run_name=DIRECT_RUN_NAME --hf_model_name=HF_MODEL_NAME"
  echo "$0 eval            --model=MODEL --direct_run_name=DIRECT_RUN_NAME --hf_model_name=HF_MODEL_NAME [--tasks=TASKS]"
  exit 0
fi

source scripts/get_tpu_info.sh

export BUCKET_NAME="$(get_bucket_name)"
export TPU_PREFIX="$(get_tpu_name)"

### ====== CONFIG ======
# place to save the maxtext ckpts
export ORBAX_CKPT_DIR="gs://${BUCKET_NAME}/model_ckpts/maxtext" # directory of orbax checkpoints
export STEP="${STEP:-0}"
export DIRECT_CKPT_DIR="gs://${BUCKET_NAME}/model_ckpts/direct" # directory of param-only orbax checkpoints
export HF_CKPT_DIR="/home/zephyr/gcs-bucket/model_ckpts/hf" # directory of Hugging Face checkpoints
export PYTHONPATH="$(pwd)":${PYTHONPATH:-''}

case "$MODE" in
  hf_to_orbax)
    echo "[INFO] 🚀 Converting Hugging Face → Orbax..."
    export HF_MODEL_PATH="${HF_CKPT_DIR}/${HF_MODEL_NAME}"
    export CONVERTED_CHECKPOINT_PATH="${ORBAX_CKPT_DIR}/${ORBAX_CKPT_NAME}/checkpoints"
    JAX_PLATFORMS=cpu python3 -m MaxText.llama_or_mistral_ckpt \
      --base-model-path ${HF_MODEL_PATH} \
      --huggingface-checkpoint True \
      --model-size $MODEL \
      --maxtext-model-path ${CONVERTED_CHECKPOINT_PATH} \
      --huggingface-checkpoint=True
    ;;

  gen_param_ckpt)
    echo "[INFO] 🧩 Generating parameter-only checkpoint..."
    export CONVERTED_CHECKPOINT="${ORBAX_CKPT_DIR}/${ORBAX_CKPT_NAME}/checkpoints/${STEP}/items"
    JAX_PLATFORMS=cpu python3 -m MaxText.generate_param_only_checkpoint \
      MaxText/configs/base.yml \
      skip_jax_distributed_system=True \
      checkpoint_dir=${ORBAX_CKPT_DIR} \
      base_output_directory=${DIRECT_CKPT_DIR} \
      load_parameters_path=${CONVERTED_CHECKPOINT} \
      run_name=${DIRECT_RUN_NAME} \
      model_name=$MODEL \
      force_unroll=true
    ;;

  orbax_to_hf)
    echo "[INFO] 🔁 Converting Orbax → Hugging Face..."
    export HF_MODEL_PATH="${HF_CKPT_DIR}/${HF_MODEL_NAME}"
    export CONVERTED_CHECKPOINT="${ORBAX_CKPT_DIR}/${ORBAX_CKPT_NAME}/checkpoints/${STEP}/items"
    JAX_PLATFORMS=cpu python3 -m MaxText.llama_mistral_mixtral_orbax_to_hf \
      MaxText/configs/base.yml \
      skip_jax_distributed_system=True \
      base_output_directory=${HF_CKPT_DIR} \
      load_parameters_path=${CONVERTED_CHECKPOINT} \
      run_name=convert_to_hf \
      model_name=${MODEL} \
      hf_model_path=${HF_MODEL_PATH}
    ;;

  logits_test)
    echo "[INFO] 🧪 Running forward pass equivalence test..."
    export HF_MODEL_PATH="${HF_CKPT_DIR}/${HF_MODEL_NAME}"
    export UNSCANNED_CKPT_PATH="${DIRECT_CKPT_DIR}/${DIRECT_RUN_NAME}/checkpoints/0/items"
    # TPU_CHIPS_PER_HOST_BOUNDS=1,1,1 \
    # TPU_HOST_BOUNDS=1,1,1 \
    # TPU_VISIBLE_DEVICES=0,1,2,3 \
    # XLA_USE_BF16=1 \
    JAX_PLATFORMS=cpu python3 -u tests/test_eq.py \
      MaxText/configs/base.yml \
      skip_jax_distributed_system=True \
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
    ;;

  eval)
    echo "[INFO] 🧪 Running evaluation..."
    export HF_MODEL_PATH="${HF_CKPT_DIR}/${HF_MODEL_NAME}"
    export UNSCANNED_CKPT_PATH="${DIRECT_CKPT_DIR}/${DIRECT_RUN_NAME}/checkpoints/0/items"
    cd lm-evaluation-harness
    TPU_CHIPS_PER_HOST_BOUNDS=1,1,1 \
    TPU_HOST_BOUNDS=1,1,1 \
    TPU_VISIBLE_DEVICES=0,1,2,3 \
    XLA_USE_BF16=1 \
    python3 -u scripts/test_orbax_eval.py \
      ../MaxText/configs/base.yml \
      skip_jax_distributed_system=True \
      load_parameters_path=${UNSCANNED_CKPT_PATH} \
      run_name=forward_pass_test \
      per_device_batch_size=1 \
      model_name=${MODEL} \
      max_prefill_predict_length=4 \
      max_target_length=8192 \
      dataset_type=synthetic \
      attention="dot_product" \
      dtype=bfloat16 \
      scan_layers=false \
      --hf_model_path=${HF_MODEL_PATH} \
      --tasks=${TASKS:-""}

    cd ..
    ;;

  weights_test)
    echo "[INFO] 🧪 Running weights test..."
    export HF_MODEL_PATH="${HF_CKPT_DIR}/${HF_MODEL_NAME}"
    export UNSCANNED_CKPT_PATH="${DIRECT_CKPT_DIR}/${DIRECT_RUN_NAME}/checkpoints/0/items"
    JAX_PLATFORMS=cpu python3 -u tests/test_weights.py \
      MaxText/configs/base.yml \
      skip_jax_distributed_system=True \
      load_parameters_path=${UNSCANNED_CKPT_PATH} \
      run_name=forward_pass_test per_device_batch_size=1 \
      model_name=${MODEL} \
      max_prefill_predict_length=4 \
      max_target_length=4 \
      dataset_type=synthetic \
      dtype=bfloat16 \
      scan_layers=false \
      --hf_model_path=${HF_MODEL_PATH}
    ;;

  *)
    echo "[ERROR] Unknown mode: $MODE"
    exit 1
    ;;
esac