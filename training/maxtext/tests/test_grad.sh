

export MODEL='llama3.1-4b-depth'
export HF_MODEL_PATH='/home/zephyr/gcs-bucket/model_ckpts/llama3-4b-depth-fms-to-hf'
export UNSCANNED_CKPT_PATH='/home/zephyr/gcs-bucket/model_ckpts/maxtext/direct_generate_param_only_checkpoint_llama3-4b-depth_from_fms/checkpoints/0/items'

export PYTHONPATH=$(pwd):$PYTHONPATH
python tests/test_grad.py \
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