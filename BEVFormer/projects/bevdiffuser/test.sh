set -e

# 用法：
# cd /root/autodl-tmp/ADMM-Diff/BEVFormer/projects/bevdiffuser
# STEP=50000 bash test.sh
source /root/autodl-tmp/ADMM-Diff/env.sh
export CUDA_VISIBLE_DEVICES=0,1
export HF_ENDPOINT=https://hf-mirror.com

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

BEV_CONFIG="${SCRIPT_DIR}/../configs/bevdiffuser/layout_tiny.py"

RUN_NAME="${RUN_NAME:-admmdiff_ori_resume_70k}"
STEP="${STEP:-70000}"
CHECKPOINT_DIR="${SCRIPT_DIR}/train/${RUN_NAME}/checkpoint-${STEP}"

BEV_CHECKPOINT="${CHECKPOINT_DIR}/bev_model.pth"

PREDICTION_TYPE="sample"
PRETRAINED_MODEL="CompVis/stable-diffusion-v1-4"
NUM_ADMM_ITERS="${NUM_ADMM_ITERS:-4}"

python -m torch.distributed.launch --nproc_per_node=2 --master_port 10000 test_bev_diffuser.py \
    --bev_config $BEV_CONFIG \
    --bev_checkpoint $BEV_CHECKPOINT \
    --checkpoint_dir $CHECKPOINT_DIR \
    --pretrained_model_name_or_path $PRETRAINED_MODEL \
    --prediction_type $PREDICTION_TYPE \
    --noise_timesteps 5 \
    --denoise_timesteps 5 \
    --num_inference_steps 5 \
    --num_admm_iters "${NUM_ADMM_ITERS}" \
    # --launcher pytorch
    # --use_classifier_guidence \
    # --use_up_down_sample \


