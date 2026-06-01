#!/usr/bin/env bash

set -e
export CUDA_VISIBLE_DEVICES=0,1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BEVFORMER_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# This script is the shell equivalent of your PyCharm run configuration.
# Override any of these values at runtime like: GPUS=1 PORT=29704 bash train.sh

PORT="${PORT:-29703}"


BEV_CONFIG="${SCRIPT_DIR}/../configs/bevdiffuser/layout_tiny.py"
BEV_CHECKPOINT="${BEV_CKPT:-/root/autodl-tmp/ckpts/bevformer/bevformer_tiny_epoch_24.pth}"
PRETRAINED_ADMM_DENOISER_CHECKPOINT=None
PRETRAINED_MODEL="CompVis/stable-diffusion-v1-4"

PROJ_NAME="${PROJ_NAME:-admmdiff_stg1_tiny}"
RUN_NAME="${RUN_NAME:-admmdiff_stg1_tiny}"

# mini集 or 全集训练改这一部分内容：
MAX_TRAINING_STEPS=50000
TRAIN_BATCH_SIZE=4
GRADIENT_ACCUMULATION_STEPS=2
CHECKPOINT_STEP=5000
DATALOADER_NUM_WORKERS=8
LR_SCHEDULER=constant
LEARNING_RATE=5e-5
CHECKPOINT_LIMIT=10


UNCOND_PROB=0.2
PREDICTION_TYPE="sample"
TASK_LOSS_SCALE="${TASK_LOSS_SCALE:-0.1}"
CONDITION_MODE="${CONDITION_MODE:-layout}"
NUM_ADMM_ITERS="${NUM_ADMM_ITERS:-4}"
FREEZE_BEV_HEAD_FOR_TASK_LOSS="${FREEZE_BEV_HEAD_FOR_TASK_LOSS:-0}"
CFG_OPTIONS="${CFG_OPTIONS:-}"

RESUME_STEP="${RESUME_STEP:-}"   # 续训传步数, 例 RESUME_STEP=5000; 留空=从头训练
OUTPUT_DIR="${SCRIPT_DIR}/train/${RUN_NAME}"


mkdir -p "${OUTPUT_DIR}"

cd "${BEVFORMER_ROOT}"

echo "GPUS=${GPUS}, PORT=${PORT}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "RESUME_STEP=${RESUME_STEP}"
echo "TASK_LOSS_SCALE=${TASK_LOSS_SCALE}"
echo "CONDITION_MODE=${CONDITION_MODE}"
echo "NUM_ADMM_ITERS=${NUM_ADMM_ITERS}"
echo "FREEZE_BEV_HEAD_FOR_TASK_LOSS=${FREEZE_BEV_HEAD_FOR_TASK_LOSS}"
echo "CFG_OPTIONS=${CFG_OPTIONS}"

EXTRA_ARGS=()
if [[ "${FREEZE_BEV_HEAD_FOR_TASK_LOSS}" == "1" ]]; then
  EXTRA_ARGS+=(--freeze_bev_head_for_task_loss)
fi
if [[ -n "${CFG_OPTIONS}" ]]; then
  # shellcheck disable=SC2206
  CFG_OPTIONS_ARRAY=(${CFG_OPTIONS})
  EXTRA_ARGS+=(--cfg-options "${CFG_OPTIONS_ARRAY[@]}")
fi
if [[ -n "${RESUME_STEP}" ]]; then
  EXTRA_ARGS+=(--resume_from_checkpoint "${OUTPUT_DIR}/checkpoint-${RESUME_STEP}")
fi

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
PYTHONPATH="${SCRIPT_DIR}:${BEVFORMER_ROOT}:${PYTHONPATH:-}" \
python -m torch.distributed.launch --nproc_per_node=2 \
  --master_port="${PORT}" \
  "${SCRIPT_DIR}/train_bev_diffuser.py" \
    --bev_config "${BEV_CONFIG}" \
    --bev_checkpoint "${BEV_CHECKPOINT}" \
    --pretrained_admm_denoiser_checkpoint "${PRETRAINED_ADMM_DENOISER_CHECKPOINT}" \
    --pretrained_model_name_or_path "${PRETRAINED_MODEL}" \
    --train_batch_size "${TRAIN_BATCH_SIZE}" \
    --dataloader_num_workers "${DATALOADER_NUM_WORKERS}" \
    --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
    --max_train_steps "${MAX_TRAINING_STEPS}" \
    --learning_rate "${LEARNING_RATE}" \
    --lr_scheduler "${LR_SCHEDULER}" \
    --output_dir "${OUTPUT_DIR}" \
    --checkpoints_total_limit "${CHECKPOINT_LIMIT}" \
    --checkpointing_steps "${CHECKPOINT_STEP}" \
    --tracker_run_name "${RUN_NAME}" \
    --tracker_project_name "${PROJ_NAME}" \
    --uncond_prob "${UNCOND_PROB}" \
    --condition_mode "${CONDITION_MODE}" \
    --prediction_type "${PREDICTION_TYPE}" \
    --task_loss_scale "${TASK_LOSS_SCALE}" \
    --num_admm_iters "${NUM_ADMM_ITERS}" \
    "${EXTRA_ARGS[@]}"
    # --use_up_down_sample \
    


# 使用方式：
# bash train.sh 2

# 你现在 train.sh 里全集是：
# MAX_TRAINING_STEPS=50000
# TRAIN_BATCH_SIZE=4
# CHECKPOINT_LIMIT=10
# GRADIENT_ACCUMULATION_STEPS=2
# CHECKPOINT_STEP=5000
# DATALOADER_NUM_WORKERS=8
# LR_SCHEDULER=constant
# LEARNING_RATE=1e-4


# mini 第一轮建议改成：
# MAX_TRAINING_STEPS=800
# CHECKPOINT_STEP=200
# CHECKPOINT_LIMIT=5
# DATALOADER_NUM_WORKERS=4
# TRAIN_BATCH_SIZE=4
# GRADIENT_ACCUMULATION_STEPS=2
# LEARNING_RATE=1e-4
# LR_SCHEDULER=constant