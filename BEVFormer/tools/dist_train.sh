#!/usr/bin/env bash

GPUS=$1
PORT=${PORT:-28508}

CONFIG="./projects/configs/diff_bevformer/layout_tiny.py"
HERE="$(cd "$(dirname "$0")" && pwd)"
LOAD_FROM="${BEV_CKPT:-${HERE}/../../ckpts/bevformer_tiny_epoch_24.pth}"
UNET_CHECKPOINT_DIR="${UNET_CKPT_DIR:-${HERE}/../../results/BEVDiffuser_BEVFormer_tiny/checkpoint-50000}"
RESUME_EPOCH="${RESUME_EPOCH:-}"   # 续训传 epoch 数, 例 RESUME_EPOCH=12; 留空=从头训练
RUN_NAME="BEVFormer_tiny_with_BEVDiffuser"
WORK_DIR="$(dirname "$0")/../../results"
RESUME_FROM=None
if [[ -n "${RESUME_EPOCH}" ]]; then
  RESUME_FROM="${WORK_DIR}/epoch_${RESUME_EPOCH}.pth"
fi

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/train.py $CONFIG \
    --launcher pytorch ${@:3} \
    --deterministic \
    --work-dir=$WORK_DIR \
    --report-to='wandb' \
    --tracker-project-name='DiffBEVFormer' \
    --tracker-run-name=$RUN_NAME \
    --unet-checkpoint-dir=$UNET_CHECKPOINT_DIR \
    --load-from=$LOAD_FROM \
    --resume-from=$RESUME_FROM \
