set -e

export CUDA_VISIBLE_DEVICES=0,1
export HF_ENDPOINT=https://hf-mirror.com

BEV_CONFIG="/root/autodl-tmp/ADMM-Diff/BEVFormer/projects/configs/bevdiffuser/layout_tiny.py"

CHECKPOINT_DIR="/root/autodl-tmp/ADMM-Diff/BEVFormer/projects/bevdiffuser/train/admmdiff_stg1_tiny-resume-35000/checkpoint-50000"

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


