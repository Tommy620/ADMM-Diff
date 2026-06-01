# Prerequisites

**Please ensure you have prepared the environment and the nuScenes dataset.**

# Train and Test

Open your bash:
```
# This step is for setting path of dataset, pkl, and ckpt of bev model, so make sure that you've set the env.sh first. And then you can run:
cd /root/autodl-tmp/ADMM-Diff (path of your project)
source env.sh

```

Train stage 1 of ADMM-Diff(For training teacher model):
```
cd /root/autodl-tmp/ADMM-Diff/BEVFormer/projects/bevdiffuser

# 从头训练
bash train.sh

# 续训（传步数即可，ckpt 在 train/<RUN_NAME>/checkpoint-<step>）
RESUME_STEP=5000 bash train.sh
```

Eval stage 1 of ADMM-Diff(For checking the performance of the teacher):
```
cd /root/autodl-tmp/ADMM-Diff/BEVFormer/projects/bevdiffuser

# 测试（传步数，默认 RUN_NAME=admmdiff_stg1_tiny）
STEP=50000 bash test.sh
```

Train stage 2 of ADMM-Diff(For training student model with teacher frozen):
```
cd /root/autodl-tmp/ADMM-Diff/BEVFormer

# 从头训练（2 卡）；如老师 ckpt 不在默认位置，再传 UNET_CKPT_DIR
bash tools/dist_train.sh 2
UNET_CKPT_DIR=/path/to/stg1/checkpoint-50000 bash tools/dist_train.sh 2
# 续训（传 epoch，ckpt 在 results/epoch_<N>.pth）
RESUME_EPOCH=12 bash tools/dist_train.sh 2
```

Eval stage 2 of ADMM-Diff(For checking the performance of the student):
```
cd /root/autodl-tmp/ADMM-Diff/BEVFormer

# 测试（位置参数：config 完整ckpt gpus）
bash tools/dist_test.sh ./projects/configs/diff_bevformer/layout_tiny.py results/epoch_24.pth 2
```