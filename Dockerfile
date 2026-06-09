# ============================================================
# ADMM-Diff / BEVFormer 训练与推理环境（容器化）
#
# 设计原则：
#   1. 镜像只装"环境"，工程代码 / nuScenes 数据集 / ckpts 在 docker run 时挂载进来。
#   2. 所有版本锁死，复刻本地可运行的 torch1.10+cu111 栈。
#   3. mmcv-full、detectron2 用官方预编译 wheel（免编译）；只有 mmdet3d 现场编译。
#
# 构建：  docker build -t admm-diff:cu111 .
# 运行：  见 README.md 的 "用 Docker 一键复现" 一节
# ============================================================

# 基础镜像：自带 CUDA 11.1.1 工具链 + cuDNN8 + Ubuntu 20.04（系统默认 python3.8）。
# 必须用 devel（开发版）而不是 runtime，因为 devel 才带 nvcc，能编译 mmdet3d 的 CUDA 算子。
FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu20.04

# DEBIAN_FRONTEND=noninteractive：apt 安装时不弹交互式提问，否则 build 会卡住。
# TORCH_CUDA_ARCH_LIST：告诉编译器把算子编译成哪些 GPU 架构（一次编译多架构 = "胖二进制"）。
#   每列一个数字会生成对应显卡的"成品机器码"(cubin)；末尾 "+PTX" 额外附带一份"中间码"(PTX)，
#   比所列最高架构更新的显卡运行时由驱动 JIT 现场转译。这样同一个镜像可通吃多种卡：
#     7.5 = 图灵 (RTX 2080 / T4)
#     8.0 = 安培数据中心 (A100)
#     8.6 = 安培消费卡 (RTX 3090 / 3080)
#     8.9 = Ada (RTX 4090)：CUDA 11.1 的 nvcc 不直接支持 sm_89，靠 8.6 的 PTX JIT 转译（与本地一致）
#   注意：CUDA 11.1 这个工具链最高只能原生编译到 sm_86；比 4090 更新的卡需更高 CUDA 底座重建镜像。
# FORCE_CUDA=1：强制按 CUDA 模式编译算子（构建时容器内看不到物理 GPU，需要这个开关）。
ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    FORCE_CUDA=1 \
    TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6+PTX"

# ---------- 1) 系统依赖 ----------
# build-essential/ninja：编译 mmdet3d 算子；git/wget：拉源码；
# libgl1/libglib2.0：opencv-python 运行所需的系统图形库。
RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget ninja-build build-essential \
    python3.8 python3.8-dev python3-pip \
    libgl1-mesa-glx libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 让 python / pip 指向 3.8；setuptools 必须锁 59.5.0，否则与 torch1.10 的旧式 setup 冲突。
RUN ln -sf /usr/bin/python3.8 /usr/bin/python && \
    python -m pip install --upgrade "pip<24" "setuptools==59.5.0" wheel

# ---------- 2) PyTorch（cu111，与本地完全一致）----------
RUN pip install torch==1.10.0+cu111 torchvision==0.11.1+cu111 \
    -f https://download.pytorch.org/whl/torch_stable.html

# ---------- 3) 纯 Python 依赖 ----------
# 先装好这些（含锁死的 numpy/numba），避免后面的包把 numpy 升级到不兼容版本。
COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

# ---------- 4) OpenMMLab ----------
# mmcv-full 用官方 cu111/torch1.10 预编译 wheel（免编译，快且稳）；mmdet/mmseg 普通安装。
RUN pip install mmcv-full==1.4.0 \
      -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.10.0/index.html && \
    pip install mmdet==2.14.0 mmsegmentation==0.14.1

# ---------- 5) detectron2（cu111/torch1.10 预编译 wheel，免编译）----------
RUN pip install detectron2==0.6 \
    -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.10/index.html

# ---------- 6) mmdet3d v0.17.1（从源码编译，锁定与本地相同的 commit）----------
# 这是唯一需要现场编译 CUDA 算子的包，耗时最长。
RUN git clone https://github.com/open-mmlab/mmdetection3d.git /opt/mmdetection3d && \
    cd /opt/mmdetection3d && \
    git checkout f1107977dfd26155fc1f83779ee6535d2468f449 && \
    pip install -v -e .

# 工作目录：docker run 时把本工程挂载到这里。
WORKDIR /workspace/ADMM-Diff

# 默认进入 bash，方便交互式跑训练 / 测试脚本。
CMD ["/bin/bash"]
