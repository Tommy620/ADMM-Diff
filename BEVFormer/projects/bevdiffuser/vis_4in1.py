import os

os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"

import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def load_bev_image(path):
    """加载BEV图片"""
    try:
        img = Image.open(path)
        return np.array(img)
    except FileNotFoundError:
        print(f"警告: 图片不存在 {path}")
        # 返回一个空白图片
        return np.zeros((224, 224, 3), dtype=np.uint8)


def load_vis_tensor_from_pt(pt_path, token):
    """从.pt文件中加载指定token的特征张量"""
    vis_dict = torch.load(pt_path, map_location="cpu")
    if token not in vis_dict:
        print(f"警告: token {token} 不在 {pt_path} 中")
        return None

    vis_ls = vis_dict[token]  # list of tensors, each shape [1, 256, 50, 50]

    # 取第一个张量，去除batch维度，shape变为[1, 256, 50, 50]
    feat = vis_ls[0]

    # 转换为numpy
    if isinstance(feat, torch.Tensor):
        feat = feat.numpy()

    # 通道维度平均 [1, C, H, W] -> [H, W]
    feat_mean = np.mean(feat[0], axis=0)  # 在通道维度上平均

    # 上下翻转180度
    feat_mean = np.flipud(feat_mean)

    return feat_mean


def visualize_four_images_by_token(token, paths_config, save_dir):
    """
    根据token从四个来源加载数据并拼图
    """
    # 加载图片A
    path_A = paths_config["A"].format(token=token)
    img_A = load_bev_image(path_A)

    # 加载图片B
    path_B = paths_config["B"].format(token=token)
    img_B = load_bev_image(path_B)

    # 加载特征C
    feat_C = load_vis_tensor_from_pt(paths_config["C_pt"], token)
    if feat_C is None:
        feat_C = np.zeros((50, 50))

    # 加载特征D
    feat_D = load_vis_tensor_from_pt(paths_config["D_pt"], token)
    if feat_D is None:
        feat_D = np.zeros((50, 50))

    # 创建2x2子图
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # 左上 - A (BEVDiffuser)
    axes[0, 0].imshow(img_A)
    axes[0, 0].set_title(f"BEVDiffuser (A)\ntoken: {token[:8]}...", fontsize=10)
    axes[0, 0].axis("off")

    # 右上 - B (BEVADMM)
    axes[0, 1].imshow(img_B)
    axes[0, 1].set_title(f"BEVADMM (B)\ntoken: {token[:8]}...", fontsize=10)
    axes[0, 1].axis("off")

    # 左下 - C (Baseline)
    im_C = axes[1, 0].imshow(
        feat_C, cmap="gray", aspect="auto", interpolation="bicubic"
    )
    axes[1, 0].set_title("Baseline (C)", fontsize=12)
    axes[1, 0].axis("off")
    plt.colorbar(im_C, ax=axes[1, 0], fraction=0.046, pad=0.04)

    # 右下 - D (ADMM)
    im_D = axes[1, 1].imshow(
        feat_D, cmap="gray", aspect="auto", interpolation="bicubic"
    )
    axes[1, 1].set_title("ADMM (D)", fontsize=12)
    axes[1, 1].axis("off")
    plt.colorbar(im_D, ax=axes[1, 1], fraction=0.046, pad=0.04)

    # 调整布局
    plt.tight_layout()

    # 保存图片
    save_path = os.path.join(save_dir, f"comparison_{token}.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()

    print(f"已保存: {save_path}")
    return True


def main():
    # 配置路径
    base_dir = "/tmp/pycharm_project_55/BEVFormer/tools/vis_doc"

    paths_config = {
        "A": f"{base_dir}/visual_dir_bevdiffuser_stg1/sample_{{token}}_bev.png",
        "B": f"{base_dir}/visual_dir_bevadmm_stg1/sample_{{token}}_bev.png",
        "C_pt": f"{base_dir}/baseline_vis.pt",
        "D_pt": f"{base_dir}/admm_vis.pt",
    }

    # 输出目录
    save_dir = f"{base_dir}/comparison_results"
    os.makedirs(save_dir, exist_ok=True)

    # 加载其中一个pt文件（比如baseline_vis.pt），获取所有token
    print("正在加载 baseline_vis.pt 获取所有token...")
    baseline_dict = torch.load(paths_config["C_pt"], map_location="cpu")
    all_tokens = list(baseline_dict.keys())
    print(f"共找到 {len(all_tokens)} 个token")

    # 遍历每个token生成拼图
    for idx, token in enumerate(all_tokens):
        print(f"处理 [{idx + 1}/{len(all_tokens)}]: {token}")
        visualize_four_images_by_token(token, paths_config, save_dir)

    print(f"\n完成！所有图片已保存到: {save_dir}")


if __name__ == "__main__":
    main()
