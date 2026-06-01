#!/usr/bin/env python3
"""Print the minimal clean/noisy experiment matrix commands for ADMM-Diff."""

from pathlib import Path


ADMM_SCRIPT_DIR = Path("/root/autodl-tmp/ADMM-Diff/BEVFormer/projects/bevdiffuser")
BASELINE_SCRIPT_DIR = Path("/root/autodl-tmp/BEVDiffuser-main/BEVFormer/projects/bevdiffuser")
PKL_ROOT = Path("/root/autodl-tmp/nuscenes")

EXPERIMENTS = {
    "clean": PKL_ROOT / "pkl_4_bevdiffuser/pkl_4_bevdiffuser_mini",
    "wlswap100": PKL_ROOT / "mini_pkl_4_bevdiffuser_wlswap100",
    "xy02": PKL_ROOT / "mini_pkl_4_bevdiffuser_xy02",
}


def cfg_options(info_root):
    info_root = str(info_root).rstrip("/") + "/"
    train_pkl = info_root + "nuscenes_infos_temporal_train.pkl"
    val_pkl = info_root + "nuscenes_infos_temporal_val.pkl"
    return " ".join(
        [
            f"data.train.ann_file={train_pkl}",
            f"data.val.ann_file={val_pkl}",
            f"data.test.ann_file={val_pkl}",
        ]
    )


def main():
    print("# Minimal ADMM-Diff / BEVDiffuser matrix")
    print("# A: noisy layout only, no noisy task loss")
    print("# B: noisy layout + noisy task loss, but frozen BEV head")
    print("# C: original full coupling")
    print()
    for name, info_root in EXPERIMENTS.items():
        if not info_root.exists():
            print(f"# WARNING: missing pkl directory: {info_root}")
        options = cfg_options(info_root)
        base = f'CFG_OPTIONS="{options}" NUM_ADMM_ITERS=4'
        baseline_base = f'CFG_OPTIONS="{options}"'
        print(f"# === {name} ===")
        print(
            f'{base} RUN_NAME="admmdiff_diag_{name}_layout_only" '
            f'PROJ_NAME="admmdiff_diag" TASK_LOSS_SCALE=0 '
            f'bash "{ADMM_SCRIPT_DIR / "train.sh"}"'
        )
        print(
            f'{baseline_base} RUN_NAME="bevdiffuser_diag_{name}_layout_only" '
            f'PROJ_NAME="bevdiffuser_diag" TASK_LOSS_SCALE=0 '
            f'bash "{BASELINE_SCRIPT_DIR / "train_tiny.sh"}"'
        )
        print(
            f'{base} RUN_NAME="admmdiff_diag_{name}_frozen_head" '
            f'PROJ_NAME="admmdiff_diag" TASK_LOSS_SCALE=0.1 FREEZE_BEV_HEAD_FOR_TASK_LOSS=1 '
            f'bash "{ADMM_SCRIPT_DIR / "train.sh"}"'
        )
        print(
            f'{baseline_base} RUN_NAME="bevdiffuser_diag_{name}_frozen_head" '
            f'PROJ_NAME="bevdiffuser_diag" TASK_LOSS_SCALE=0.1 FREEZE_BEV_HEAD_FOR_TASK_LOSS=1 '
            f'bash "{BASELINE_SCRIPT_DIR / "train_tiny.sh"}"'
        )
        print(
            f'{base} RUN_NAME="admmdiff_diag_{name}_full" '
            f'PROJ_NAME="admmdiff_diag" TASK_LOSS_SCALE=0.1 '
            f'bash "{ADMM_SCRIPT_DIR / "train.sh"}"'
        )
        print(
            f'{baseline_base} RUN_NAME="bevdiffuser_diag_{name}_full" '
            f'PROJ_NAME="bevdiffuser_diag" TASK_LOSS_SCALE=0.1 '
            f'bash "{BASELINE_SCRIPT_DIR / "train_tiny.sh"}"'
        )
        print()


if __name__ == "__main__":
    main()
