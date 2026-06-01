#!/usr/bin/env python3
"""Create noisy nuScenes info pkl files for label-noise robustness studies.

The nuScenes info file stores boxes as [x, y, z, w, l, h, yaw].  This script
modifies only gt_boxes and leaves images, calibration, sweeps and metadata
unchanged.  Use it on training pkl files; keep validation/test pkl files clean.
"""

import argparse
import copy
import json
import os
import pickle
from typing import Any, Dict, Iterable, Optional

import numpy as np

# mini集训练先验证鲁棒性是否存在。

# 使用方法：
# cd /root/autodl-tmp/ADMM-Diff/BEVFormer
# 生成 yaw 噪声训练 pkl：
# python tools/create_noisy_pkl.py --input /root/autodl-tmp/nuscenes/pkl_4_bevdiffuser/pkl_4_bevdiffuser_mini/nuscenes_infos_temporal_train.pkl --output /root/autodl-tmp/nuscenes/mini_pkl_4_bevdiffuser_yaw15/nuscenes_infos_temporal_train.pkl --yaw-std-deg 15 --seed 0

# 生成位置噪声训练 pkl
# python tools/create_noisy_pkl.py --input /root/autodl-tmp/nuscenes/pkl_4_bevdiffuser/pkl_4_bevdiffuser_mini/nuscenes_infos_temporal_train.pkl --output /root/autodl-tmp/nuscenes/mini_pkl_4_bevdiffuser_xy02/nuscenes_infos_temporal_train.pkl --xy-std 0.2 --seed 0

# 生成尺寸噪声训练 pkl
# python tools/create_noisy_pkl.py --input /root/autodl-tmp/nuscenes/pkl_4_bevdiffuser/pkl_4_bevdiffuser_mini/nuscenes_infos_temporal_train.pkl --output /root/autodl-tmp/nuscenes/mini_pkl_4_bevdiffuser_size10/nuscenes_infos_temporal_train.pkl --size-mode relative --size-std 0.10 --seed 0

# 生成宽长交换 pkl
# python tools/create_noisy_pkl.py --input /root/autodl-tmp/nuscenes/pkl_4_bevdiffuser/pkl_4_bevdiffuser_mini/nuscenes_infos_temporal_train.pkl --output /root/autodl-tmp/nuscenes/mini_pkl_4_bevdiffuser_wlswap100/nuscenes_infos_temporal_train.pkl --wl-swap-prob 1.0 --seed 0

# 上述四条指令生成歪的train集，这条指令执行四次生成复制正确的val集。训练用歪的，测试用正的。以验证：在错误的监督下模型是否能够保持鲁棒？
# cp /root/autodl-tmp/nuscenes/pkl_4_bevdiffuser/pkl_4_bevdiffuser_mini/nuscenes_infos_temporal_val.pkl /root/autodl-tmp/nuscenes/mini_pkl_4_bevdiffuser_yaw15/nuscenes_infos_temporal_val.pkl
# cp /root/autodl-tmp/nuscenes/pkl_4_bevdiffuser/pkl_4_bevdiffuser_mini/nuscenes_infos_temporal_val.pkl /root/autodl-tmp/nuscenes/mini_pkl_4_bevdiffuser_xy02/nuscenes_infos_temporal_val.pkl
# cp /root/autodl-tmp/nuscenes/pkl_4_bevdiffuser/pkl_4_bevdiffuser_mini/nuscenes_infos_temporal_val.pkl /root/autodl-tmp/nuscenes/mini_pkl_4_bevdiffuser_wlswap100/
# cp /root/autodl-tmp/nuscenes/pkl_4_bevdiffuser/pkl_4_bevdiffuser_mini/nuscenes_infos_temporal_val.pkl /root/autodl-tmp/nuscenes/mini_pkl_4_bevdiffuser_size10/nuscenes_infos_temporal_val.pkl

BOX_DIM = 7
XYZ_SLICE = slice(0, 3)
W_IDX = 3
L_IDX = 4
H_IDX = 5
YAW_IDX = 6


def detect_pickle_protocol(path: str) -> Optional[int]:
    with open(path, "rb") as f:
        header = f.read(2)
    if len(header) == 2 and header[0] == 0x80:
        return header[1]
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inject controlled label noise into nuScenes info pkl files."
    )
    parser.add_argument("--input", required=True, help="Path to the clean input pkl.")
    parser.add_argument("--output", required=True, help="Path to write the noisy pkl.")
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed. Use the same seed for fair repeated experiments.",
    )
    parser.add_argument(
        "--object-prob",
        type=float,
        default=1.0,
        help="Probability that each GT box receives the configured noise.",
    )
    parser.add_argument(
        "--classes",
        nargs="*",
        default=None,
        help="Optional class names to corrupt. By default all classes are corrupted.",
    )
    parser.add_argument(
        "--yaw-std-deg",
        type=float,
        default=0.0,
        help="Gaussian yaw noise std in degrees.",
    )
    parser.add_argument(
        "--yaw-offset-deg",
        type=float,
        default=0.0,
        help="Constant yaw offset in degrees, useful for systematic label bias.",
    )
    parser.add_argument(
        "--xy-std",
        type=float,
        default=0.0,
        help="Gaussian x/y translation noise std in meters.",
    )
    parser.add_argument(
        "--z-std",
        type=float,
        default=0.0,
        help="Gaussian z translation noise std in meters.",
    )
    parser.add_argument(
        "--size-std",
        type=float,
        default=0.0,
        help="Gaussian size noise std. Interpreted by --size-mode.",
    )
    parser.add_argument(
        "--size-mode",
        choices=("relative", "absolute"),
        default="relative",
        help=(
            "relative: multiply w/l/h by 1 + N(0, size_std); "
            "absolute: add N(0, size_std) meters to w/l/h."
        ),
    )
    parser.add_argument(
        "--wl-swap-prob",
        type=float,
        default=0.0,
        help="Probability of swapping width and length for selected boxes.",
    )
    parser.add_argument(
        "--min-size",
        type=float,
        default=0.05,
        help="Lower bound for noisy w/l/h to avoid invalid boxes.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned corruption summary without writing output.",
    )
    parser.add_argument(
        "--pickle-protocol",
        type=int,
        default=None,
        help=(
            "Pickle protocol used for output. Default: preserve the input pkl "
            "protocol when it can be detected."
        ),
    )
    parser.add_argument(
        "--record-metadata",
        action="store_true",
        help=(
            "Record noise parameters under metadata['label_noise']. Disabled by "
            "default so only gt_boxes differ from the input pkl."
        ),
    )
    return parser.parse_args()


def load_pkl(path: str) -> Dict[str, Any]:
    with open(path, "rb") as f:
        return pickle.load(f)


def dump_pkl(data: Dict[str, Any], path: str, protocol: int) -> None:
    output_dir = os.path.dirname(os.path.abspath(path))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(data, f, protocol=protocol)


def wrap_to_pi(angle: np.ndarray) -> np.ndarray:
    return (angle + np.pi) % (2 * np.pi) - np.pi


def class_mask(gt_names: Optional[Iterable[str]], classes: Optional[set]) -> Optional[np.ndarray]:
    if classes is None:
        return None
    if gt_names is None:
        raise KeyError("--classes was set, but the pkl info has no 'gt_names' field.")
    return np.array([name in classes for name in gt_names], dtype=bool)


def corrupt_boxes(
    boxes: np.ndarray,
    rng: np.random.Generator,
    args: argparse.Namespace,
    gt_names: Optional[Iterable[str]] = None,
) -> Dict[str, int]:
    stats = {
        "selected_boxes": 0,
        "yaw_noisy_boxes": 0,
        "xy_noisy_boxes": 0,
        "z_noisy_boxes": 0,
        "size_noisy_boxes": 0,
        "wl_swapped_boxes": 0,
    }

    if boxes.size == 0:
        return stats
    if boxes.ndim != 2 or boxes.shape[1] < BOX_DIM:
        raise ValueError(f"Expected gt_boxes with shape [N, >=7], got {boxes.shape}.")

    selected = rng.random(boxes.shape[0]) < args.object_prob
    selected_by_class = class_mask(gt_names, set(args.classes) if args.classes else None)
    if selected_by_class is not None:
        selected &= selected_by_class

    stats["selected_boxes"] = int(selected.sum())
    if not selected.any():
        return stats

    if args.xy_std > 0:
        boxes[selected, 0:2] += rng.normal(0.0, args.xy_std, size=(selected.sum(), 2))
        stats["xy_noisy_boxes"] = int(selected.sum())

    if args.z_std > 0:
        boxes[selected, 2] += rng.normal(0.0, args.z_std, size=selected.sum())
        stats["z_noisy_boxes"] = int(selected.sum())

    if args.size_std > 0:
        if args.size_mode == "relative":
            scale = 1.0 + rng.normal(0.0, args.size_std, size=(selected.sum(), 3))
            boxes[selected, 3:6] *= scale
        else:
            boxes[selected, 3:6] += rng.normal(0.0, args.size_std, size=(selected.sum(), 3))
        boxes[:, 3:6] = np.maximum(boxes[:, 3:6], args.min_size)
        stats["size_noisy_boxes"] = int(selected.sum())

    swap_selected = selected & (rng.random(boxes.shape[0]) < args.wl_swap_prob)
    if swap_selected.any():
        swapped_width = boxes[swap_selected, L_IDX].copy()
        swapped_length = boxes[swap_selected, W_IDX].copy()
        boxes[swap_selected, W_IDX] = swapped_width
        boxes[swap_selected, L_IDX] = swapped_length
        stats["wl_swapped_boxes"] = int(swap_selected.sum())

    if args.yaw_std_deg > 0 or args.yaw_offset_deg != 0:
        yaw_noise = np.zeros(selected.sum(), dtype=boxes.dtype)
        if args.yaw_std_deg > 0:
            yaw_noise += rng.normal(0.0, np.deg2rad(args.yaw_std_deg), size=selected.sum())
        if args.yaw_offset_deg != 0:
            yaw_noise += np.deg2rad(args.yaw_offset_deg)
        boxes[selected, YAW_IDX] = wrap_to_pi(boxes[selected, YAW_IDX] + yaw_noise)
        stats["yaw_noisy_boxes"] = int(selected.sum())

    return stats


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.object_prob <= 1.0:
        raise ValueError("--object-prob must be in [0, 1].")
    if not 0.0 <= args.wl_swap_prob <= 1.0:
        raise ValueError("--wl-swap-prob must be in [0, 1].")

    rng = np.random.default_rng(args.seed)
    data = load_pkl(args.input)
    noisy_data = copy.deepcopy(data)
    infos = noisy_data.get("infos")
    if not isinstance(infos, list):
        raise KeyError("Input pkl must be a dict containing a list under key 'infos'.")

    total_stats = {
        "samples": len(infos),
        "boxes": 0,
        "selected_boxes": 0,
        "yaw_noisy_boxes": 0,
        "xy_noisy_boxes": 0,
        "z_noisy_boxes": 0,
        "size_noisy_boxes": 0,
        "wl_swapped_boxes": 0,
    }

    for info in infos:
        boxes = info.get("gt_boxes")
        if boxes is None:
            continue
        boxes = np.asarray(boxes).copy()
        total_stats["boxes"] += int(boxes.shape[0])
        stats = corrupt_boxes(boxes, rng, args, info.get("gt_names"))
        for key, value in stats.items():
            total_stats[key] += value
        info["gt_boxes"] = boxes

    if args.record_metadata:
        metadata = noisy_data.setdefault("metadata", {})
        metadata["label_noise"] = {
            "input": os.path.abspath(args.input),
            "seed": args.seed,
            "object_prob": args.object_prob,
            "classes": args.classes,
            "yaw_std_deg": args.yaw_std_deg,
            "yaw_offset_deg": args.yaw_offset_deg,
            "xy_std": args.xy_std,
            "z_std": args.z_std,
            "size_std": args.size_std,
            "size_mode": args.size_mode,
            "wl_swap_prob": args.wl_swap_prob,
            "min_size": args.min_size,
        }

    print(json.dumps(total_stats, indent=2, sort_keys=True))
    if args.dry_run:
        print("Dry run: output was not written.")
        return

    input_protocol = detect_pickle_protocol(args.input)
    output_protocol = args.pickle_protocol
    if output_protocol is None:
        output_protocol = input_protocol if input_protocol is not None else pickle.HIGHEST_PROTOCOL

    dump_pkl(noisy_data, args.output, output_protocol)
    print(f"Wrote noisy pkl to: {args.output}")
    print(f"Pickle protocol: input={input_protocol}, output={output_protocol}")


if __name__ == "__main__":
    main()
