#!/usr/bin/env python3
"""Compare ADMM-Diff and BEVDiffuser results under one evaluation protocol."""

import argparse
import json
import re
from pathlib import Path


METRIC_KEYS = (
    "mean_ap",
    "nd_score",
    "trans_err",
    "scale_err",
    "orient_err",
    "vel_err",
    "attr_err",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Summarize metrics_summary.json files for ADMM-Diff and BEVDiffuser."
    )
    parser.add_argument(
        "--admm-root",
        type=Path,
        default=Path("/root/autodl-tmp/ADMM-Diff"),
        help="ADMM-Diff repository root.",
    )
    parser.add_argument(
        "--baseline-root",
        type=Path,
        default=Path("/root/autodl-tmp/BEVDiffuser-main"),
        help="BEVDiffuser baseline repository root.",
    )
    parser.add_argument("--checkpoint", type=int, default=800)
    parser.add_argument("--protocol", default="5_5_5")
    parser.add_argument(
        "--format",
        choices=("markdown", "csv"),
        default="markdown",
        help="Output format.",
    )
    return parser.parse_args()


def experiment_name(run_name, prefixes):
    name = run_name
    for prefix in prefixes:
        if name.startswith(prefix):
            name = name[len(prefix) :]
    return name or "clean"


def read_metrics(path):
    with path.open("r", encoding="utf-8") as f:
        metrics = json.load(f)
    tp_errors = metrics.get("tp_errors", {})
    return {
        "mean_ap": metrics.get("mean_ap"),
        "nd_score": metrics.get("nd_score"),
        "trans_err": tp_errors.get("trans_err"),
        "scale_err": tp_errors.get("scale_err"),
        "orient_err": tp_errors.get("orient_err"),
        "vel_err": tp_errors.get("vel_err"),
        "attr_err": tp_errors.get("attr_err"),
        "path": str(path),
    }


def discover_admm(root, checkpoint, protocol):
    train_root = root / "BEVFormer/projects/bevdiffuser/train"
    pattern = f"*/checkpoint-{checkpoint}/val/{protocol}/pts_bbox/metrics_summary.json"
    rows = {}
    for path in sorted(train_root.glob(pattern)):
        run_name = path.parts[-6]
        exp = experiment_name(
            run_name, ("admmdiff_stg1_tiny_mini_", "admmdiff_stg1_tiny_mini")
        )
        rows[exp] = read_metrics(path)
    return rows


def discover_baseline(root, checkpoint, protocol):
    train_root = root / "BEVFormer/train"
    pattern = f"*/checkpoint-{checkpoint}/val/{protocol}/pts_bbox/metrics_summary.json"
    rows = {}
    for path in sorted(train_root.glob(pattern)):
        run_name = path.parts[-6]
        exp = experiment_name(
            run_name, ("BEVDiffuser_stg1_tiny_mini_", "BEVDiffuser_stg1_tiny_mini")
        )
        rows[exp] = read_metrics(path)
    return rows


def read_shell_var(script_path, name):
    if not script_path.exists():
        return None
    pattern = re.compile(
        rf"^\s*{re.escape(name)}=(?P<quote>['\"]?)(?P<value>.*?)(?P=quote)\s*$"
    )
    for line in script_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.strip().startswith("#"):
            continue
        match = pattern.match(line)
        if match:
            return match.group("value")
    return None


def protocol_warnings(admm_root, baseline_root):
    admm_test = admm_root / "BEVFormer/projects/bevdiffuser/test.sh"
    baseline_test = baseline_root / "BEVFormer/projects/bevdiffuser/test.sh"
    admm_bev = read_shell_var(admm_test, "BEV_CHECKPOINT")
    baseline_bev = read_shell_var(baseline_test, "BEV_CHECKPOINT")
    warnings = []
    if admm_bev and baseline_bev:
        admm_uses_trained_bev = "checkpoint-" in admm_bev and admm_bev.endswith(
            "bev_model.pth"
        )
        baseline_uses_trained_bev = (
            "checkpoint-" in baseline_bev and baseline_bev.endswith("bev_model.pth")
        )
        if admm_uses_trained_bev != baseline_uses_trained_bev:
            warnings.append(
                "BEV checkpoint policy differs: ADMM uses "
                f"`{admm_bev}`, baseline uses `{baseline_bev}`."
            )
    return warnings


def fmt(value):
    if value is None:
        return ""
    return f"{value:.4f}"


def print_markdown(admm_rows, baseline_rows):
    experiments = sorted(set(admm_rows) | set(baseline_rows))
    headers = [
        "experiment",
        "admm_mAP",
        "base_mAP",
        "delta_mAP",
        "admm_NDS",
        "base_NDS",
        "delta_NDS",
        "admm_mATE",
        "base_mATE",
        "admm_mASE",
        "base_mASE",
        "admm_mAOE",
        "base_mAOE",
    ]
    print("| " + " | ".join(headers) + " |")
    print("| " + " | ".join(["---"] * len(headers)) + " |")
    for exp in experiments:
        admm = admm_rows.get(exp, {})
        base = baseline_rows.get(exp, {})
        delta_map = None
        delta_nds = None
        if admm.get("mean_ap") is not None and base.get("mean_ap") is not None:
            delta_map = admm["mean_ap"] - base["mean_ap"]
        if admm.get("nd_score") is not None and base.get("nd_score") is not None:
            delta_nds = admm["nd_score"] - base["nd_score"]
        values = [
            exp,
            fmt(admm.get("mean_ap")),
            fmt(base.get("mean_ap")),
            fmt(delta_map),
            fmt(admm.get("nd_score")),
            fmt(base.get("nd_score")),
            fmt(delta_nds),
            fmt(admm.get("trans_err")),
            fmt(base.get("trans_err")),
            fmt(admm.get("scale_err")),
            fmt(base.get("scale_err")),
            fmt(admm.get("orient_err")),
            fmt(base.get("orient_err")),
        ]
        print("| " + " | ".join(values) + " |")


def print_csv(admm_rows, baseline_rows):
    experiments = sorted(set(admm_rows) | set(baseline_rows))
    columns = ["experiment"]
    for prefix in ("admm", "baseline"):
        columns.extend(f"{prefix}_{key}" for key in METRIC_KEYS)
    columns.extend(("delta_mean_ap", "delta_nd_score"))
    print(",".join(columns))
    for exp in experiments:
        admm = admm_rows.get(exp, {})
        base = baseline_rows.get(exp, {})
        row = [exp]
        row.extend(fmt(admm.get(key)) for key in METRIC_KEYS)
        row.extend(fmt(base.get(key)) for key in METRIC_KEYS)
        delta_map = (
            admm.get("mean_ap") - base.get("mean_ap")
            if admm.get("mean_ap") is not None and base.get("mean_ap") is not None
            else None
        )
        delta_nds = (
            admm.get("nd_score") - base.get("nd_score")
            if admm.get("nd_score") is not None and base.get("nd_score") is not None
            else None
        )
        row.extend((fmt(delta_map), fmt(delta_nds)))
        print(",".join(row))


def main():
    args = parse_args()
    admm_rows = discover_admm(args.admm_root, args.checkpoint, args.protocol)
    baseline_rows = discover_baseline(
        args.baseline_root, args.checkpoint, args.protocol
    )

    warnings = protocol_warnings(args.admm_root, args.baseline_root)
    for warning in warnings:
        print(f"WARNING: {warning}")
    if warnings:
        print()

    if args.format == "markdown":
        print_markdown(admm_rows, baseline_rows)
    else:
        print_csv(admm_rows, baseline_rows)


if __name__ == "__main__":
    main()
