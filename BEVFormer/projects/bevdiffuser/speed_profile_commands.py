#!/usr/bin/env python3
"""Print timing commands for ADMM-Diff iteration sweeps and BEVDiffuser."""

from pathlib import Path


ADMM_TEST = Path("/root/autodl-tmp/ADMM-Diff/BEVFormer/projects/bevdiffuser/test.sh")
BASELINE_TEST = Path(
    "/root/autodl-tmp/BEVDiffuser-main/BEVFormer/projects/bevdiffuser/test.sh"
)


def main():
    print("# ADMM-Diff NDS-latency sweep")
    for num_iters in (1, 2, 4):
        print(f'ENABLE_TIMING=1 NUM_ADMM_ITERS={num_iters} bash "{ADMM_TEST}"')
    print()
    print("# BEVDiffuser baseline latency")
    print(f'ENABLE_TIMING=1 bash "{BASELINE_TEST}"')


if __name__ == "__main__":
    main()
