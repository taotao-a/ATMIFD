"""Run 4 ablation variants by toggling args.

Usage (from project root):
  python scripts/run_ablation.py --seeds 42

Notes:
- NoTraceNodeFeatures and MetricTraceFusion require different cached dataset_path folders
  because the node feature dimension differs.
- This script just launches main.py multiple times.
"""

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--python", default=sys.executable, help="python executable")
    ap.add_argument("--seeds", default="42", help="comma-separated seeds; paper runs use 42")
    ap.add_argument("--result_dir", default="./result_ablation", help="base result dir")
    ap.add_argument("--data_path", default="./data/MSDS-pre", help="raw MSDS-pre path")
    ap.add_argument("--ds_metric", default="./data/MSDS-save-v2-metric", help="cache without trace-derived node features")
    ap.add_argument("--ds_fusion", default="./data/MSDS-save-v2-fusion", help="cache for MetricTraceFusion*")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--epochs", default=50, type=int)
    ap.add_argument("--batch_size", default=50, type=int)
    ap.add_argument("--dry-run", action="store_true", help="print commands without training or writing results")
    return ap.parse_args()


def run(cmd, dry_run=False):
    print("\n>>>", " ".join(cmd))
    if not dry_run:
        subprocess.run(cmd, check=True, cwd=str(PROJECT_ROOT))


def main():
    a = parse_args()
    seeds = [int(x) for x in a.seeds.split(",") if x.strip()]
    if not seeds:
        raise ValueError('--seeds must contain at least one integer')
    if a.epochs < 3 or a.batch_size < 1:
        raise ValueError('Ablation runs require epochs >= 3 and batch_size >= 1')

    variants = [
        # name, overrides
        ("NoTraceNodeFeatures", {
            "trace_node_dim": "0",
            "dataset_path": a.ds_metric,
            "thr_search": "false",
            "imb_loss": "false",
        }),
        ("MetricTraceFusion", {
            "trace_node_dim": "6",
            "dataset_path": a.ds_fusion,
            "thr_search": "false",
            "imb_loss": "false",
        }),
        ("MetricTraceFusion+ThrSearch", {
            "trace_node_dim": "6",
            "dataset_path": a.ds_fusion,
            "thr_search": "true",
            "imb_loss": "false",
        }),
        ("ATMIFD", {
            "trace_node_dim": "6",
            "dataset_path": a.ds_fusion,
            "thr_search": "true",
            "imb_loss": "true",
        }),
    ]

    for seed in seeds:
        for name, ov in variants:
            cmd = [
                a.python, str(PROJECT_ROOT / "main.py"),
                "--main_model", name,
                "--device", a.device,
                "--epochs", str(a.epochs),
                "--batch_size", str(a.batch_size),
                "--random_seed", str(seed),
                "--data_path", a.data_path,
                "--dataset_path", ov["dataset_path"],
                "--result_dir", a.result_dir,
                "--trace_node_dim", ov["trace_node_dim"],
                "--thr_search", ov["thr_search"],
                "--imb_loss", ov["imb_loss"],
                "--eval_stage", "f1",
                # keep window-level score aggregation consistent
                "--win_agg", "max",
            ]
            run(cmd, dry_run=a.dry_run)


if __name__ == "__main__":
    main()
