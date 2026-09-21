#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run the dictionary baseline across the three pre-registered alpha0 values
=============================================================================
Robustness check on eq. (23)'s alpha0 ("prior sample size", see
02_build_dictionary.py's module docstring for what low/medium/high mean).
Runs 02_build_dictionary.py and 03_evaluate_dictionary.py once per mode,
then assembles a single side-by-side comparison table.

Usage: python 04_run_alpha0_sweep.py
(run from anywhere -- paths are resolved relative to this file)

Output: dictionary/output/alpha0_sweep_comparison.csv
"""

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
MODES = ["low", "medium", "high"]


def run(script: str, mode: str):
    cmd = [sys.executable, str(ROOT / script), "--alpha0-mode", mode]
    print(f"$ {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def main():
    for mode in MODES:
        run("02_build_dictionary.py", mode)
        run("03_evaluate_dictionary.py", mode)

    rows = []
    for mode in MODES:
        out_dir = ROOT / "output" / mode
        with open(ROOT / "dictionaries" / mode / "alpha0.json", encoding="utf-8") as f:
            alpha0_info = json.load(f)
        label_df = pd.read_csv(out_dir / "per_specific_category_metrics.csv")
        cat_df = pd.read_csv(out_dir / "per_broad_class_metrics.csv")

        micro_tp, micro_fp, micro_fn = label_df["tp"].sum(), label_df["fp"].sum(), label_df["fn"].sum()
        micro_p = micro_tp / (micro_tp + micro_fp) if (micro_tp + micro_fp) else 0.0
        micro_r = micro_tp / (micro_tp + micro_fn) if (micro_tp + micro_fn) else 0.0
        micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) else 0.0

        rows.append({
            "alpha0_mode": mode,
            "alpha0": round(alpha0_info["alpha0"]),
            "alpha0_description": alpha0_info["description"],
            "label_macro_f1": round(label_df["f1"].mean(), 3),
            "label_micro_precision": round(micro_p, 3),
            "label_micro_recall": round(micro_r, 3),
            "label_micro_f1": round(micro_f1, 3),
            "category_macro_f1": round(cat_df["f1"].mean(), 3),
        })

    comparison = pd.DataFrame(rows)
    out_path = ROOT / "output" / "alpha0_sweep_comparison.csv"
    comparison.to_csv(out_path, index=False)
    print("\n" + comparison.to_string(index=False))
    print(f"\nDone. Comparison written to: {out_path}")


if __name__ == "__main__":
    main()
