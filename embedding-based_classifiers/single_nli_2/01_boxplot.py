#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 2 boxplots for the single NLI model
=========================================
Same figures as for the main models (figures/step_2_boxplot.py: one model per
broad class) for the single NLI model trained on all 57 labels at once
(data/: 5-fold CV, same 7,234 sentences and gold labels as the main models).

step_2_boxplot.py expects one folder per broad class, each with
fold_*_per_label.csv (plain label names) and fold_*_metrics.json (metrics on
all of the class's pairs of the fold, pooled). This script writes the single
model's results in that layout to output/per_class_layout/ -- per-label rows
copied from data/fold_*_per_label.csv, class-level metrics computed from
data/fold_*_human_vs_model.csv with the same definitions as
src/step_2/step_2_cv_pipeline.py -- and then runs step_2_boxplot.py on it, so
both sets of figures are drawn by the same code.

Outputs (output/boxplots/):
    step_2_perf_broad_class_boxplot.pdf        broad classes
    step_2_perf_boxplot_part1.pdf, _part2.pdf  all labels, grouped by broad class
    cat_per_broad_class_boxplot/<class>_boxplot.pdf  one figure per broad class
    (+ the *_summary.txt / *_best_fold_used.txt / *_neg_class_pct.txt logs)

Usage: python 01_boxplot.py   (run from anywhere)
"""

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
DATA_DIR = ROOT / "data"
LAYOUT_DIR = ROOT / "output" / "per_class_layout"
BOXPLOT_DIR = ROOT / "output" / "boxplots"
BOXPLOT_SCRIPT = REPO / "figures" / "step_2_boxplot.py"
N_FOLDS = 5

# broad class as named in the single model's hypothesis_label -> folder name of the
# main models (data/model_performance/step_2/<folder>), as expected by step_2_boxplot.py
CLASS_FOLDER = {
    "Age and Family Status": "age_family_status",
    "Identities": "identities_minority_majority_status",
    "Labor Market Position": "labor_market_position",
    "Profession": "profession",
    "Real Estate Ownership": "real_estate_ownership",
    "Social Deviance": "social_deviance",
    "Social Roles and Behavior": "social_roles_behavior",
    "Socio-Economic Position": "socio_economic_position",
}
# step_2_boxplot.py --broad_class codes (keys of its FOLDER_NAME_MAP)
BOXPLOT_CLASS_CODES = ["age_family", "identity", "labor_market_position", "profession",
                       "real_estate", "social_deviance", "social_roles", "socio_economic"]


def class_metrics(pairs: pd.DataFrame) -> dict:
    """Metrics on all (sentence, label) pairs of one class and fold, pooled, as
    compute_all_metrics() in src/step_2/step_2_cv_pipeline.py (0 = entailment)."""
    y_true, y_pred = pairs["nli_label"], pairs["pred_label"]
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_binary": precision_score(y_true, y_pred, pos_label=0, zero_division=0),
        "recall_binary": recall_score(y_true, y_pred, pos_label=0),
        "f1_binary": f1_score(y_true, y_pred, pos_label=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
    }


def write_layout():
    for fold in range(N_FOLDS):
        per_label = pd.read_csv(DATA_DIR / f"fold_{fold}_per_label.csv")
        parts = per_label["hypothesis_label"].str.split(" | ", n=1, regex=False)
        per_label["broad"], per_label["hypothesis_label"] = parts.str[0], parts.str[1]
        pairs = pd.read_csv(DATA_DIR / f"fold_{fold}_human_vs_model.csv",
                            usecols=["dimension", "nli_label", "pred_label"])
        assert set(per_label["broad"]) == set(CLASS_FOLDER), f"unexpected broad classes: {set(per_label['broad'])}"
        for broad, folder in CLASS_FOLDER.items():
            out = LAYOUT_DIR / folder
            out.mkdir(parents=True, exist_ok=True)
            per_label[per_label["broad"] == broad].drop(columns="broad").to_csv(
                out / f"fold_{fold}_per_label.csv", index=False)
            metrics = {**class_metrics(pairs[pairs["dimension"] == broad]), "fold": fold}
            (out / f"fold_{fold}_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Per-class layout written to: {LAYOUT_DIR}")


def run_boxplot(*extra_args):
    cmd = [sys.executable, str(BOXPLOT_SCRIPT), "--data_dir", str(LAYOUT_DIR),
           "--output_dir", str(BOXPLOT_DIR), *extra_args]
    # step_2_boxplot.py resolves its other paths relative to figures/
    subprocess.run(cmd, cwd=BOXPLOT_SCRIPT.parent, check=True, stdout=subprocess.DEVNULL)


def main():
    write_layout()
    run_boxplot()
    for code in BOXPLOT_CLASS_CODES:
        run_boxplot("--broad_class", code)
    print(f"Done. Boxplots written to: {BOXPLOT_DIR}")


if __name__ == "__main__":
    main()
