#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate the dictionary baseline on step 1 (does a sentence mention a social group?)
=====================================================================================
The dictionaries are built from the step 2 annotations (02_build_dictionary.py),
whose sentences all mention a group, so 03_evaluate_dictionary.py only shows how
well they tell groups apart. Here they are applied to step 1's held-out test set,
which also contains sentences without any group: a sentence is predicted to
mention a group if any word of any of the 57 label dictionaries occurs in it
(same tokenizer as for building and step 2 evaluation). Step 1 test sentences
are excluded from the dictionaries' training data (01_prepare_data.py).

Same test set (N=1,260, gold column "has_group") and metrics as the SOCCAT
mDeBERTa baseline (data/model_performance/step_1/performance_all_levels.csv) and
the LLM step 1 runs (llm/analysis/report_step1.py): accuracy, positive-class
precision/recall/F1, and macro F1 over both classes.

Requires 04_run_alpha0_sweep.py to have been run first (one row per
(alpha0-mode, z-threshold) dictionary variant).

Usage: python 06_evaluate_step1.py
Output: dictionary/output/step1_comparison.csv
        dictionary/output/step1_table.tex
"""

import csv
import json
import sys
from importlib import import_module
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))
from tokenizer import tokenize  # noqa: E402

z_suffix = import_module("02_build_dictionary").z_suffix
load_dictionaries = import_module("03_evaluate_dictionary").load_dictionaries

DICTIONARIES_DIR = ROOT / "dictionaries"
OUTPUT_DIR = ROOT / "output"
STEP1_TEST_FILE = REPO_ROOT / "data" / "model_performance" / "step_1" / "test_with_all_outlets.json"
BASELINE_PATH = REPO_ROOT / "data" / "model_performance" / "step_1" / "performance_all_levels.csv"

MODES = ["low", "medium", "high"]
Z_THRESHOLDS = [0.0, 1.96]
MODE_LABELS = {"low": "Low", "medium": "Medium", "high": "High"}  # display only
Z_LABELS = {0.0: "$z>0$", 1.96: "$z>1.96$"}  # as in 05_make_latex_table.py


def load_step1_test() -> pd.DataFrame:
    with STEP1_TEST_FILE.open(encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return pd.DataFrame({"text": [r["text"] for r in rows], "has_group": [int(r["has_group"]) for r in rows]})


def load_baseline_metrics() -> dict:
    """SOCCAT mDeBERTa's overall row (group == "ALL"), as in llm/analysis/report_step1.py."""
    with BASELINE_PATH.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("group") == "ALL":
                return {
                    "accuracy": float(row["Accuracy"]),
                    "precision": float(row["Precision_binary"]),
                    "recall": float(row["Recall_binary"]),
                    "f1": float(row["F1_binary"]),
                    "f1_macro": float(row["F1_macro"]),
                    "tp": int(row["TP"]), "fp": int(row["FP"]), "fn": int(row["FN"]), "tn": int(row["TN"]),
                    "n": int(row["N"]),
                }
    raise SystemExit(f"error: no overall row (group == 'ALL') in {BASELINE_PATH}")


def compute_metrics(y_true: list, y_pred: list) -> dict:
    tp = sum(t == 1 and p == 1 for t, p in zip(y_true, y_pred))
    fp = sum(t == 0 and p == 1 for t, p in zip(y_true, y_pred))
    fn = sum(t == 1 and p == 0 for t, p in zip(y_true, y_pred))
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "tp": tp, "fp": fp, "fn": fn, "tn": len(y_true) - tp - fp - fn,
        "n": len(y_true),
    }


def build_latex_table(rows: list, baseline: dict) -> list:
    """One row per dictionary variant (sweep order), SOCCAT pinned as the last row.
    Layout follows llm/analysis/output/step_1_table.tex."""
    lines = [
        r"\begin{table}[htb]",
        r"\centering",
        r"\begin{threeparttable}",
        r"\begin{tabular}{llrrrrrr}",
        r"\toprule",
        r"Model & $\alpha_0$ & $z$ & Accuracy & Precision & Recall & F1 & Macro F1 \\",
        r"\midrule",
    ]
    for r in rows:
        lines.append(
            f"Dictionary & {MODE_LABELS[r['alpha0_mode']]} & {Z_LABELS[r['z_threshold']]} & "
            f"{r['accuracy']:.2f} & {r['precision']:.2f} & {r['recall']:.2f} & {r['f1']:.2f} & "
            f"{r['f1_macro']:.2f} \\\\"
        )
    lines += [
        r"\midrule",
        f"SOCCAT & -- & -- & {baseline['accuracy']:.2f} & {baseline['precision']:.2f} & "
        f"{baseline['recall']:.2f} & {baseline['f1']:.2f} & {baseline['f1_macro']:.2f} \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\begin{tablenotes}[flushleft]",
        r"\footnotesize",
        r"\item \textit{Note:} A sentence is classified as mentioning a social group if any word of "
        r"any of the 57 label dictionaries occurs in it. $\alpha_0$ and $z$ are model parameters "
        r"that yield different dictionary word lists. Precision, recall and F1 refer to the positive "
        r"class (sentence mentions a social group); macro F1 averages the F1 scores of both classes. "
        f"Held-out test set, N = {baseline['n']:,} sentences.",
        r"\end{tablenotes}",
        r"\end{threeparttable}",
        r"\caption{Step 1 classification performance of the dictionary baseline and the SOCCAT mDeBERTa baseline.}",
        r"\label{tab:dictionary-step1}",
        r"\end{table}",
        "",
    ]
    return lines


def main():
    test = load_step1_test()
    tokens = [set(tokenize(t)) for t in test["text"]]
    y_true = test["has_group"].tolist()
    baseline = load_baseline_metrics()
    assert baseline["n"] == len(test), f"SOCCAT N={baseline['n']} but step 1 test set has {len(test)} sentences"

    rows = []
    for mode in MODES:
        for z in Z_THRESHOLDS:
            variant = f"{mode}_{z_suffix(z)}"
            dict_dir = DICTIONARIES_DIR / variant
            alpha0 = json.loads((dict_dir / "alpha0.json").read_text(encoding="utf-8"))["alpha0"]
            all_words = set().union(*load_dictionaries(dict_dir).values())
            y_pred = [int(bool(toks & all_words)) for toks in tokens]
            rows.append({"alpha0_mode": mode, "z_threshold": z, "alpha0": round(alpha0),
                         "n_words": len(all_words), **compute_metrics(y_true, y_pred)})

    df = pd.DataFrame(rows + [{"alpha0_mode": "SOCCAT", **baseline}])
    df.to_csv(OUTPUT_DIR / "step1_comparison.csv", index=False)
    (OUTPUT_DIR / "step1_table.tex").write_text("\n".join(build_latex_table(rows, baseline)), encoding="utf-8")

    pd.set_option("display.width", 200)
    cols = ["alpha0_mode", "z_threshold", "alpha0", "n_words", "accuracy", "precision", "recall",
            "f1", "f1_macro", "tp", "fp", "fn", "tn"]
    print(df[cols].round(3).to_string(index=False))
    print(f"\nDone. Written to: {OUTPUT_DIR / 'step1_comparison.csv'} and {OUTPUT_DIR / 'step1_table.tex'}")


if __name__ == "__main__":
    main()
