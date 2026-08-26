#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate the dictionary baseline on the held-out test split
==============================================================
For every test sentence and every one of the 57 labels, predicts a match if
any word from that label's dictionary (dictionary/dictionaries/*.txt) is
present as a token in the sentence. Ground truth comes from test.csv's
'labels' / 'categories' columns (produced by 01_prepare_data.py). Test
sentences are tokenized with the exact same pipeline (stopwords + stemming,
see tokenizer.py) used to build the dictionaries -- required for stemmed
dictionary entries to match at all.

Unlike the NLI models' within-category evaluation (each category's model is
only tested on sentences already known to be positive for that category, see
src/step_2/step_2_cv_pipeline.py), every test sentence is scored against
every label here -- there is no reason to restrict it, since dictionary
matching has no training-time coupling to a subset of sentences. This gives
a full-corpus picture, including how often the dictionary fires on
sentences that don't belong to that label/category at all.

Reads dictionaries from dictionary/dictionaries/<alpha0-mode>/ (see
--alpha0-mode and 02_build_dictionary.py's module docstring for what the
three modes mean) and writes to dictionary/output/<alpha0-mode>/:
  per_specific_category_metrics.csv  precision/recall/F1/support per label
  per_broad_class_metrics.csv        same, rolled up to broad category (match =
                                      any label dictionary in that category fires)
  summary.txt               overall micro/macro summary, including the
                             alpha0 value the dictionaries were built with
"""

import argparse
import json
import re
from pathlib import Path

import pandas as pd

from tokenizer import tokenize

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
DICTIONARIES_DIR = ROOT / "dictionaries"
OUTPUT_DIR = ROOT / "output"


def load_dictionaries(dict_dir: Path):
    """Returns {(category, label): set(words)}."""
    summary = pd.read_csv(dict_dir / "summary.csv")
    dicts = {}
    for _, row in summary.iterrows():
        fname = f"{row['category']}__{row['label']}".replace("/", "-")
        fname = re.sub(r'[<>:"\\|?*]', "_", fname) + ".txt"
        words = set(open(dict_dir / fname, encoding="utf-8").read().split())
        dicts[(row["category"], row["label"])] = words
    return dicts


def prf1(tp, fp, fn):
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha0-mode", choices=["low", "medium", "high"], default="medium",
                         help="which alpha0-mode's dictionaries to evaluate (must have been "
                              "built already by 02_build_dictionary.py --alpha0-mode <mode>)")
    args = parser.parse_args()

    dict_dir = DICTIONARIES_DIR / args.alpha0_mode
    out_dir = OUTPUT_DIR / args.alpha0_mode
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(dict_dir / "alpha0.json", encoding="utf-8") as f:
        alpha0_info = json.load(f)

    test = pd.read_csv(DATA_DIR / "test.csv", keep_default_na=False)
    test["label_set"] = test["labels"].apply(lambda s: set(s.split(";")) if s else set())
    test["category_set"] = test["categories"].apply(lambda s: set(s.split(";")) if s else set())
    test["tokens"] = test.apply(lambda r: set(tokenize(r["text"], r["country"])), axis=1)

    dicts = load_dictionaries(dict_dir)

    # ── Per-label metrics ───────────────────────────────────────────────────
    label_rows = []
    # predicted_categories[i] = set of categories that fired for test sentence i
    predicted_categories = [set() for _ in range(len(test))]
    true_categories = test["category_set"].tolist()

    for (cat, label), words in dicts.items():
        matched = test["tokens"].apply(lambda toks: bool(toks & words))
        truth = test["label_set"].apply(lambda s: label in s)

        tp = int((matched & truth).sum())
        fp = int((matched & ~truth).sum())
        fn = int((~matched & truth).sum())
        tn = int((~matched & ~truth).sum())
        precision, recall, f1 = prf1(tp, fp, fn)

        for i, m in enumerate(matched):
            if m:
                predicted_categories[i].add(cat)

        label_rows.append({
            "category": cat, "label": label,
            "support": int(truth.sum()),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": round(precision, 3), "recall": round(recall, 3), "f1": round(f1, 3),
        })

    label_df = pd.DataFrame(label_rows)
    label_df.to_csv(out_dir / "per_specific_category_metrics.csv", index=False)

    # ── Per-category metrics (match = any label dictionary in category fires) ─
    cat_rows = []
    for cat in sorted(label_df["category"].unique()):
        matched = pd.Series([cat in pc for pc in predicted_categories])
        truth = pd.Series([cat in tc for tc in true_categories])

        tp = int((matched & truth).sum())
        fp = int((matched & ~truth).sum())
        fn = int((~matched & truth).sum())
        tn = int((~matched & ~truth).sum())
        precision, recall, f1 = prf1(tp, fp, fn)

        cat_rows.append({
            "category": cat, "support": int(truth.sum()),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": round(precision, 3), "recall": round(recall, 3), "f1": round(f1, 3),
        })
    cat_df = pd.DataFrame(cat_rows)
    cat_df.to_csv(out_dir / "per_broad_class_metrics.csv", index=False)

    # ── Overall summary ─────────────────────────────────────────────────────
    macro_f1_label = label_df["f1"].mean()
    macro_f1_cat = cat_df["f1"].mean()
    micro_tp, micro_fp, micro_fn = label_df["tp"].sum(), label_df["fp"].sum(), label_df["fn"].sum()
    micro_p, micro_r, micro_f1 = prf1(micro_tp, micro_fp, micro_fn)

    # "any group at all" -- mirrors step 1's binary mention-detection task
    any_matched = pd.Series([len(pc) > 0 for pc in predicted_categories])
    any_truth = pd.Series([len(tc) > 0 for tc in true_categories])
    any_tp = int((any_matched & any_truth).sum())
    any_fp = int((any_matched & ~any_truth).sum())
    any_fn = int((~any_matched & any_truth).sum())
    any_p, any_r, any_f1 = prf1(any_tp, any_fp, any_fn)

    lines = [
        f"alpha0-mode: {args.alpha0_mode}  (alpha0 = {alpha0_info['alpha0']:.0f}, "
        f"{alpha0_info['description']})",
        f"Test set size: {len(test):,}",
        "",
        f"Per-specific-category (57)  macro F1 = {macro_f1_label:.3f}",
        f"Per-specific-category (57)  micro P/R/F1 = {micro_p:.3f} / {micro_r:.3f} / {micro_f1:.3f}",
        f"Per-broad-class (8)         macro F1 = {macro_f1_cat:.3f}",
        "",
        "Any-group detection (mirrors step 1's binary task, full test set):",
        f"  precision={any_p:.3f}  recall={any_r:.3f}  f1={any_f1:.3f}  "
        f"(support={int(any_truth.sum())}/{len(test)} sentences have >=1 label)",
    ]
    summary_text = "\n".join(lines)
    (out_dir / "summary.txt").write_text(summary_text, encoding="utf-8")
    print(summary_text)
    print(f"\nDone. Metrics written to: {out_dir}")


if __name__ == "__main__":
    main()
