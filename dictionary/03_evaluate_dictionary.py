#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate the dictionary baseline on the held-out test split
==============================================================
For every test sentence and every one of the 57 labels, predicts a match if
any word from that label's dictionary (dictionary/dictionaries/*.txt) is
present as a token in the sentence. Ground truth comes from test.csv's
'labels' / 'categories' columns (produced by 01_prepare_data.py).

Unlike the NLI models' within-category evaluation (each category's model is
only tested on sentences already known to be positive for that category, see
src/step_2/step_2_cv_pipeline.py), every test sentence is scored against
every label here -- there is no reason to restrict it, since dictionary
matching has no training-time coupling to a subset of sentences. This gives
a full-corpus picture, including how often the dictionary fires on
sentences that don't belong to that label/category at all.

Outputs (dictionary/output/):
  per_label_metrics.csv     precision/recall/F1/support per label
  per_category_metrics.csv  same, rolled up to broad category (match = any
                             label dictionary in that category fires)
  summary.txt               overall micro/macro summary
"""

import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
DICT_DIR = ROOT / "dictionaries"
OUT_DIR = ROOT / "output"

WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)


def tokenize(text: str) -> set:
    return set(WORD_RE.findall(str(text).lower()))


def load_dictionaries():
    """Returns {(category, label): set(words)}."""
    summary = pd.read_csv(DICT_DIR / "summary.csv")
    dicts = {}
    for _, row in summary.iterrows():
        fname = f"{row['category']}__{row['label']}".replace("/", "-")
        fname = re.sub(r'[<>:"\\|?*]', "_", fname) + ".txt"
        words = set(open(DICT_DIR / fname, encoding="utf-8").read().split())
        dicts[(row["category"], row["label"])] = words
    return dicts


def prf1(tp, fp, fn):
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


def main():
    test = pd.read_csv(DATA_DIR / "test.csv", keep_default_na=False)
    test["label_set"] = test["labels"].apply(lambda s: set(s.split(";")) if s else set())
    test["category_set"] = test["categories"].apply(lambda s: set(s.split(";")) if s else set())
    test["tokens"] = test["text"].apply(tokenize)

    dicts = load_dictionaries()

    OUT_DIR.mkdir(exist_ok=True)

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
    label_df.to_csv(OUT_DIR / "per_label_metrics.csv", index=False)

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
    cat_df.to_csv(OUT_DIR / "per_category_metrics.csv", index=False)

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
        f"Test set size: {len(test):,}",
        "",
        f"Per-label (57 labels)     macro F1 = {macro_f1_label:.3f}",
        f"Per-label (57 labels)     micro P/R/F1 = {micro_p:.3f} / {micro_r:.3f} / {micro_f1:.3f}",
        f"Per-category (8 broad)    macro F1 = {macro_f1_cat:.3f}",
        "",
        "Any-group detection (mirrors step 1's binary task, full test set):",
        f"  precision={any_p:.3f}  recall={any_r:.3f}  f1={any_f1:.3f}  "
        f"(support={int(any_truth.sum())}/{len(test)} sentences have >=1 label)",
    ]
    summary_text = "\n".join(lines)
    (OUT_DIR / "summary.txt").write_text(summary_text, encoding="utf-8")
    print(summary_text)
    print(f"\nDone. Metrics written to: {OUT_DIR}")


if __name__ == "__main__":
    main()
