#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SOCCAT — Replicate Results from Hugging Face Hub
=================================================
Loads each fine-tuned model from the Hub and runs inference on the
corresponding held-out test data. Reproduces the evaluation tables
reported in the paper without requiring retraining.

Usage
-----
# Reproduce all categories:
    python replicate_from_hub.py \\
        --data_root data/ \\
        --out       output/replication/

# Single category:
    python replicate_from_hub.py \\
        --data_root data/ \\
        --out       output/replication/ \\
        --category  socio_economic_position

# Use a custom categories manifest:
    python replicate_from_hub.py \\
        --manifest  categories.json \\
        --data_root data/ \\
        --out       output/replication/

Data format
-----------
Each file in data_root must be a CSV or XLSX with columns:
    sentence_id, premise, hypothesis, nli_label, hypothesis_label,
    outlet, country  +  one of: year, date

"""

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    cohen_kappa_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments


# ── Dataset ───────────────────────────────────────────────────────────────────
class NLIDataset:
    def __init__(self, df: pd.DataFrame, tokenizer, max_length: int = 256):
        self.df         = df.reset_index(drop=True)
        self.tokenizer  = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        enc = self.tokenizer(
            row["premise"], row["hypothesis"],
            truncation=True, max_length=self.max_length,
        )
        enc["labels"] = int(row["nli_label"])
        return enc


# ── IO ────────────────────────────────────────────────────────────────────────
def load_data(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext in [".xlsx", ".xls"]:
        df = pd.read_excel(path, engine="openpyxl")
    else:
        with open(path, "r", encoding="utf-8", newline="") as f:
            dialect = csv.Sniffer().sniff(f.read(4096), delimiters=",;\t|")
        df = pd.read_csv(path, delimiter=dialect.delimiter)

    df["sentence_id"] = df["sentence_id"].astype(int)
    df["nli_label"]   = df["nli_label"].astype(int)

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        if "year" not in df.columns:
            df["year"] = df["date"].dt.year.astype("Int64")

    if "decade" not in df.columns:
        if "date" in df.columns:
            mask = df["date"].notna()
            df["decade"] = pd.NA
            if mask.any():
                df.loc[mask, "decade"] = (df.loc[mask, "date"].dt.year // 10 * 10).astype("Int64")
        elif "year" in df.columns:
            df["decade"] = (pd.to_numeric(df["year"], errors="coerce") // 10 * 10).astype("Int64")

    return df


# ── Metrics ───────────────────────────────────────────────────────────────────
def softmax(logits: np.ndarray) -> np.ndarray:
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def compute_metrics(y_true: np.ndarray, prob_entail: np.ndarray) -> Dict[str, float]:
    """label convention: 0 = entailment (positive), 1 = not_entailment."""
    y_pred   = np.where(prob_entail > 0.5, 0, 1)
    pos_mask = (y_true == 0).astype(int)
    out = dict(
        accuracy          = accuracy_score(y_true, y_pred),
        precision_binary  = precision_score(y_true, y_pred, pos_label=0, zero_division=0),
        recall_binary     = recall_score(y_true, y_pred, pos_label=0),
        f1_binary         = f1_score(y_true, y_pred, pos_label=0),
        precision_micro   = precision_score(y_true, y_pred, average="micro", zero_division=0),
        recall_micro      = recall_score(y_true, y_pred, average="micro"),
        f1_micro          = f1_score(y_true, y_pred, average="micro"),
        f1_macro          = f1_score(y_true, y_pred, average="macro"),
        cohen_kappa       = cohen_kappa_score(y_true, y_pred),
        prevalence        = float(pos_mask.mean()),
    )
    try:    out["roc_auc"] = roc_auc_score(pos_mask, prob_entail)
    except: out["roc_auc"] = np.nan
    try:    out["pr_auc"]  = average_precision_score(pos_mask, prob_entail)
    except: out["pr_auc"]  = np.nan
    return out


def per_group_metrics(df: pd.DataFrame, prob_entail: np.ndarray, group_col: str) -> pd.DataFrame:
    tmp = df.copy()
    tmp["prob_entail"] = prob_entail
    rows = []
    for g, sub in tmp.groupby(group_col, dropna=False):
        yt = sub["nli_label"].astype(int).values
        pe = sub["prob_entail"].values
        m  = compute_metrics(yt, pe)
        m.update({group_col: g, "n_pairs": len(sub), "n_pos_entail": int((yt == 0).sum())})
        rows.append(m)
    return pd.DataFrame(rows)


# ── Per-category inference ────────────────────────────────────────────────────
def replicate_category(cat: dict, data_root: str, out_root: str, batch_size: int = 32):
    name       = cat["name"]
    hub_repo   = cat["hub_repo"]
    data_path  = os.path.join(data_root, os.path.basename(cat["data"]))
    out_dir    = os.path.join(out_root, name)
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Category : {cat['display_name']}")
    print(f"  Model    : https://huggingface.co/{hub_repo}")
    print(f"  Data     : {data_path}")
    print(f"{'='*60}")

    if not os.path.exists(data_path):
        print(f"  WARNING: data file not found at {data_path} — skipping.")
        return None

    df        = load_data(data_path)
    tokenizer = AutoTokenizer.from_pretrained(hub_repo, use_fast=True)
    model     = AutoModelForSequenceClassification.from_pretrained(hub_repo)

    dataset = NLIDataset(df, tokenizer)

    # Use Trainer purely for batched inference (no training config needed)
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=os.path.join(out_dir, "tmp"),
            per_device_eval_batch_size=batch_size,
            report_to=[],
        ),
    )

    pred        = trainer.predict(dataset)
    logits      = pred.predictions[0] if isinstance(pred.predictions, (tuple, list)) else pred.predictions
    prob_entail = softmax(logits)[:, 0]
    y_true      = df["nli_label"].astype(int).values
    y_pred      = np.where(prob_entail > 0.5, 0, 1)

    # Attach predictions to dataframe
    df["prob_entail"] = prob_entail
    df["pred_label"]  = y_pred

    # ── Metrics ───────────────────────────────────────────────────────────────
    overall = compute_metrics(y_true, prob_entail)
    overall["category"] = name

    slices = {}
    for col in ["hypothesis_label", "outlet", "country", "year", "decade"]:
        if col in df.columns:
            slices[col] = per_group_metrics(df, prob_entail, col)

    # ── Save ──────────────────────────────────────────────────────────────────
    df.to_csv(os.path.join(out_dir, "predictions.csv"), index=False)
    with open(os.path.join(out_dir, "metrics_overall.json"), "w") as f:
        json.dump(overall, f, indent=2)
    for col, sdf in slices.items():
        sdf.to_csv(os.path.join(out_dir, f"metrics_per_{col}.csv"), index=False)

    print(f"  F1 (binary)  : {overall['f1_binary']:.4f}")
    print(f"  F1 (macro)   : {overall['f1_macro']:.4f}")
    print(f"  Accuracy     : {overall['accuracy']:.4f}")
    print(f"  Results saved: {out_dir}")

    return overall


# ── Summary table ─────────────────────────────────────────────────────────────
def write_summary(results: List[dict], out_dir: str):
    summary = pd.DataFrame(results).set_index("category")
    cols = ["accuracy", "f1_binary", "f1_macro", "f1_micro",
            "precision_binary", "recall_binary", "cohen_kappa", "pr_auc", "roc_auc", "prevalence"]
    cols = [c for c in cols if c in summary.columns]
    summary = summary[cols].round(4)

    print("\n" + "="*60)
    print("REPLICATION SUMMARY — all categories")
    print("="*60)
    print(summary.to_string())

    summary.to_csv(os.path.join(out_dir, "summary_all_categories.csv"))

    with pd.ExcelWriter(os.path.join(out_dir, "summary_all_categories.xlsx")) as writer:
        summary.to_excel(writer, sheet_name="Summary")

    print(f"\nSummary saved to: {out_dir}/summary_all_categories.{{csv,xlsx}}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Reproduce SOCCAT evaluation tables from Hugging Face Hub models."
    )
    parser.add_argument("--manifest",   type=str, default="categories.json",
                        help="Path to categories.json (default: categories.json)")
    parser.add_argument("--data_root",  type=str, required=True,
                        help="Directory containing the NLI pair CSV files")
    parser.add_argument("--out",        type=str, default="output/replication",
                        help="Output directory for predictions and metric tables")
    parser.add_argument("--category",   type=str, default=None,
                        help="Run a single category by name (e.g. socio_economic_position)")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Eval batch size (default: 32; reduce if OOM)")
    args = parser.parse_args()

    with open(args.manifest) as f:
        manifest = json.load(f)

    if args.category:
        manifest = [c for c in manifest if c["name"] == args.category]
        if not manifest:
            raise ValueError(f"Category '{args.category}' not found in {args.manifest}.")

    os.makedirs(args.out, exist_ok=True)

    results = []
    skipped = []
    for cat in manifest:
        overall = replicate_category(cat, args.data_root, args.out, args.batch_size)
        if overall is not None:
            results.append(overall)
        else:
            skipped.append(cat["name"])

    if results:
        write_summary(results, args.out)
    if skipped:
        print(f"\nSkipped (data not found): {skipped}")


if __name__ == "__main__":
    main()
