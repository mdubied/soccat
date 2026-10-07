#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SOCCAT: Social Group Mention Detection
mDeBERTa-v3 Fine-Tuning & Evaluation Pipeline

Evaluation metrics, computed overall and per outlet / country / decade:
  - accuracy
  - positive-class (sentence mentions a social group) precision / recall / F1
  - specificity (TNR), balanced accuracy, MCC, Cohen's kappa
  - ROC-AUC and PR-AUC (average precision) from the positive-class probability
  - macro F1 and support-weighted precision / recall / F1. With imbalanced
    classes (910/1260 test sentences have no group mention), the weighted
    scores are dominated by the negative class and weighted recall always
    equals accuracy; report the positive-class scores instead.
  - confusion-matrix counts (TP, FN, FP, TN), N_pos and prevalence
All slices are also written to one long-format CSV (--all_levels_csv,
default data/model_performance/step_1/performance_all_levels.csv).

Usage
-----
# Full training + evaluation:
    python SOCCAT_mDeBERTa_replication.py --train_path ../../data/model_performance/step_1/train_with_all_outlets.json --test_path ../../data/model_performance/step_1/test_with_all_outlets.json --output_dir output/

# Inference only (load fine-tuned model from the Hub, skip training):
    python SOCCAT_mDeBERTa_replication.py --test_path ../../data/model_performance/step_1/test_with_all_outlets.json --output_dir output/ --inference_only --model_name selsar/social_group_detection
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns

from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    set_seed,
)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    precision_recall_curve,
    auc,
    cohen_kappa_score,
    balanced_accuracy_score,
    matthews_corrcoef,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
)


# ── Configuration defaults ────────────────────────────────────────────────────
DEFAULT_BASE_MODEL = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"
HF_FINETUNED_MODEL = "selsar/social_group_detection"
SEED               = 98
LEARNING_RATE      = 2e-5
TRAIN_BATCH_SIZE   = 8
EVAL_BATCH_SIZE    = 8
NUM_EPOCHS         = 4
WEIGHT_DECAY       = 0.01
WARMUP_RATIO       = 0.06


# ── IO helpers ────────────────────────────────────────────────────────────────
def read_json_lines(filepath):
    """Read a JSONL file into a list of dicts."""
    with open(filepath, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def process_data(data):
    """Return list of {text, label} dicts with integer labels."""
    return [{"text": item["text"], "label": int(bool(item["label"]))} for item in data]


# ── Metadata helpers ──────────────────────────────────────────────────────────
def extract_year(row):
    """Extract a 4-digit year from an explicit year field or a date string."""
    y = row.get("year")
    if pd.notna(y):
        try:
            return int(float(y))
        except (TypeError, ValueError):
            pass
    d = row.get("date")
    if d is None or pd.isna(d):
        return None
    m = re.search(r"(19|20)\d{2}", str(d))
    return int(m.group(0)) if m else None


def year_to_decade(y):
    if y is None or pd.isna(y):
        return "Unknown"
    return f"{(int(y) // 10) * 10}s"


def map_country(lang):
    return {"French": "France", "German": "Germany"}.get(lang, "Unknown")


# ── Trainer metrics callback ──────────────────────────────────────────────────
def compute_metrics_trainer(p):
    """Called by HF Trainer after each evaluation epoch."""
    preds, labels = p.predictions, p.label_ids
    pred_labels = preds.argmax(axis=1)

    try:
        n_classes = preds.shape[1]
        y_true_bin = np.eye(n_classes)[labels]
        prec_curve, rec_curve, _ = precision_recall_curve(y_true_bin.ravel(), preds.ravel())
        pr_auc = auc(rec_curve, prec_curve)
    except Exception:
        pr_auc = float("nan")

    return {
        "accuracy":           accuracy_score(labels, pred_labels),
        "precision_weighted": precision_score(labels, pred_labels, average="weighted", zero_division=0),
        "recall_weighted":    recall_score(labels, pred_labels, average="weighted", zero_division=0),
        "f1_weighted":        f1_score(labels, pred_labels, average="weighted", zero_division=0),
        "f1_macro":           f1_score(labels, pred_labels, average="macro", zero_division=0),
        "f1_micro":           f1_score(labels, pred_labels, average="micro", zero_division=0),
        "cohen_kappa":        cohen_kappa_score(labels, pred_labels),
        "pr_auc":             pr_auc,
    }


# ── Sliced evaluation helpers ─────────────────────────────────────────────────
def _safe(fn, *args):
    """Threshold-free scores are undefined when a slice has only one class."""
    try:
        return fn(*args)
    except ValueError:
        return float("nan")


def metrics_for_group(df):
    y_true = df["true"].astype(int)
    y_pred = df["pred"].astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "Accuracy":           accuracy_score(y_true, y_pred),
        # Positive class = sentence mentions a social group
        "Precision_binary":   precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "Recall_binary":      recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        "F1_binary":          f1_score(y_true, y_pred, pos_label=1, zero_division=0),
        "Specificity":        tn / (tn + fp) if (tn + fp) else float("nan"),
        "Balanced_accuracy":  balanced_accuracy_score(y_true, y_pred) if y_true.nunique() > 1 else float("nan"),
        "MCC":                matthews_corrcoef(y_true, y_pred),
        "Cohen_kappa":        cohen_kappa_score(y_true, y_pred),
        "ROC_AUC":            _safe(roc_auc_score, y_true, df["prob_pos"]),
        "PR_AUC":             _safe(average_precision_score, y_true, df["prob_pos"]),
        "Precision_weighted": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "Recall_weighted":    recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "F1_weighted":        f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "F1_macro":           f1_score(y_true, y_pred, average="macro", zero_division=0),
        "TP":                 int(tp),
        "FN":                 int(fn),
        "FP":                 int(fp),
        "TN":                 int(tn),
        "N_pos":              int(tp + fn),
        "Prevalence":         (tp + fn) / len(df),
        "N":                  len(df),
    }


def compute_sliced_metrics(df, group_cols):
    rows = []
    for keys, grp in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        rec = metrics_for_group(grp)
        for col, val in zip(group_cols, keys):
            rec[col] = val
        rows.append(rec)
    return pd.DataFrame(rows)


# ── Visualisation ─────────────────────────────────────────────────────────────
def plot_results(outlet_df, country_df, results_df, output_dir):
    sns.set_theme(style="whitegrid")

    plot_df = outlet_df.merge(
        results_df[["paper", "country"]].drop_duplicates(), on="paper", how="left"
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, max(4, len(plot_df) * 0.45)))
    for ax, metric, title in zip(
        axes,
        ["F1_weighted", "Accuracy"],
        ["Weighted F1 per Outlet", "Accuracy per Outlet"],
    ):
        sns.barplot(data=plot_df, x=metric, y="paper", hue="country", dodge=False, ax=ax)
        ax.set_xlim(0.7, 1.0)
        ax.set_title(title)
        ax.set_xlabel(metric)
        ax.set_ylabel("Outlet")
    plt.tight_layout()
    plt.savefig(output_dir / "performance_by_outlet.pdf", bbox_inches="tight")
    plt.close()

    country_melted = country_df.melt(
        id_vars="country",
        value_vars=["Accuracy", "F1_weighted", "F1_macro"],
        var_name="Metric",
        value_name="Score",
    )
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(data=country_melted, x="Metric", y="Score", hue="country", ax=ax)
    ax.set_ylim(0.7, 1.0)
    ax.set_title("Performance by Country")
    plt.tight_layout()
    plt.savefig(output_dir / "performance_by_country.pdf", bbox_inches="tight")
    plt.close()

    print(f"Figures saved to: {output_dir}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="SOCCAT mDeBERTa fine-tuning and evaluation pipeline."
    )
    parser.add_argument("--train_path",     type=Path, default=None,
                        help="JSONL training file (required unless --inference_only)")
    parser.add_argument("--test_path",      type=Path, required=True,
                        help="JSONL test file")
    parser.add_argument("--output_dir",     type=Path, default=Path("output"),
                        help="Directory for checkpoints, predictions, and tables")
    parser.add_argument("--model_name",     type=str,  default=DEFAULT_BASE_MODEL,
                        help="HF model name/path. In --inference_only mode defaults "
                             f"to '{HF_FINETUNED_MODEL}'")
    parser.add_argument("--inference_only", action="store_true",
                        help="Skip training; load fine-tuned model from Hub")
    parser.add_argument("--seed",           type=int,  default=SEED)
    parser.add_argument("--epochs",         type=int,  default=NUM_EPOCHS)
    parser.add_argument("--lr",             type=float, default=LEARNING_RATE)
    parser.add_argument("--no_plots",       action="store_true",
                        help="Skip saving visualisation figures")
    parser.add_argument("--all_levels_csv", type=Path,
                        default=Path(__file__).resolve().parents[2] / "data" / "model_performance"
                                / "step_1" / "performance_all_levels.csv",
                        help="Long-format CSV with all slices (overall/outlet/country/decade)")
    args = parser.parse_args()

    if args.inference_only and args.model_name == DEFAULT_BASE_MODEL:
        args.model_name = HF_FINETUNED_MODEL

    if not args.inference_only and args.train_path is None:
        parser.error("--train_path is required unless --inference_only is set.")

    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ── Data loading ──────────────────────────────────────────────────────────
    print("Loading data...")
    raw_test = read_json_lines(args.test_path)
    test_dataset = Dataset.from_pandas(pd.DataFrame(process_data(raw_test)))

    if not args.inference_only:
        raw_train = read_json_lines(args.train_path)
        train_dataset = Dataset.from_pandas(pd.DataFrame(process_data(raw_train)))
        dataset = DatasetDict({"train": train_dataset, "test": test_dataset})
        print(f"Train: {len(train_dataset):,} | Test: {len(test_dataset):,}")
    else:
        dataset = DatasetDict({"test": test_dataset})
        print(f"Test: {len(test_dataset):,}")

    # ── Tokenisation ──────────────────────────────────────────────────────────
    print(f"Loading tokenizer from '{args.model_name}'...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenized = dataset.map(
        lambda x: tokenizer(x["text"], padding="max_length", truncation=True),
        batched=True,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    print(f"Loading model from '{args.model_name}'...")
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=2,
        ignore_mismatched_sizes=True,
    )

    if not args.inference_only:
        # Replace NLI head with a binary classification head
        model.classifier = torch.nn.Linear(model.config.hidden_size, 2)
        model.num_labels = 2

    # ── Training ──────────────────────────────────────────────────────────────
    if not args.inference_only:
        print("Starting training...")
        training_args = TrainingArguments(
            output_dir=str(args.output_dir / "checkpoints"),
            logging_dir=str(args.output_dir / "logs"),
            eval_strategy="epoch",
            learning_rate=args.lr,
            per_device_train_batch_size=TRAIN_BATCH_SIZE,
            per_device_eval_batch_size=EVAL_BATCH_SIZE,
            num_train_epochs=args.epochs,
            weight_decay=WEIGHT_DECAY,
            warmup_ratio=WARMUP_RATIO,
            seed=args.seed,
            report_to=[],
        )
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized["train"],
            eval_dataset=tokenized["test"],
            compute_metrics=compute_metrics_trainer,
        )
        trainer.train()

        model_save_path = args.output_dir / "mDeBERTa_social_group_detection"
        model.save_pretrained(model_save_path)
        tokenizer.save_pretrained(model_save_path)
        print(f"Model saved to: {model_save_path}")
    else:
        # Inference-only: minimal Trainer with no training config needed
        trainer = Trainer(
            model=model,
            args=TrainingArguments(
                output_dir=str(args.output_dir / "tmp"),
                per_device_eval_batch_size=EVAL_BATCH_SIZE,
                report_to=[],
            ),
        )

    # ── Inference ─────────────────────────────────────────────────────────────
    print("Running inference on test set...")
    pred_output = trainer.predict(tokenized["test"])
    preds = np.argmax(pred_output.predictions, axis=1)
    probs = torch.softmax(torch.tensor(pred_output.predictions), dim=1)[:, 1].numpy()

    results_df = pd.DataFrame(raw_test)
    results_df["pred"]       = preds
    results_df["prob_pos"]   = probs
    results_df["true"]       = test_dataset["label"]
    results_df["language"]   = results_df.get("language", pd.Series(dtype=str)).fillna("Unknown")
    results_df["country"]    = results_df["language"].map(map_country)
    results_df["paper"]      = results_df.get("paper", pd.Series(dtype=str)).fillna("Unknown")
    results_df["year_clean"] = results_df.apply(extract_year, axis=1)
    results_df["decade"]     = results_df["year_clean"].apply(year_to_decade)

    # ── Sliced evaluation ─────────────────────────────────────────────────────
    print("Computing evaluation metrics...")
    overall_df       = pd.DataFrame([metrics_for_group(results_df)])
    outlet_df        = compute_sliced_metrics(results_df, ["paper"]).sort_values("F1_weighted", ascending=False)
    country_df       = compute_sliced_metrics(results_df, ["country"]).sort_values("F1_weighted", ascending=False)
    decade_df        = compute_sliced_metrics(results_df, ["decade"]).sort_values("decade")
    outlet_decade_df = compute_sliced_metrics(results_df, ["paper", "decade"])

    print("\n=== Overall ===")
    print(overall_df.to_string(index=False))
    print("\n=== By Outlet ===")
    print(outlet_df.to_string(index=False))

    # ── Save results ──────────────────────────────────────────────────────────
    results_df.to_csv(args.output_dir / "test_predictions.csv", index=False)
    overall_df.to_csv(args.output_dir / "performance_overall.csv", index=False)
    outlet_df.to_csv(args.output_dir / "performance_per_outlet.csv", index=False)
    country_df.to_csv(args.output_dir / "performance_per_country.csv", index=False)
    decade_df.to_csv(args.output_dir / "performance_per_decade.csv", index=False)
    outlet_decade_df.to_csv(args.output_dir / "performance_outlet_x_decade.csv", index=False)

    all_levels_df = pd.concat([
        overall_df.assign(group_type="overall", group="ALL"),
        outlet_df.rename(columns={"paper": "group"}).assign(group_type="outlet"),
        country_df.rename(columns={"country": "group"}).assign(group_type="country"),
        decade_df.rename(columns={"decade": "group"}).assign(group_type="decade"),
    ], ignore_index=True)
    args.all_levels_csv.parent.mkdir(parents=True, exist_ok=True)
    all_levels_df.to_csv(args.all_levels_csv, index=False)
    print(f"All-levels table saved to: {args.all_levels_csv}")

    with pd.ExcelWriter(args.output_dir / "model_performance_summary.xlsx") as writer:
        overall_df.to_excel(writer, sheet_name="Overall", index=False)
        outlet_df.to_excel(writer, sheet_name="Per_Outlet", index=False)
        country_df.to_excel(writer, sheet_name="Per_Country", index=False)
        decade_df.to_excel(writer, sheet_name="Per_Decade", index=False)
        outlet_decade_df.to_excel(writer, sheet_name="Outlet_x_Decade", index=False)

    print(f"\nAll results saved to: {args.output_dir}")

    # ── Visualisation ─────────────────────────────────────────────────────────
    if not args.no_plots:
        plot_results(outlet_df, country_df, results_df, args.output_dir)


if __name__ == "__main__":
    main()
