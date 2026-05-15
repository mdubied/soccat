#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SOCCAT: Social Categories Mention Detection

=================================================================
Fine-tunes MoritzLaurer/mDeBERTa-v3-base-mnli-xnli with stratified K-fold CV
on prepared sentence–hypothesis NLI pairs for binary socio-economic position
mention detection.

Usage
-----
# Full CV training:
    python CV_Socio_Economic_Position.py \\
        --pairs  data/nli_dataset_full.csv \\
        --out    output/socio_economic_position/

# With Hub upload after training:
    python CV_Socio_Economic_Position.py \\
        --pairs        data/nli_dataset_full.csv \\
        --out          output/socio_economic_position/ \\
        --push_to_hub  selsar/cv_socio_economic_position

# Key optional flags:
    --kfolds 5            Number of CV folds (default: 5)
    --epochs 3            Training epochs per fold (default: 3)
    --sel_metric f1_binary  Metric used to select best fold (default: f1_binary)
    --downsample_neg        Activate negative downsampling
    --no_plots              Skip saving visualisation figures

Input format
------------
CSV or XLSX with required columns:
    sentence_id, premise, hypothesis, nli_label, hypothesis_label, outlet, country
    plus one of: year, date

"""

import argparse
import csv
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

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
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)


# ── Configuration dataclass ───────────────────────────────────────────────────
@dataclass
class Config:
    pairs_path:      str
    out_dir:         str
    model_name:      str   = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"
    seed:            int   = 42
    k_folds:         int   = 5
    max_length:      int   = 256
    epochs:          int   = 3
    lr:              float = 2e-5
    batch_train:     int   = 16
    batch_eval:      int   = 32
    downsample_neg:  bool  = False
    neg_per_pos_cap: int   = 2
    sel_metric:      str   = "f1_binary"
    push_to_hub:     Optional[str] = None   # HF repo id, e.g. "selsar/cv_sep"

    @property
    def cv_models_dir(self): return os.path.join(self.out_dir, "cv_models")
    @property
    def cv_results_dir(self): return os.path.join(self.out_dir, "cv_results")
    @property
    def best_model_dir(self): return os.path.join(self.out_dir, "best_model")


# ── Required columns ──────────────────────────────────────────────────────────
REQ_CORE = ["sentence_id", "premise", "hypothesis", "nli_label", "hypothesis_label"]
REQ_META = ["outlet", "country"]


# ── IO helpers ────────────────────────────────────────────────────────────────
def ensure_dirs(cfg: Config):
    for d in [cfg.out_dir, cfg.cv_models_dir, cfg.cv_results_dir, cfg.best_model_dir]:
        os.makedirs(d, exist_ok=True)


def load_pairs(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext in [".xlsx", ".xls"]:
        df = pd.read_excel(path, engine="openpyxl")
    else:
        with open(path, "r", encoding="utf-8", newline="") as f:
            dialect = csv.Sniffer().sniff(f.read(4096), delimiters=",;\t|")
        df = pd.read_csv(path, delimiter=dialect.delimiter)

    missing = [c for c in REQ_CORE if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    miss_meta = [c for c in REQ_META if c not in df.columns]
    if miss_meta:
        raise ValueError(f"Missing required metadata columns: {miss_meta}")
    if "year" not in df.columns and "date" not in df.columns:
        raise ValueError("Pairs file must contain 'year' or 'date'.")

    df["sentence_id"] = df["sentence_id"].astype(int)
    df["nli_label"]   = df["nli_label"].astype(int)

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        if "year" not in df.columns:
            df["year"] = df["date"].dt.year.astype("Int64")

    if "date" in df.columns:
        mask = df["date"].notna()
        df["decade"] = pd.NA
        if mask.any():
            df.loc[mask, "decade"] = (df.loc[mask, "date"].dt.year // 10 * 10).astype("Int64")
    else:
        df["decade"] = (pd.to_numeric(df["year"], errors="coerce") // 10 * 10).astype("Int64")

    return df


# ── Dataset class ─────────────────────────────────────────────────────────────
class NLIDataset:
    """Premise–hypothesis NLI dataset for the HF Trainer."""
    def __init__(self, df: pd.DataFrame, tokenizer, max_length: int):
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


# ── Stratified K-fold ─────────────────────────────────────────────────────────
def _target_per_fold(count: int, k: int) -> List[int]:
    base, rem = divmod(count, k)
    return [base + (1 if i < rem else 0) for i in range(k)]


def iterative_stratified_kfold(
    pos_long: pd.DataFrame, labels: List[str], k: int, seed: int = 42
) -> pd.DataFrame:
    """
    Assign each sentence_id to exactly one fold such that positive label
    frequency is as balanced as possible across folds (rare-first greedy).
    """
    rng      = random.Random(seed)
    sents    = sorted(pos_long["sentence_id"].unique())
    lab_map  = {sid: set(pos_long.loc[pos_long.sentence_id == sid, "lab"]) for sid in sents}
    lab_counts = {L: int((pos_long["lab"] == L).sum()) for L in labels}
    target   = {L: _target_per_fold(lab_counts[L], k) for L in labels}
    cur      = {L: [0] * k for L in labels}
    assign   = {sid: None for sid in sents}
    remaining  = set(sents)
    label_order = sorted(labels, key=lambda L: lab_counts[L])  # rare-first

    while remaining:
        best_def, best_lab, best_fold = -10**9, None, None
        for L in label_order:
            for f in range(k):
                deficit = target[L][f] - cur[L][f]
                if deficit > best_def:
                    best_def, best_lab, best_fold = deficit, L, f
        if best_def <= 0:
            for i, sid in enumerate(sorted(remaining)):
                assign[sid] = i % k
            break
        cands = [sid for sid in remaining if best_lab in lab_map[sid]]
        if not cands:
            cur[best_lab][best_fold] = target[best_lab][best_fold]
            continue

        def score(sid):
            return sum(1 for L in lab_map[sid] if (target[L][best_fold] - cur[L][best_fold]) > 0)

        scored = [(sid, score(sid)) for sid in cands]
        rng.shuffle(scored)
        best_sid = max(scored, key=lambda x: x[1])[0]
        assign[best_sid] = best_fold
        remaining.remove(best_sid)
        for L in lab_map[best_sid]:
            cur[L][best_fold] += 1

    return pd.DataFrame({"sentence_id": list(assign.keys()), "fold": list(assign.values())})


def export_fold_shares(pos_long: pd.DataFrame, fold_map: pd.DataFrame, out_csv: str):
    rows = []
    for f in sorted(fold_map["fold"].unique()):
        rep = (
            pos_long.merge(fold_map, on="sentence_id")
            .assign(split=lambda d: np.where(d["fold"] == f, "test", "train"))
            .groupby(["lab", "split"]).size().unstack(fill_value=0)
            .assign(
                total=lambda d: d.train + d.test,
                test_share=lambda d: np.where(d.total > 0, d.test / d.total, np.nan),
            )
            .reset_index().assign(fold=f)
        )
        rows.append(rep)
    pd.concat(rows, ignore_index=True).to_csv(out_csv, index=False)


# ── Train / test split builders ───────────────────────────────────────────────
def build_train_pairs(all_pairs: pd.DataFrame, train_ids: set, cfg: Config) -> pd.DataFrame:
    train = all_pairs[all_pairs.sentence_id.isin(train_ids)].copy()
    if cfg.downsample_neg:
        keep = []
        for _, sub in train.groupby("sentence_id"):
            pos = sub[sub.nli_label == 0]
            neg = sub[sub.nli_label == 1]
            max_negs = cfg.neg_per_pos_cap * max(1, len(pos))
            if len(neg) > max_negs:
                neg = neg.sample(n=max_negs, random_state=cfg.seed)
            keep.append(pd.concat([pos, neg], ignore_index=True))
        if keep:
            train = pd.concat(keep, ignore_index=True).sample(frac=1.0, random_state=cfg.seed).reset_index(drop=True)
    return train


def build_test_pairs(all_pairs: pd.DataFrame, test_ids: set) -> pd.DataFrame:
    return all_pairs[all_pairs.sentence_id.isin(test_ids)].copy()


# ── Metrics ───────────────────────────────────────────────────────────────────
def softmax(logits: np.ndarray) -> np.ndarray:
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def compute_all_metrics(y_true: np.ndarray, prob_entail: np.ndarray) -> Dict[str, float]:
    """
    label convention: 0 = entailment (positive class), 1 = not_entailment.
    prob_entail is the softmax probability for class 0.
    """
    y_pred     = np.where(prob_entail > 0.5, 0, 1)
    pos_mask   = (y_true == 0)
    prevalence = float(pos_mask.mean())
    y_bin      = pos_mask.astype(int)

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
        prevalence        = prevalence,
    )
    try:
        out["roc_auc"] = roc_auc_score(y_bin, prob_entail)
    except Exception:
        out["roc_auc"] = np.nan
    try:
        out["pr_auc"] = average_precision_score(y_bin, prob_entail)
    except Exception:
        out["pr_auc"] = np.nan
    return out


def per_group_metrics(
    df_pairs: pd.DataFrame, prob_entail: np.ndarray, group_col: str
) -> pd.DataFrame:
    tmp = df_pairs.copy()
    tmp["prob_entail"] = prob_entail
    rows = []
    for g, sub in tmp.groupby(group_col, dropna=False):
        yt = sub["nli_label"].astype(int).values
        pe = sub["prob_entail"].values
        m  = compute_all_metrics(yt, pe)
        m.update({group_col: g, "n_pairs": int(len(sub)), "n_pos_entail": int((yt == 0).sum())})
        rows.append(m)
    return pd.DataFrame(rows)


def add_time_fields(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    if "year" not in d.columns and "date" in d.columns:
        d["date"] = pd.to_datetime(d["date"], errors="coerce")
        d["year"] = d["date"].dt.year.astype("Int64")
    if "decade" not in d.columns:
        if "date" in d.columns:
            d["date"] = pd.to_datetime(d["date"], errors="coerce")
            mask = d["date"].notna()
            d["decade"] = pd.NA
            if mask.any():
                d.loc[mask, "decade"] = (d.loc[mask, "date"].dt.year // 10 * 10).astype("Int64")
        else:
            d["decade"] = (pd.to_numeric(d.get("year", pd.NA), errors="coerce") // 10 * 10).astype("Int64")
    return d


# ── CV training loop ──────────────────────────────────────────────────────────
def run_cv(pairs: pd.DataFrame, cfg: Config):
    """Run stratified K-fold CV and return (best_model_dir, best_fold, best_metrics)."""

    labels   = sorted(pairs["hypothesis_label"].dropna().unique().tolist())
    pos_long = (
        pairs[pairs.nli_label == 0][["sentence_id", "hypothesis_label"]]
        .drop_duplicates()
        .rename(columns={"hypothesis_label": "lab"})
    )
    if pos_long.empty:
        raise RuntimeError("No positive (entailment=0) rows; cannot stratify.")

    fold_map = iterative_stratified_kfold(pos_long, labels, cfg.k_folds, seed=cfg.seed)
    fold_map.to_csv(os.path.join(cfg.out_dir, "cv_sentence_folds.csv"), index=False)
    export_fold_shares(
        pos_long, fold_map,
        os.path.join(cfg.cv_results_dir, "stratification_shares.csv"),
    )

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=True)

    def compute_metrics_fn(eval_pred):
        logits    = eval_pred.predictions[0] if isinstance(eval_pred.predictions, (tuple, list)) else eval_pred.predictions
        labels_np = eval_pred.label_ids
        prob_entail = softmax(logits)[:, 0]
        return compute_all_metrics(labels_np.astype(int), prob_entail)

    epoch_tables, all_overall = [], []

    for fold in range(cfg.k_folds):
        print(f"\n========== Fold {fold + 1}/{cfg.k_folds} ==========")
        test_ids  = set(fold_map.loc[fold_map.fold == fold, "sentence_id"])
        train_ids = set(pairs["sentence_id"]) - test_ids

        train_pairs = build_train_pairs(pairs, train_ids, cfg)
        test_pairs  = build_test_pairs(pairs, test_ids)

        ds_train = NLIDataset(train_pairs, tokenizer, cfg.max_length)
        ds_test  = NLIDataset(test_pairs,  tokenizer, cfg.max_length)

        config = AutoConfig.from_pretrained(
            cfg.model_name, num_labels=2,
            id2label={0: "entailment", 1: "not_entailment"},
            label2id={"entailment": 0, "not_entailment": 1},
            problem_type="single_label_classification",
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            cfg.model_name, config=config, ignore_mismatched_sizes=True
        )

        training_args = TrainingArguments(
            output_dir            = os.path.join(cfg.cv_models_dir, f"fold_{fold}"),
            learning_rate         = cfg.lr,
            per_device_train_batch_size = cfg.batch_train,
            per_device_eval_batch_size  = cfg.batch_eval,
            num_train_epochs      = cfg.epochs,
            eval_strategy         = "epoch",
            save_strategy         = "epoch",
            save_total_limit      = 1,
            load_best_model_at_end      = True,
            metric_for_best_model = "eval_accuracy",
            seed                  = cfg.seed,
            report_to             = [],
            logging_steps         = 50,
        )

        trainer = Trainer(
            model            = model,
            args             = training_args,
            train_dataset    = ds_train,
            eval_dataset     = ds_test,
            tokenizer        = tokenizer,
            data_collator    = DataCollatorWithPadding(tokenizer=tokenizer),
            compute_metrics  = compute_metrics_fn,
        )
        trainer.train()

        best_ckpt = trainer.state.best_model_checkpoint or trainer.args.output_dir

        # Epoch loss table
        logs = pd.DataFrame(trainer.state.log_history)
        logs = logs[logs["epoch"].notna()].copy()
        tr   = (logs[logs["loss"].notna()].groupby("epoch", as_index=False)["loss"].last()
                .rename(columns={"loss": "training_loss"})
                if "loss" in logs.columns else pd.DataFrame(columns=["epoch", "training_loss"]))
        ev   = (logs[logs["eval_loss"].notna()].groupby("epoch", as_index=False)["eval_loss"].last()
                .rename(columns={"eval_loss": "validation_loss"})
                if "eval_loss" in logs.columns else pd.DataFrame(columns=["epoch", "validation_loss"]))
        epoch_tbl = pd.merge(tr, ev, on="epoch", how="outer").sort_values("epoch")
        epoch_tbl["fold"] = fold
        epoch_tbl.to_csv(os.path.join(cfg.cv_results_dir, f"fold_{fold}_epoch_losses.csv"), index=False)
        epoch_tables.append(epoch_tbl)

        # Held-out predictions
        pred        = trainer.predict(ds_test)
        logits      = pred.predictions[0] if isinstance(pred.predictions, (tuple, list)) else pred.predictions
        y_true      = pred.label_ids.astype(int)
        prob_entail = softmax(logits)[:, 0]
        y_pred      = np.where(prob_entail > 0.5, 0, 1)

        overall          = compute_all_metrics(y_true, prob_entail)
        overall["fold"]      = fold
        overall["best_ckpt"] = best_ckpt
        with open(os.path.join(cfg.cv_results_dir, f"fold_{fold}_metrics.json"), "w") as f:
            json.dump(overall, f, indent=2)
        all_overall.append(overall)

        # Per-group diagnostics
        tp = add_time_fields(test_pairs.copy())
        tp["prob_entail"] = prob_entail
        tp["pred_label"]  = y_pred

        def dump(df, name):
            df.to_csv(os.path.join(cfg.cv_results_dir, f"fold_{fold}_per_{name}.csv"), index=False)

        dump(per_group_metrics(tp, prob_entail, "hypothesis_label"), "label")
        dump(per_group_metrics(tp, prob_entail, "outlet"),           "outlet")
        dump(per_group_metrics(tp, prob_entail, "country"),          "country")
        dump(per_group_metrics(tp, prob_entail, "year"),             "year")
        dump(per_group_metrics(tp, prob_entail, "decade"),           "decade")

        output_cols = ["sentence_id", "premise", "hypothesis", "nli_label", "pred_label",
                       "prob_entail", "hypothesis_label", "outlet", "country", "year"]
        if "date" in tp.columns:
            output_cols.insert(6, "date")
        tp[output_cols].to_csv(
            os.path.join(cfg.cv_results_dir, f"fold_{fold}_human_vs_model.csv"), index=False
        )

    # CV summary
    metric_keys = ["accuracy", "precision_binary", "recall_binary", "f1_binary",
                   "precision_micro", "recall_micro", "f1_micro", "f1_macro",
                   "pr_auc", "roc_auc", "cohen_kappa"]
    summary = {k: float(np.nanmean([m.get(k, np.nan) for m in all_overall])) for k in metric_keys}
    with open(os.path.join(cfg.cv_results_dir, "summary_overall.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("\n==== CV SUMMARY ====\n", summary)

    if epoch_tables:
        epochs_all = pd.concat(epoch_tables, ignore_index=True)
        epochs_all.rename(columns={
            "epoch": "Epoch",
            "training_loss": "Training Loss",
            "validation_loss": "Validation Loss",
        }, inplace=True)
        epochs_all.to_csv(os.path.join(cfg.cv_results_dir, "epochs_all_folds.csv"), index=False)

    # Select best fold
    def _metric_val(m, key):
        v = m.get(key)
        return float("-inf") if v is None or (isinstance(v, float) and np.isnan(v)) else float(v)

    best_rec  = max(all_overall, key=lambda m: _metric_val(m, cfg.sel_metric))
    best_fold = int(best_rec["fold"])
    best_ckpt = best_rec["best_ckpt"]
    print(f"\nSelected best fold: {best_fold} | {cfg.sel_metric} = {best_rec[cfg.sel_metric]:.4f}")

    best_model = AutoModelForSequenceClassification.from_pretrained(best_ckpt)
    best_model.save_pretrained(cfg.best_model_dir)
    tokenizer.save_pretrained(cfg.best_model_dir)

    with open(os.path.join(cfg.best_model_dir, "cv_selection.json"), "w") as f:
        to_dump = {k: (float(v) if isinstance(v, (np.floating, float)) else v)
                   for k, v in best_rec.items() if k != "best_ckpt"}
        to_dump.update({"selected_metric": cfg.sel_metric, "best_checkpoint": best_ckpt})
        json.dump(to_dump, f, indent=2)

    return cfg.best_model_dir, best_fold, best_rec


# ── Hub push (optional) ───────────────────────────────────────────────────────
def push_to_hub(best_model_dir: str, repo_id: str):
    from huggingface_hub import create_repo

    print(f"\nPushing to https://huggingface.co/{repo_id} ...")
    try:
        create_repo(repo_id, exist_ok=True)
    except Exception as e:
        print(f"create_repo warning: {e}")

    model     = AutoModelForSequenceClassification.from_pretrained(best_model_dir)
    tokenizer = AutoTokenizer.from_pretrained(best_model_dir)
    model.push_to_hub(repo_id)
    tokenizer.push_to_hub(repo_id)
    print(f"Pushed to https://huggingface.co/{repo_id}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="SOCCAT socio-economic position NLI cross-validation pipeline."
    )
    parser.add_argument("--pairs",           type=str, required=True,
                        help="Path to NLI pairs file (.csv or .xlsx)")
    parser.add_argument("--out",             type=str, default="output",
                        help="Output directory")
    parser.add_argument("--model",           type=str,
                        default="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli")
    parser.add_argument("--kfolds",          type=int, default=5)
    parser.add_argument("--seed",            type=int, default=42)
    parser.add_argument("--epochs",          type=int, default=3)
    parser.add_argument("--max_length",      type=int, default=256)
    parser.add_argument("--lr",              type=float, default=2e-5)
    parser.add_argument("--b_train",         type=int, default=16)
    parser.add_argument("--b_eval",          type=int, default=32)
    parser.add_argument("--downsample_neg",  action="store_true")
    parser.add_argument("--neg_per_pos_cap", type=int, default=2)
    parser.add_argument("--sel_metric",      type=str, default="f1_binary",
                        help="Metric used to select best fold (e.g. f1_binary, accuracy)")
    parser.add_argument("--push_to_hub",     type=str, default=None, metavar="REPO_ID",
                        help="HF Hub repo id to push best model to (e.g. selsar/cv_sep). "
                             "Requires HF_TOKEN env var or prior huggingface-cli login.")
    args = parser.parse_args()

    cfg = Config(
        pairs_path      = args.pairs,
        out_dir         = args.out,
        model_name      = args.model,
        seed            = args.seed,
        k_folds         = args.kfolds,
        max_length      = args.max_length,
        epochs          = args.epochs,
        lr              = args.lr,
        batch_train     = args.b_train,
        batch_eval      = args.b_eval,
        downsample_neg  = args.downsample_neg,
        neg_per_pos_cap = args.neg_per_pos_cap,
        sel_metric      = args.sel_metric,
        push_to_hub     = args.push_to_hub,
    )

    # Reproducibility
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    set_seed(cfg.seed)

    ensure_dirs(cfg)

    pairs = load_pairs(cfg.pairs_path)
    labels_found = sorted(pairs["hypothesis_label"].dropna().unique().tolist())
    with open(os.path.join(cfg.out_dir, "labels_found.json"), "w") as f:
        json.dump(labels_found, f, indent=2)
    print(f"Labels found: {labels_found}")
    print(f"Total pairs:  {len(pairs):,}")

    best_model_dir, best_fold, best_rec = run_cv(pairs, cfg)

    if cfg.push_to_hub:
        push_to_hub(best_model_dir, cfg.push_to_hub)

    print("\nDone. Outputs in:", cfg.out_dir)


if __name__ == "__main__":
    main()
