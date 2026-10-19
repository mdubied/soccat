#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shared data, fold split, metrics and outputs for the embedding-based baselines
================================================================================
Used by fine_tuned_classifier/train_cv.py and zero_shot_nli/predict_cv.py so that
both are evaluated exactly like the single NLI model (single_nli/, notebook
examples/CV_combined_all_dimensions_v4.ipynb):

- Data: the (sentence, hypothesis) pairs of all 57 labels stacked, read from the
  single NLI model's fold_*_human_vs_model.csv files (PAIRS_DIR). These are the 8
  nli_dataset_full.csv files of the main models stacked: the 7,234 sentences with a
  non-empty ground truth (target_abroad = 1 removed), each paired with the 57
  hypotheses, with the same gold labels (nli_label: 0 = entailment / group present).
- Folds: the notebook's 5-fold split, stratified on the positives of all 57 labels
  jointly (seed 42), sentences without any positive dealt round-robin after a seeded
  shuffle. It is recomputed here and checked against the fold each sentence was
  tested in by the single NLI model, so all three models share the same split.
- Metrics and output files: as in the notebook (prediction = P(entailment) > 0.5).

Per-fold outputs (directly in OUT_DIR, as in single_nli/data/; same names and columns as
the notebook's cv_results/):
    fold_{k}_human_vs_model.csv   one row per test (sentence, label) pair
    fold_{k}_metrics.json         metrics on all test pairs of the fold
    fold_{k}_per_{label,outlet,country,year,decade}.csv
finalize() then writes summary_overall.json, per_label_pooled_all_folds.csv,
stratification_shares.csv, epochs_all_folds.csv (if any) and labels_found.json.
"""

import json
import random
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score, cohen_kappa_score,
                             f1_score, precision_score, recall_score, roc_auc_score)

ROOT = Path(__file__).resolve().parent
PAIRS_DIR = ROOT / "single_nli" / "data"
K_FOLDS = 5
SEED = 42
PAIR_COLS = ["sentence_id", "premise", "hypothesis", "nli_label", "hypothesis_label", "dimension",
             "label_orig", "outlet", "country", "date", "year"]
OUT_COLS = ["sentence_id", "premise", "hypothesis", "nli_label", "pred_label", "prob_entail",
            "hypothesis_label", "dimension", "label_orig", "outlet", "country", "date", "year"]
METRIC_KEYS = ["accuracy", "precision_binary", "recall_binary", "f1_binary", "precision_micro",
               "recall_micro", "f1_micro", "f1_macro", "pr_auc", "roc_auc", "cohen_kappa"]


# ── Data ────────────────────────────────────────────────────────────────────
def load_pairs(pairs_dir: Path = PAIRS_DIR):
    """All (sentence, label) pairs and the fold each sentence was tested in by the single
    NLI model (the file it appears in)."""
    parts = []
    for k in range(K_FOLDS):
        path = Path(pairs_dir) / f"fold_{k}_human_vs_model.csv"
        if not path.exists():
            raise FileNotFoundError(f"{path} not found (expected the single NLI model's fold files)")
        parts.append(pd.read_csv(path, usecols=PAIR_COLS).assign(reference_fold=k))
    pairs = pd.concat(parts, ignore_index=True)
    pairs["sentence_id"] = pairs["sentence_id"].astype(int)
    pairs["nli_label"] = pairs["nli_label"].astype(int)
    if pairs.duplicated(["sentence_id", "hypothesis_label"]).any():
        raise ValueError("duplicate (sentence_id, hypothesis_label) pairs")
    n_labels = pairs["hypothesis_label"].nunique()
    per_sentence = pairs.groupby("sentence_id").size()
    if not (per_sentence == n_labels).all():
        raise ValueError("not every sentence is paired with every label")
    if (pairs.groupby("sentence_id")["reference_fold"].nunique() > 1).any():
        raise ValueError("a sentence appears in several reference folds")
    reference = pairs.drop_duplicates("sentence_id").set_index("sentence_id")["reference_fold"]
    print(f"{len(pairs):,} pairs | {pairs['sentence_id'].nunique():,} sentences | {n_labels} labels")
    return pairs.drop(columns="reference_fold"), reference


def labels_sorted(pairs: pd.DataFrame) -> List[str]:
    return sorted(pairs["hypothesis_label"].unique().tolist())


# ── Fold split (copied from examples/CV_combined_all_dimensions_v4.ipynb) ─────
def _target_per_fold(count: int, k: int) -> List[int]:
    base, rem = divmod(count, k)
    return [base + (1 if i < rem else 0) for i in range(k)]


def iterative_stratified_kfold(pos_long: pd.DataFrame, labels: List[str], k: int, seed: int = 42) -> pd.DataFrame:
    rng = random.Random(seed)
    sents = sorted(pos_long["sentence_id"].unique())
    lab_map = pos_long.groupby("sentence_id")["lab"].agg(set).to_dict()
    lab_counts = {L: int((pos_long["lab"] == L).sum()) for L in labels}
    target = {L: _target_per_fold(lab_counts[L], k) for L in labels}
    cur = {L: [0] * k for L in labels}
    assign = {sid: None for sid in sents}
    remaining = set(sents)
    label_order = sorted(labels, key=lambda L: lab_counts[L])

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


def positives_long(pairs: pd.DataFrame) -> pd.DataFrame:
    return (pairs[pairs.nli_label == 0][["sentence_id", "hypothesis_label"]]
            .drop_duplicates().rename(columns={"hypothesis_label": "lab"}))


def make_fold_map(pairs: pd.DataFrame, k: int = K_FOLDS, seed: int = SEED) -> pd.DataFrame:
    fold_map = iterative_stratified_kfold(positives_long(pairs), labels_sorted(pairs), k, seed=seed)
    rng = random.Random(seed)
    no_pos = sorted(set(pairs["sentence_id"]) - set(fold_map["sentence_id"]))  # ascending ids, as in the notebook's file order
    rng.shuffle(no_pos)
    return pd.concat([fold_map, pd.DataFrame({"sentence_id": no_pos, "fold": [i % k for i in range(len(no_pos))]})],
                     ignore_index=True)


def get_fold_map(pairs: pd.DataFrame, reference: pd.Series, out_dir: Path) -> pd.DataFrame:
    """The notebook's fold map, checked against the single NLI model's folds and saved."""
    fold_map = make_fold_map(pairs)
    mismatch = (fold_map.set_index("sentence_id")["fold"] != reference.loc[fold_map["sentence_id"]].values).sum()
    if mismatch:
        raise ValueError(f"fold split differs from the single NLI model's for {mismatch} sentences")
    print("Fold split identical to the single NLI model's (all sentences)")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fold_map.to_csv(out_dir / "cv_sentence_folds.csv", index=False)
    return fold_map


# ── Metrics (copied from the notebook) ──────────────────────────────────────
def softmax(logits: np.ndarray) -> np.ndarray:
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def compute_all_metrics(y_true, prob_entail) -> Dict[str, float]:
    """y_true: 0 = entailment (positive class), 1 = not entailment."""
    y_pred = np.where(prob_entail > 0.5, 0, 1)
    pos_mask = (y_true == 0)
    out = dict(
        accuracy=accuracy_score(y_true, y_pred),
        precision_binary=precision_score(y_true, y_pred, pos_label=0, zero_division=0),
        recall_binary=recall_score(y_true, y_pred, pos_label=0, zero_division=0),
        f1_binary=f1_score(y_true, y_pred, pos_label=0, zero_division=0),
        precision_micro=precision_score(y_true, y_pred, average="micro", zero_division=0),
        recall_micro=recall_score(y_true, y_pred, average="micro", zero_division=0),
        f1_micro=f1_score(y_true, y_pred, average="micro", zero_division=0),
        f1_macro=f1_score(y_true, y_pred, average="macro", zero_division=0),
        cohen_kappa=cohen_kappa_score(y_true, y_pred),
        prevalence=float(pos_mask.mean()),
    )
    y_bin = pos_mask.astype(int)
    try:
        out["roc_auc"] = roc_auc_score(y_bin, prob_entail)
    except Exception:
        out["roc_auc"] = np.nan
    try:
        out["pr_auc"] = average_precision_score(y_bin, prob_entail)
    except Exception:
        out["pr_auc"] = np.nan
    return out


def per_group_metrics(df: pd.DataFrame, prob_entail, group_col: str) -> pd.DataFrame:
    tmp = df.copy()
    tmp["prob_entail"] = prob_entail
    rows = []
    for g, sub in tmp.groupby(group_col, dropna=False):
        yt = sub["nli_label"].astype(int).values
        m = compute_all_metrics(yt, sub["prob_entail"].values)
        m.update({group_col: g, "n_pairs": int(len(sub)), "n_pos_entail": int((yt == 0).sum())})
        rows.append(m)
    return pd.DataFrame(rows)


def add_time_fields(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    if "date" in d.columns:
        d["date"] = pd.to_datetime(d["date"], errors="coerce")
        if "year" not in d.columns:
            d["year"] = d["date"].dt.year.astype("Int64")
    if "decade" not in d.columns:
        if "date" in d.columns and d["date"].notna().any():
            d["decade"] = pd.NA
            m = d["date"].notna()
            d.loc[m, "decade"] = (d.loc[m, "date"].dt.year // 10 * 10).astype("Int64")
        else:
            d["decade"] = (pd.to_numeric(d.get("year", pd.NA), errors="coerce") // 10 * 10).astype("Int64")
    return d


def export_fold_shares(pos_long: pd.DataFrame, fold_map: pd.DataFrame, out_csv: Path):
    rows = []
    for f in sorted(fold_map["fold"].unique()):
        rep = (pos_long.merge(fold_map, on="sentence_id")
               .assign(split=lambda d: np.where(d["fold"] == f, "test", "train"))
               .groupby(["lab", "split"]).size().unstack(fill_value=0)
               .assign(total=lambda d: d.train + d.test,
                       test_share=lambda d: np.where(d.total > 0, d.test / d.total, np.nan))
               .reset_index().assign(fold=f))
        rows.append(rep)
    pd.concat(rows, ignore_index=True).to_csv(out_csv, index=False)


# ── Outputs ─────────────────────────────────────────────────────────────────
def cv_dir(out_dir: Path) -> Path:
    """Result files go directly into out_dir (the notebook's cv_results/ content), the
    same flat layout as single_nli/data/."""
    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    return d


def fold_done(out_dir: Path, fold: int) -> bool:
    return (cv_dir(out_dir) / f"fold_{fold}_human_vs_model.csv").exists()


def write_fold_outputs(out_dir: Path, fold: int, test_pairs: pd.DataFrame, prob_entail: np.ndarray, extra: dict = None):
    """Per-fold files, from the test pairs and P(entailment) of each pair (same row order)."""
    d = cv_dir(out_dir)
    tp = test_pairs.reset_index(drop=True).copy()
    tp["prob_entail"] = np.asarray(prob_entail, dtype=float)
    tp["pred_label"] = np.where(tp["prob_entail"] > 0.5, 0, 1)

    overall = compute_all_metrics(tp["nli_label"].astype(int).values, tp["prob_entail"].values)
    overall.update({"fold": fold, **(extra or {})})
    (d / f"fold_{fold}_metrics.json").write_text(json.dumps(overall, indent=2, default=float), encoding="utf-8")

    timed = add_time_fields(tp)
    for grp, name in [("hypothesis_label", "label"), ("outlet", "outlet"), ("country", "country"),
                      ("year", "year"), ("decade", "decade")]:
        per_group_metrics(timed, timed["prob_entail"].values, grp).to_csv(d / f"fold_{fold}_per_{name}.csv", index=False)
    # written last: its presence marks the fold as done
    tp[[c for c in OUT_COLS if c in tp.columns]].to_csv(d / f"fold_{fold}_human_vs_model.csv", index=False)
    print(f"Fold {fold}: f1_binary={overall['f1_binary']:.4f}  f1_macro={overall['f1_macro']:.4f}  -> {d}")


def finalize(out_dir: Path, pairs: pd.DataFrame, fold_map: pd.DataFrame):
    """Cross-fold files, once all folds are done (as the notebook's output cells)."""
    d = cv_dir(out_dir)
    missing = [k for k in range(K_FOLDS) if not fold_done(out_dir, k)]
    if missing:
        raise FileNotFoundError(f"folds not finished yet: {missing}")
    (Path(out_dir) / "labels_found.json").write_text(json.dumps(labels_sorted(pairs), indent=2), encoding="utf-8")
    export_fold_shares(positives_long(pairs), fold_map, d / "stratification_shares.csv")

    epoch_files = [d / f"fold_{k}_epoch_losses.csv" for k in range(K_FOLDS)]
    if all(p.exists() for p in epoch_files):
        ep = pd.concat([pd.read_csv(p) for p in epoch_files], ignore_index=True)
        ep.rename(columns={"epoch": "Epoch", "training_loss": "Training Loss", "validation_loss": "Validation Loss"}) \
          .to_csv(d / "epochs_all_folds.csv", index=False)

    fold_metrics = [json.loads((d / f"fold_{k}_metrics.json").read_text(encoding="utf-8")) for k in range(K_FOLDS)]
    summary = {k: float(np.nanmean([m.get(k, np.nan) for m in fold_metrics])) for k in METRIC_KEYS}
    (d / "summary_overall.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    oof = pd.concat([pd.read_csv(d / f"fold_{k}_human_vs_model.csv") for k in range(K_FOLDS)], ignore_index=True)
    per_label = per_group_metrics(oof, oof["prob_entail"].values, "hypothesis_label").sort_values("f1_binary", ascending=False)
    per_label.to_csv(d / "per_label_pooled_all_folds.csv", index=False)
    pooled = compute_all_metrics(oof["nli_label"].astype(int).values, oof["prob_entail"].values)
    print(f"Pooled out-of-fold ({len(oof):,} pairs): f1_binary={pooled['f1_binary']:.4f}  "
          f"precision={pooled['precision_binary']:.4f}  recall={pooled['recall_binary']:.4f}  "
          f"f1_macro={pooled['f1_macro']:.4f}")
    print(f"Mean over folds: {json.dumps({k: round(v, 4) for k, v in summary.items()})}")
