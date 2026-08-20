#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SOCCAT — Confusion Matrices for the Step 1 and Step 2 models
============================================================
Computes 2x2 confusion matrices (counts + normalised) and the derived
rates (precision, recall/TPR, specificity/TNR, FPR, FNR, F1, balanced
accuracy, MCC) for both fine-tuned models. 

  Step 1 : binary social-group-mention detector (selsar/social_group_detection)
           reads  step_1/output/test_predictions.csv  (cols: true, pred)
           labels 0 = no group mention, 1 = group mention (positive).

  Step 2 : the 8 per-category NLI detectors (selsar/cv_*)
           reads  step_2/output/replication/{category}/predictions.csv
           (cols: nli_label, pred_label), manifest step_2/categories.json.
           labels 0 = entailment / group present (positive), 1 = not_entailment.

"""

import argparse
import glob
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless (scicore login/compute nodes)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, matthews_corrcoef


# ── Repo-relative default paths (resolved from THIS file's location) ───────────
SRC_DIR   = Path(__file__).resolve().parent          # soccat/src/
REPO_ROOT = SRC_DIR.parent                            # soccat/

DEFAULTS = {
    "step1_pred":  SRC_DIR / "step_1" / "output" / "test_predictions.csv",
    "step2_root":  SRC_DIR / "step_2" / "output" / "replication",
    "manifest":    SRC_DIR / "step_2" / "categories.json",
    "tables_out":  REPO_ROOT / "tables"  / "confusion",
    "figures_out": REPO_ROOT / "figures" / "confusion",
}


# ── Core computation ──────────────────────────────────────────────────────────
def binary_confusion(y_true, y_pred, pos_label):
    """Counts and rates for a binary problem, oriented around `pos_label`.
    Robust to a class being absent from y_true or y_pred (labels=[0, 1])."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])  # rows=true, cols=pred
    n00, n01 = cm[0, 0], cm[0, 1]
    n10, n11 = cm[1, 0], cm[1, 1]

    if pos_label == 1:
        TP, FN, FP, TN = n11, n10, n01, n00
    elif pos_label == 0:
        TP, FN, FP, TN = n00, n01, n10, n11
    else:
        raise ValueError("pos_label must be 0 or 1")

    total = TP + FN + FP + TN

    def safe(num, den):
        return float(num) / float(den) if den else float("nan")

    precision   = safe(TP, TP + FP)
    recall      = safe(TP, TP + FN)          # sensitivity / TPR
    specificity = safe(TN, TN + FP)          # TNR
    fpr         = safe(FP, FP + TN)
    fnr         = safe(FN, FN + TP)
    f1          = safe(2 * precision * recall, precision + recall) \
                  if not (np.isnan(precision) or np.isnan(recall)) else float("nan")
    accuracy    = safe(TP + TN, total)
    bal_acc     = (recall + specificity) / 2 \
                  if not (np.isnan(recall) or np.isnan(specificity)) else float("nan")
    try:
        mcc = matthews_corrcoef(y_true, y_pred)
    except Exception:
        mcc = float("nan")

    return {
        "TP": int(TP), "FP": int(FP), "FN": int(FN), "TN": int(TN),
        "n": int(total), "n_pos": int(TP + FN), "n_neg": int(TN + FP),
        "prevalence": safe(TP + FN, total),
        "accuracy": accuracy, "balanced_accuracy": bal_acc,
        "precision": precision, "recall": recall, "f1": f1,
        "specificity": specificity, "fpr": fpr, "fnr": fnr, "mcc": mcc,
    }


def display_matrix(res):
    """2x2 matrix in [pos, neg] order: [[TP, FN], [FP, TN]] (rows=true)."""
    return np.array([[res["TP"], res["FN"]],
                     [res["FP"], res["TN"]]], dtype=float)


# ── Plotting ──────────────────────────────────────────────────────────────────
def _annot(cm):
    """Cell text: count on top, row-normalised % below."""
    row_sums = cm.sum(axis=1, keepdims=True)
    pct = np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums != 0) * 100
    return np.array([[f"{int(cm[i, j])}\n{pct[i, j]:.1f}%"
                      for j in range(cm.shape[1])] for i in range(cm.shape[0])])


def plot_confusion(res, pos_name, neg_name, title, path, ax=None):
    cm = display_matrix(res)
    ticks = [pos_name, neg_name]
    row_sums = cm.sum(axis=1, keepdims=True)
    norm = np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums != 0)

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(4.6, 4.0))

    sns.heatmap(norm, annot=_annot(cm), fmt="", cmap="Blues",
                vmin=0, vmax=1, cbar=False, square=True,
                linewidths=0.5, linecolor="white",
                xticklabels=ticks, yticklabels=ticks, ax=ax,
                annot_kws={"fontsize": 10})
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title, fontsize=11)
    ax.tick_params(length=0)

    if own_fig:
        plt.tight_layout()
        fig.savefig(path, bbox_inches="tight")
        fig.savefig(str(path).replace(".pdf", ".png"), dpi=200, bbox_inches="tight")
        plt.close(fig)


def plot_grid(results, titles, pos_name, neg_name, path):
    n = len(results)
    ncols = 4 if n > 6 else (3 if n > 4 else n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.4 * ncols, 3.2 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for i, (res, title) in enumerate(zip(results, titles)):
        plot_confusion(res, pos_name, neg_name, title, None, ax=axes[i])
    for j in range(n, len(axes)):
        axes[j].axis("off")
    plt.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(str(path).replace(".pdf", ".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


# ── IO ────────────────────────────────────────────────────────────────────────
def read_table(path):
    """Read a CSV/XLSX. engine="python" honours quoting robustly, so premise
    text with embedded newlines / carriage returns doesn't mis-split rows."""
    ext = os.path.splitext(str(path))[1].lower()
    if ext in (".xlsx", ".xls"):
        return pd.read_excel(path, engine="openpyxl")
    return pd.read_csv(path, engine="python")


def save_cm_csv(res, pos_name, neg_name, path):
    cm = display_matrix(res).astype(int)
    # strip newlines used for figure tick labels so CSV headers stay clean
    pn, nn = pos_name.replace("\n", " "), neg_name.replace("\n", " ")
    df = pd.DataFrame(
        cm,
        index=[f"true: {pn}", f"true: {nn}"],
        columns=[f"pred: {pn}", f"pred: {nn}"],
    )
    df.to_csv(path)


# ── Step 1 ────────────────────────────────────────────────────────────────────
def run_step1(args, tables_dir, figures_dir):
    print("\n" + "=" * 60)
    print("STEP 1 — social-group-mention detector")
    print("=" * 60)
    print(f"  reading: {args.step1_pred}")
    df = read_table(args.step1_pred)
    for col in (args.step1_true_col, args.step1_pred_col):
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not in {args.step1_pred}. "
                             f"Available: {list(df.columns)}")

    pos_name, neg_name = args.step1_pos_name, args.step1_neg_name
    res = binary_confusion(df[args.step1_true_col], df[args.step1_pred_col],
                           pos_label=1)  # 1 = group mention

    save_cm_csv(res, pos_name, neg_name, tables_dir / "step1_confusion_matrix.csv")
    with open(tables_dir / "step1_rates.json", "w") as f:
        json.dump(res, f, indent=2)
    plot_confusion(res, pos_name, neg_name, "Step 1 — social-group mention",
                   figures_dir / "step1_confusion_matrix.pdf")

    print(f"  n = {res['n']:,}  (pos = {res['n_pos']:,}, neg = {res['n_neg']:,})")
    print(f"  TP={res['TP']}  FP={res['FP']}  FN={res['FN']}  TN={res['TN']}")
    print(f"  precision={res['precision']:.4f}  recall={res['recall']:.4f}  "
          f"F1={res['f1']:.4f}  acc={res['accuracy']:.4f}")


# ── Step 2 ────────────────────────────────────────────────────────────────────
def _step2_prediction_files(args):
    """Return [(category_key, display_name, dataframe), ...] for Step 2."""
    if args.step2_cv_results:
        files = sorted(glob.glob(os.path.join(str(args.step2_cv_results),
                                              "fold_*_human_vs_model.csv")))
        if not files:
            raise ValueError(f"No fold_*_human_vs_model.csv under {args.step2_cv_results}")
        pooled = pd.concat([read_table(f) for f in files], ignore_index=True)
        return [("cv_pooled_all_folds", "CV — all folds pooled", pooled)]

    with open(args.manifest) as f:
        manifest = json.load(f)
    if args.category:
        manifest = [c for c in manifest if c["name"] == args.category]
        if not manifest:
            raise ValueError(f"Category '{args.category}' not in {args.manifest}")

    items = []
    for cat in manifest:
        pred_path = os.path.join(str(args.step2_root), cat["name"], "predictions.csv")
        if not os.path.exists(pred_path):
            print(f"  WARNING: missing {pred_path} — skipping {cat['name']}")
            continue
        items.append((cat["name"], cat.get("display_name", cat["name"]),
                      read_table(pred_path)))
    if not items:
        raise ValueError("No Step 2 prediction files found under "
                         f"{args.step2_root}. Run replicate_from_hub_step_2.py first.")
    return items


def run_step2(args, tables_dir, figures_dir):
    print("\n" + "=" * 60)
    print("STEP 2 — per-category NLI detectors (selsar/cv_*)")
    print("=" * 60)

    pos_name, neg_name = args.step2_pos_name, args.step2_neg_name
    items = _step2_prediction_files(args)

    per_cat_results, titles = [], []
    pooled_true, pooled_pred, summary_rows = [], [], []

    for key, disp, df in items:
        for col in (args.step2_true_col, args.step2_pred_col):
            if col not in df.columns:
                raise ValueError(f"Column '{col}' not in Step 2 file for '{key}'. "
                                 f"Available: {list(df.columns)}")
        y_true = df[args.step2_true_col].astype(int).values
        y_pred = df[args.step2_pred_col].astype(int).values

        res = binary_confusion(y_true, y_pred, pos_label=0)  # 0 = entailment
        per_cat_results.append(res)
        titles.append(disp)
        pooled_true.append(y_true)
        pooled_pred.append(y_pred)

        save_cm_csv(res, pos_name, neg_name,
                    tables_dir / f"step2_{key}_confusion_matrix.csv")
        with open(tables_dir / f"step2_{key}_rates.json", "w") as f:
            json.dump(res, f, indent=2)
        summary_rows.append({"category": key, "display_name": disp, **res})

        print(f"  {disp:42s}  TP={res['TP']:5d} FP={res['FP']:5d} "
              f"FN={res['FN']:5d} TN={res['TN']:6d}  F1={res['f1']:.3f}")

    if len(items) > 1:
        yt, yp = np.concatenate(pooled_true), np.concatenate(pooled_pred)
        pooled = binary_confusion(yt, yp, pos_label=0)
        save_cm_csv(pooled, pos_name, neg_name,
                    tables_dir / "step2_pooled_confusion_matrix.csv")
        with open(tables_dir / "step2_pooled_rates.json", "w") as f:
            json.dump(pooled, f, indent=2)
        plot_confusion(pooled, pos_name, neg_name,
                       "Step 2 — pooled across categories",
                       figures_dir / "step2_pooled_confusion_matrix.pdf")
        summary_rows.append({"category": "POOLED", "display_name": "All categories",
                             **pooled})
        print(f"  {'POOLED (all categories)':42s}  TP={pooled['TP']:5d} "
              f"FP={pooled['FP']:5d} FN={pooled['FN']:5d} TN={pooled['TN']:6d}  "
              f"F1={pooled['f1']:.3f}")

    if per_cat_results:
        plot_grid(per_cat_results, titles, pos_name, neg_name,
                  figures_dir / "step2_confusion_grid.pdf")

    cols = ["category", "display_name", "TP", "FP", "FN", "TN", "n",
            "n_pos", "n_neg", "prevalence", "accuracy", "balanced_accuracy",
            "precision", "recall", "f1", "specificity", "fpr", "fnr", "mcc"]
    pd.DataFrame(summary_rows)[cols].round(4).to_csv(
        tables_dir / "step2_rates_summary.csv", index=False)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(
        description="Confusion matrices for the SOCCAT Step 1 & Step 2 models.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Step 1
    p.add_argument("--step1_pred", type=Path, default=None,
                   help="test_predictions.csv from Step 1 (default: repo-relative)")
    p.add_argument("--step1_true_col", type=str, default="true")
    p.add_argument("--step1_pred_col", type=str, default="pred")
    p.add_argument("--step1_pos_name", type=str, default="Group mention")
    p.add_argument("--step1_neg_name", type=str, default="No group mention")
    # Step 2
    p.add_argument("--step2_root", type=Path, default=None,
                   help="Dir with {category}/predictions.csv (default: repo-relative)")
    p.add_argument("--manifest", type=Path, default=None,
                   help="categories.json manifest (default: repo-relative)")
    p.add_argument("--category", type=str, default=None,
                   help="Restrict Step 2 to a single category name")
    p.add_argument("--step2_cv_results", type=Path, default=None,
                   help="Alternative: pool fold_*_human_vs_model.csv from a cv_results dir")
    p.add_argument("--step2_true_col", type=str, default="nli_label")
    p.add_argument("--step2_pred_col", type=str, default="pred_label")
    p.add_argument("--step2_pos_name", type=str, default="entailment\n(group present)")
    p.add_argument("--step2_neg_name", type=str, default="not_entailment")
    # Output (split to match repo: tables/ vs figures/)
    p.add_argument("--tables_out", type=Path, default=None,
                   help="Dir for count/rate CSVs (default: repo tables/confusion/)")
    p.add_argument("--figures_out", type=Path, default=None,
                   help="Dir for heatmap PDFs (default: repo figures/confusion/)")
    p.add_argument("--skip_step1", action="store_true", help="Skip Step 1")
    p.add_argument("--skip_step2", action="store_true", help="Skip Step 2")
    args = p.parse_args()

    # Fill repo-relative defaults; remember which were user-supplied (must exist).
    explicit = {k: getattr(args, k) is not None
                for k in ("step1_pred", "step2_root", "manifest",
                          "tables_out", "figures_out")}
    for k in explicit:
        if getattr(args, k) is None:
            setattr(args, k, DEFAULTS[k])

    tables_dir, figures_dir = args.tables_out, args.figures_out
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="white")

    ran = 0
    # Step 1 — run if the file exists; only hard-error if the user named it explicitly.
    if not args.skip_step1:
        if os.path.exists(args.step1_pred):
            run_step1(args, tables_dir, figures_dir)
            ran += 1
        elif explicit["step1_pred"]:
            raise FileNotFoundError(f"--step1_pred not found: {args.step1_pred}")
        else:
            print(f"\n[skip] Step 1 predictions not found at {args.step1_pred}")

    # Step 2 — run if the replication root (or CV dir) exists.
    if not args.skip_step2:
        step2_available = (args.step2_cv_results is not None
                           or os.path.isdir(args.step2_root))
        if step2_available:
            run_step2(args, tables_dir, figures_dir)
            ran += 1
        elif explicit["step2_root"]:
            raise FileNotFoundError(f"--step2_root not found: {args.step2_root}")
        else:
            print(f"\n[skip] Step 2 predictions not found at {args.step2_root}")

    if ran == 0:
        raise SystemExit(
            "\nNothing ran. Expected prediction files were not found.\n"
            "  Step 1: run SOCCAT_mDeBERTa_replication.py  -> step_1/output/test_predictions.csv\n"
            "  Step 2: run replicate_from_hub_step_2.py    -> step_2/output/replication/{cat}/predictions.csv")

    print(f"\nDone.")
    print(f"  tables : {tables_dir}")
    print(f"  figures: {figures_dir}")


if __name__ == "__main__":
    main()
