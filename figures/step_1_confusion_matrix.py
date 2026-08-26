"""
step_1_confusion_matrix.py

Description:
2x2 confusion matrix (counts + row-normalised) and the derived rates
(precision, recall/TPR, specificity/TNR, FPR, FNR, F1, balanced accuracy,
MCC) for the Step 1 binary social-group-mention detector
(selsar/social_group_detection). labels: 0 = no group mention,
1 = group mention (positive).

Data:
- Reads the first *.csv found under
  data/annotated_validation_corpus/step_1/ (cols: true, pred by default;
  override with --true_col/--pred_col if the annotated file uses different
  names). That data isn't available yet, so this script currently just
  prints a [skip] message -- rerun once the file is in place.
- If data/step_1/confusion_matrix.csv already exists it is used directly and
  the raw predictions are not re-read; delete it to force a recompute.

Outputs:
- data/step_1/confusion_matrix.csv                          (cached TP/FN/FP/TN grid)
- figures/step_1/confusion_matrix/step_1_confusion_matrix.pdf (+ .png)
- figures/step_1/confusion_matrix/step_1_confusion_matrix_summary.txt

Usage (from anywhere):
python figures/step_1_confusion_matrix.py
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, matthews_corrcoef

FIGURES_DIR = Path(__file__).resolve().parent
REPO_ROOT = FIGURES_DIR.parent

DEFAULT_INPUT_DIR = REPO_ROOT / "data" / "annotated_validation_corpus" / "step_1"
DEFAULT_CACHE_CSV = REPO_ROOT / "data" / "step_1" / "confusion_matrix.csv"
DEFAULT_FIGURES_OUT = FIGURES_DIR / "step_1" / "confusion_matrix"

POS_NAME, NEG_NAME = "Group mention", "No group mention"


# ── Core computation ─────────────────────────────────────────────────────
def binary_confusion(y_true, y_pred, pos_label=1):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])  # rows=true, cols=pred
    n00, n01 = cm[0, 0], cm[0, 1]
    n10, n11 = cm[1, 0], cm[1, 1]

    if pos_label == 1:
        TP, FN, FP, TN = n11, n10, n01, n00
    else:
        TP, FN, FP, TN = n00, n01, n10, n11

    total = TP + FN + FP + TN

    def safe(num, den):
        return float(num) / float(den) if den else float("nan")

    precision = safe(TP, TP + FP)
    recall = safe(TP, TP + FN)
    specificity = safe(TN, TN + FP)
    fpr = safe(FP, FP + TN)
    fnr = safe(FN, FN + TP)
    f1 = safe(2 * precision * recall, precision + recall) \
        if not (np.isnan(precision) or np.isnan(recall)) else float("nan")
    accuracy = safe(TP + TN, total)
    bal_acc = (recall + specificity) / 2 \
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
    return np.array([[res["TP"], res["FN"]],
                     [res["FP"], res["TN"]]], dtype=float)


def rates_from_cells(TP, FN, FP, TN):
    total = TP + FN + FP + TN

    def safe(num, den):
        return float(num) / float(den) if den else float("nan")

    precision = safe(TP, TP + FP)
    recall = safe(TP, TP + FN)
    specificity = safe(TN, TN + FP)
    f1 = safe(2 * precision * recall, precision + recall) \
        if not (np.isnan(precision) or np.isnan(recall)) else float("nan")
    return {
        "TP": int(TP), "FP": int(FP), "FN": int(FN), "TN": int(TN),
        "n": int(total), "n_pos": int(TP + FN), "n_neg": int(TN + FP),
        "prevalence": safe(TP + FN, total),
        "accuracy": safe(TP + TN, total),
        "balanced_accuracy": (recall + specificity) / 2
        if not (np.isnan(recall) or np.isnan(specificity)) else float("nan"),
        "precision": precision, "recall": recall, "f1": f1,
        "specificity": specificity, "fpr": safe(FP, FP + TN),
        "fnr": safe(FN, FN + TP), "mcc": float("nan"),
    }


# ── IO ────────────────────────────────────────────────────────────────────
def find_input_csv(input_dir):
    files = sorted(Path(input_dir).glob("*.csv"))
    return files[0] if files else None


def load_or_build_result(input_dir, cache_csv, true_col, pred_col):
    if cache_csv.exists():
        print(f"  [cache] loading confusion cells from {cache_csv}")
        df = pd.read_csv(cache_csv, index_col=0)
        TP, FN = df.iloc[0, 0], df.iloc[0, 1]
        FP, TN = df.iloc[1, 0], df.iloc[1, 1]
        return rates_from_cells(TP, FN, FP, TN)

    csv_path = find_input_csv(input_dir)
    if csv_path is None:
        return None

    print(f"  reading: {csv_path}")
    df = pd.read_csv(csv_path, engine="python")
    for col in (true_col, pred_col):
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not in {csv_path}. "
                             f"Available: {list(df.columns)}")

    res = binary_confusion(df[true_col], df[pred_col], pos_label=1)

    cache_csv.parent.mkdir(parents=True, exist_ok=True)
    cm = display_matrix(res).astype(int)
    pd.DataFrame(
        cm,
        index=[f"true: {POS_NAME}", f"true: {NEG_NAME}"],
        columns=[f"pred: {POS_NAME}", f"pred: {NEG_NAME}"],
    ).to_csv(cache_csv)
    print(f"  [saved] {cache_csv}")
    return res


def write_summary_txt(res, path):
    lines = [
        "Step 1 — social-group-mention detector",
        "=" * 45,
        f"n = {res['n']}  (pos = {res['n_pos']}, neg = {res['n_neg']})",
        f"TP={res['TP']}  FP={res['FP']}  FN={res['FN']}  TN={res['TN']}",
        f"accuracy          = {res['accuracy']:.4f}",
        f"balanced_accuracy = {res['balanced_accuracy']:.4f}",
        f"precision         = {res['precision']:.4f}",
        f"recall            = {res['recall']:.4f}",
        f"f1                = {res['f1']:.4f}",
        f"specificity (TNR) = {res['specificity']:.4f}",
        f"fpr               = {res['fpr']:.4f}",
        f"fnr               = {res['fnr']:.4f}",
        f"mcc               = {res['mcc']:.4f}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  [saved] {path}")


# ── Plotting ──────────────────────────────────────────────────────────────
def _annot(cm):
    row_sums = cm.sum(axis=1, keepdims=True)
    pct = np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums != 0) * 100
    return np.array([[f"{int(cm[i, j])}\n{pct[i, j]:.1f}%"
                      for j in range(cm.shape[1])] for i in range(cm.shape[0])])


def plot_confusion(res, title, path):
    cm = display_matrix(res)
    ticks = [POS_NAME, NEG_NAME]
    row_sums = cm.sum(axis=1, keepdims=True)
    norm = np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums != 0)

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

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(str(path).replace(".pdf", ".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {path}")


# ── Main ──────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(
        description="Confusion matrix for the Step 1 social-group-mention detector.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--input_dir", type=Path, default=DEFAULT_INPUT_DIR,
                   help="Dir containing the annotated predictions CSV")
    p.add_argument("--cache_csv", type=Path, default=DEFAULT_CACHE_CSV,
                   help="Cached TP/FN/FP/TN grid; reused if present")
    p.add_argument("--figures_out", type=Path, default=DEFAULT_FIGURES_OUT,
                   help="Output dir for the figure and the rates summary")
    p.add_argument("--true_col", type=str, default="true")
    p.add_argument("--pred_col", type=str, default="pred")
    args = p.parse_args()

    print("\n" + "=" * 60)
    print("STEP 1 — social-group-mention detector")
    print("=" * 60)

    res = load_or_build_result(args.input_dir, args.cache_csv,
                               args.true_col, args.pred_col)
    if res is None:
        print(f"\n[skip] No predictions CSV found under {args.input_dir} yet.")
        return

    args.figures_out.mkdir(parents=True, exist_ok=True)
    write_summary_txt(res, args.figures_out / "step_1_confusion_matrix_summary.txt")
    plot_confusion(res, "Step 1 — social-group mention",
                  args.figures_out / "step_1_confusion_matrix.pdf")

    print(f"\nDone. figures: {args.figures_out}  cache: {args.cache_csv}")


if __name__ == "__main__":
    main()
