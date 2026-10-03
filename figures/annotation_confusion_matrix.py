"""
annotation_confusion_matrix.py

Description:
Confusion matrix between the Step 2 ground truth (produced collegially by two expert
coders) and the independent coders, over all specific group labels. Rows are ground-truth
labels, columns are coder labels, grouped by broad class (thicker separator lines between
broad classes), plus a "(none)" row/column.

Each unit is a (sentence, independent coder) pair: sentences annotated by two independent
coders contribute two units. Labels are normalised with the same codebook as the ICR
(tables/icr_step2_krippendorff.ipynb); "target abroad" is not a group label and is ignored.
Labels are multi-label, so cells are filled per unit as follows:
- labels in both annotations      -> diagonal cell (label, label)
- labels only in one annotation   -> cross product of the ground-truth-only labels with the
                                     coder-only labels (a confusion between labels)
- if one side has no unmatched label left, the other side's unmatched labels are paired
  with "(none)" (missed label: (label, none); extra label: (none, label))
- both annotations empty          -> (none, none)

Per-label TP / FP / FN, precision, recall and F1 of the coders against the ground truth
are computed directly from the label sets (not from the matrix, whose row sums count
cross-product pairs).

Data:
- data/manual_annotations/step_2/annotation_step2.csv (via src/step_2/annotation_labels.py)

Outputs:
- figures/step_2/annotation_confusion_matrix/annotation_confusion_matrix.pdf
- figures/step_2/annotation_confusion_matrix/annotation_confusion_matrix.csv (raw counts)
- figures/step_2/annotation_confusion_matrix/annotation_per_label_agreement.csv

Usage (from anywhere):
python figures/annotation_confusion_matrix.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FIGURES_DIR = Path(__file__).resolve().parent
REPO_ROOT = FIGURES_DIR.parent
sys.path.insert(0, str(REPO_ROOT / "src" / "step_2"))
from annotation_labels import CODEBOOK, load_sentences  # noqa: E402

OUT_DIR = FIGURES_DIR / "step_2" / "annotation_confusion_matrix"
NONE_LABEL = "(none)"  # not "None": pandas reads that back as NaN


def label_order():
    """Specific labels in codebook order, plus None; and the broad-class boundaries."""
    labels, boundaries = [], []
    for lbls in CODEBOOK.values():
        labels += lbls
        boundaries.append(len(labels) - 1)
    return labels + [NONE_LABEL], boundaries  # last boundary separates the None row/column


def compute_matrix(units, labels):
    index = {l: i for i, l in enumerate(labels)}
    none = index[NONE_LABEL]
    mat = np.zeros((len(labels), len(labels)), dtype=np.int64)
    for gt, coder in units:
        for l in gt & coder:
            mat[index[l], index[l]] += 1
        gt_only = [index[l] for l in gt - coder] or [none]
        coder_only = [index[l] for l in coder - gt] or [none]
        if gt_only == [none] and coder_only == [none] and (gt or coder):
            continue  # identical non-empty annotations: only diagonal cells
        for t in gt_only:
            for p in coder_only:
                mat[t, p] += 1
    return pd.DataFrame(mat, index=labels, columns=labels)


def per_label_agreement(units, labels):
    def safe(num, den):
        return num / den if den else float("nan")

    broad_of = {l: b for b, lbls in CODEBOOK.items() for l in lbls}
    rows = []
    for l in labels:
        if l == NONE_LABEL:
            continue
        tp = sum(1 for g, c in units if l in g and l in c)
        fn = sum(1 for g, c in units if l in g and l not in c)
        fp = sum(1 for g, c in units if l not in g and l in c)
        p, r = safe(tp, tp + fp), safe(tp, tp + fn)
        rows.append({
            "broad_class": broad_of[l], "label": l, "support_ground_truth": tp + fn,
            "TP": tp, "FP": fp, "FN": fn,
            "precision": p, "recall": r, "f1": safe(2 * p * r, p + r),
        })
    return pd.DataFrame(rows)


def plot_matrix(df, boundaries, path):
    mat = df.values.astype(float)
    n = mat.shape[0]
    color = np.log1p(mat)

    fig_side = max(8.0, 0.16 * n)
    fig, ax = plt.subplots(figsize=(fig_side, fig_side))
    im = ax.imshow(color, cmap="Blues", aspect="equal")

    tick_labels = ["LGBTQIA+" if l == "lgbtqia+" else l[:1].upper() + l[1:] for l in df.index]
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(tick_labels, rotation=90, fontsize=4.5)
    ax.set_yticklabels(tick_labels, fontsize=4.5)
    ax.tick_params(length=0)

    for i in range(n):
        for j in range(n):
            if mat[i, j] > 0:
                ax.text(j, i, f"{int(mat[i, j])}", ha="center", va="center",
                        fontsize=4 if mat[i, j] < 1000 else 2.8,  # 4-digit counts must fit the cell
                        color="white" if color[i, j] > color.max() * 0.55 else "black")

    for k in range(n + 1):
        ax.axhline(k - 0.5, color="white", linewidth=0.3)
        ax.axvline(k - 0.5, color="white", linewidth=0.3)
    for b in boundaries:
        ax.axhline(b + 0.5, color="black", linewidth=1.0)
        ax.axvline(b + 0.5, color="black", linewidth=1.0)
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)

    ax.set_xlabel("Independent coder")
    ax.set_ylabel("Ground truth")

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("log(count + 1)")

    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {path}")


def main():
    sentences = load_sentences()
    units = [(gt, cl) for gt, cls in zip(sentences["gt_labels"], sentences["coder_labels"]) for cl in cls]
    print(f"  {len(sentences):,} sentences, {len(units):,} (sentence, coder) units")

    labels, boundaries = label_order()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = compute_matrix(units, labels)
    df.to_csv(OUT_DIR / "annotation_confusion_matrix.csv")
    print(f"  [saved] {OUT_DIR / 'annotation_confusion_matrix.csv'}")

    agreement = per_label_agreement(units, labels)
    agreement.to_csv(OUT_DIR / "annotation_per_label_agreement.csv", index=False)
    print(f"  [saved] {OUT_DIR / 'annotation_per_label_agreement.csv'}")

    plot_matrix(df, boundaries, OUT_DIR / "annotation_confusion_matrix.pdf")


if __name__ == "__main__":
    main()
