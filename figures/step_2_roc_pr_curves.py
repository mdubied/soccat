"""
step_2_roc_pr_curves.py

Description:
Threshold-free companion to step_2_boxplot.py. Draws ROC and Precision-Recall
curves per category (hypothesis_label) of a broad class, using the raw
per-sentence predictions in "fold_X_human_vs_model.csv" (columns: nli_label
as ground truth, prob_entail as the model's positive-class score).

Unlike accuracy/precision/recall/F1, ROC-AUC is prevalence-invariant, so it
is the right tool for judging whether the classifier's underlying ranking
ability changed when the test set's positive/negative balance changed (see
project discussion: the retrain used a much larger, much more negative-heavy
test set than before). PR curves are shown with their no-skill baseline
(= prevalence) since PR-AUC is prevalence-sensitive and not interpretable
without that reference.

One figure with a small-multiples grid (one panel per category): 5 thin
per-fold curves + 1 bold mean curve, mean AUC annotated per panel.

Outputs:
- PDF files in "step_2/roc_pr_curves/": "<broad_class>_roc.pdf" and
  "<broad_class>_pr.pdf".

Usage (from this directory):
python step_2_roc_pr_curves.py --broad_class age_family
(or no argument for all broad classes)
"""
import argparse
import glob
import math
import os
import re
import textwrap

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless: this script only writes PDFs, no display
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score

# ============================================================
# PARSE COMMAND-LINE ARGUMENTS
# ============================================================
parser = argparse.ArgumentParser()
parser.add_argument("--broad_class", type=str, default=None, help="Broad category name (e.g., 'age_family')")
args = parser.parse_args()

# ============================================================
# PARAMETERS
# ============================================================
STEP_2_DATA_DIR = "../data/model_performance/step_2"
OUTPUT_DIR = "step_2/roc_pr_curves"

# old_name -> new folder name (same mapping as step_2_boxplot.py)
FOLDER_NAME_MAP = {
    "age_family": "age_family_status",
    "identity": "identities_minority_majority_status",
    "labor_market_w_entrepreneurs": "labor_market_position",
    "profession": "profession",
    "real_estate": "real_estate_ownership",
    "social_deviance": "social_deviance",
    "social_roles": "social_roles_behavior",
    "socio_economic": "socio_economic_position",
}
BROAD_CLASS_LIST = list(FOLDER_NAME_MAP.keys())

RENAME_DICT = {
    "entrepreneurs in [specific] sector": "entrepreneurs in specific sector",
    "lgbtqqia+": "LGBTQIA+",
    "people with an immigration background, including immigrants": "people with immigration background",
    "offenders, criminals, prisoners and/or accused people": "offenders, criminals, prisoners, accused people",
    "terrorists, rebels, revolutionaries and/or movements of armed resistance": "terrorists, revolutionaries, rebels, armed resistance",
}

N_GRID = 200  # points on the common FPR / recall grid used for averaging curves across folds

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Latin Modern Sans"],
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "text.latex.preamble": r"""
        \usepackage[T1]{fontenc}
        \usepackage{lmodern}
        \renewcommand{\familydefault}{\sfdefault}
    """,
})

# ============================================================
# HELPERS
# ============================================================

def load_predictions(broad_class):
    """Concatenate all fold_X_human_vs_model.csv files for a broad class, tagged with fold."""
    folder = FOLDER_NAME_MAP.get(broad_class, broad_class)
    files = sorted(
        glob.glob(f"{STEP_2_DATA_DIR}/{folder}/fold_*_human_vs_model.csv"),
        key=lambda f: int(re.search(r"fold_(\d+)_human_vs_model", f).group(1))
    )
    assert files, f"No fold_*_human_vs_model.csv found for broad class '{broad_class}' in {STEP_2_DATA_DIR}/{folder}"
    dfs = []
    for f in files:
        fold_num = int(re.search(r"fold_(\d+)_human_vs_model", f).group(1))
        d = pd.read_csv(f, usecols=["nli_label", "prob_entail", "hypothesis_label"])
        d["fold"] = fold_num
        dfs.append(d)
    df = pd.concat(dfs, ignore_index=True)
    df["hypothesis_label"] = df["hypothesis_label"].replace(RENAME_DICT)
    # nli_label uses NLI convention 0 = entailment (positive), 1 = not-entailment
    # (verified against n_pos_entail in fold_*_per_label.csv) -- flip to the
    # usual 1 = positive convention expected by sklearn's *_curve functions.
    df["y_true"] = (df["nli_label"] == 0).astype(int)
    return df


def per_category_curves(df, curve_type):
    """
    For each category, compute per-fold curves plus a fold-averaged curve.
    curve_type: "roc" or "pr".
    Returns dict: label -> {"x": grid, "fold_ys": [array,...], "mean_y": array,
                             "mean_auc": float, "std_auc": float, "prevalence": float (pr only)}
    """
    out = {}
    grid = np.linspace(0, 1, N_GRID)
    for label, g in df.groupby("hypothesis_label"):
        fold_ys = []
        aucs = []
        for fold, gf in g.groupby("fold"):
            y_true = gf["y_true"].to_numpy()
            y_score = gf["prob_entail"].to_numpy()
            if y_true.sum() == 0 or y_true.sum() == len(y_true):
                continue  # undefined AUC for this fold/label (no positives or no negatives)

            if curve_type == "roc":
                fpr, tpr, _ = roc_curve(y_true, y_score)
                y_interp = np.interp(grid, fpr, tpr)
                y_interp[0] = 0.0
                aucs.append(roc_auc_score(y_true, y_score))
            else:
                precision, recall, _ = precision_recall_curve(y_true, y_score)
                # recall is decreasing; sort ascending for interpolation
                order = np.argsort(recall)
                y_interp = np.interp(grid, recall[order], precision[order])
                aucs.append(average_precision_score(y_true, y_score))

            fold_ys.append(y_interp)

        if not fold_ys:
            continue

        fold_ys = np.vstack(fold_ys)
        out[label] = {
            "x": grid,
            "fold_ys": fold_ys,
            "mean_y": fold_ys.mean(axis=0),
            "mean_auc": float(np.mean(aucs)),
            "std_auc": float(np.std(aucs)),
            "prevalence": float(g["y_true"].mean()),
        }
    return out


def wrap_title(label, width=26):
    # each wrapped physical line must be independently LaTeX-balanced, since
    # matplotlib's usetex rendering processes "\n"-separated lines separately
    lines = textwrap.wrap(str(label), width)
    return "\n".join(rf"\textsf{{{line}}}" for line in lines)


def plot_grid(curves, curve_type, output_path, broad_class_title):
    labels_sorted = sorted(curves.keys(), key=lambda l: curves[l]["mean_auc"], reverse=True)
    n = len(labels_sorted)
    ncols = math.ceil(math.sqrt(n))
    nrows = math.ceil(n / ncols)

    fig, axs = plt.subplots(
        nrows, ncols,
        figsize=(2.2 * ncols, 2.2 * nrows),
        squeeze=False,
    )

    for i, label in enumerate(labels_sorted):
        ax = axs[i // ncols][i % ncols]
        c = curves[label]

        for fold_y in c["fold_ys"]:
            ax.plot(c["x"], fold_y, color="gray", alpha=0.35, linewidth=0.8)
        ax.plot(c["x"], c["mean_y"], color="black", linewidth=1.5)

        if curve_type == "roc":
            ax.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=0.7)
            auc_label = "AUC"
        else:
            ax.axhline(c["prevalence"], color="gray", linestyle="--", linewidth=0.7)
            auc_label = "AP"

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.set_title(
            wrap_title(label) + "\n" +
            rf"\textsf{{\textit{{{auc_label} = {c['mean_auc']:.2f} $\pm$ {c['std_auc']:.2f}}}}}"
        )

        if i // ncols == nrows - 1:
            ax.set_xlabel("FPR" if curve_type == "roc" else "Recall")
        if i % ncols == 0:
            ax.set_ylabel("TPR" if curve_type == "roc" else "Precision")

    # hide unused axes
    for j in range(n, nrows * ncols):
        axs[j // ncols][j % ncols].axis("off")

    safe_title = broad_class_title.replace("_", r"\_")
    curve_name = "ROC" if curve_type == "roc" else "Precision-Recall"
    fig.suptitle(rf"\textsf{{{safe_title}: {curve_name} curves}}")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Saved file as: {output_path}")


# ============================================================
# MAIN
# ============================================================
os.makedirs(OUTPUT_DIR, exist_ok=True)

broad_classes = [args.broad_class] if args.broad_class else BROAD_CLASS_LIST

for bc in broad_classes:
    df = load_predictions(bc)

    roc_curves = per_category_curves(df, "roc")
    pr_curves = per_category_curves(df, "pr")

    plot_grid(roc_curves, "roc", os.path.join(OUTPUT_DIR, f"{bc}_roc.pdf"), bc)
    plot_grid(pr_curves, "pr", os.path.join(OUTPUT_DIR, f"{bc}_pr.pdf"), bc)
