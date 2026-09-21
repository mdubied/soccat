"""
step_2_confusion_matrix.py

Description:
Build a combined confusion matrix for the Step 2 per-category NLI detectors
(selsar/cv_*). Each broad-category CSV under
data/annotated_validation_corpus/step_2/<fold_set>/ was produced by running
one broad-category model against every sentence in its own validation slice,
once per specific label ("hypothesis") that belongs to that broad category.
For a given sentence a label counts as TRUE if the human annotation entails
it (nli_label == 0) and PREDICTED if the model entails it (pred_label == 0);
a sentence can have zero, one, or several true/predicted labels within a
broad category (multi-label).

All specific labels from all broad-category CSVs are combined into a single
matrix, ordered/grouped by broad category, with a thicker separator line
drawn at each broad-category boundary. Labels are paired per sentence with a
true-label x predicted-label cross product (so a sentence with 2 true and 1
predicted label contributes 2 cells); sentences with no true/predicted label
in a file are attributed to a "None" pseudo-category local to that broad
class's own block. Because each broad-category model is only ever tested
against its own hypotheses, the off-diagonal blocks between different broad
categories are structurally empty -- only the diagonal blocks (each
including its own None row/column) carry counts.

In addition to the combined matrix, one smaller matrix per broad category is
also produced (just that category's own specific labels + its None row/col).

Data:
- Reads *_human_vs_model.csv files (cols: sentence_id, nli_label, pred_label,
  hypothesis_label, ...) from data/annotated_validation_corpus/step_2/<fold_set>/.
  Default fold_set is "best_fold" (the only one populated so far).
- If data/step_2/confusion_matrix.csv already exists it is used directly and
  the raw CSVs are not re-read; delete it to force a recompute.

Outputs:
- data/step_2/confusion_matrix.csv        (cached raw matrix cells)
- figures/step_2/confusion_matrix/step_2_confusion_matrix.pdf (+ .png)
- figures/step_2/confusion_matrix/step_2_confusion_matrix_summary.txt
  (per-category precision/recall/F1/support + micro/macro averages)
- figures/step_2/confusion_matrix/by_broad_category/<broad_key>_confusion_matrix.pdf (+ .png)
- figures/step_2/confusion_matrix/by_broad_category/<broad_key>_confusion_matrix_summary.txt

Usage (from anywhere):
python figures/step_2_confusion_matrix.py
"""
import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FIGURES_DIR = Path(__file__).resolve().parent
REPO_ROOT = FIGURES_DIR.parent

DEFAULT_INPUT_DIR = REPO_ROOT / "data" / "annotated_validation_corpus" / "step_2" / "best_fold"
DEFAULT_CACHE_CSV = REPO_ROOT / "data" / "step_2" / "confusion_matrix.csv"
DEFAULT_FIGURES_OUT = FIGURES_DIR / "step_2" / "confusion_matrix"

NONE_LABEL = "None"  # per-broad-class pseudo-category: no true/predicted label in that file
SEP = "::"  # joins broad_key and specific label for the cache CSV header/index

# Thematic ordering used elsewhere in the repo (figures/step_2_heatmap.py).
# Any broad category found on disk but not listed here is appended
# alphabetically at the end, so new categories never get silently dropped.
BROAD_CATEGORY_ORDER = [
    "socio_economic_position",
    "labor_market_position_entrepreneur",
    "age_and_family_status",
    "identities_and_majority_minority_status",
    "profession",
    "social_roles_and_behavior_wo_volunteers",
    "social_deviance",
    "real_estate_ownership",
    "business_activity",
]

FILE_SUFFIX_RE = re.compile(r"_(?:best_)?fold\d+_human_vs_model$")


# ── Discovery ─────────────────────────────────────────────────────────────
def broad_key_from_filename(path):
    return FILE_SUFFIX_RE.sub("", path.stem)


def broad_display_name(broad_key):
    return broad_key.replace("_", " ").title()


def discover_files(input_dir):
    files = sorted(Path(input_dir).glob("*.csv"))
    if not files:
        return []
    by_broad = {broad_key_from_filename(f): f for f in files}

    ordered_keys = [k for k in BROAD_CATEGORY_ORDER if k in by_broad]
    remaining = sorted(k for k in by_broad if k not in ordered_keys)
    ordered_keys += remaining
    return [(k, by_broad[k]) for k in ordered_keys]


# ── Matrix construction ──────────────────────────────────────────────────
def build_label_index(broad_files):
    """Return (labels, block_boundaries).

    labels: ordered list of (broad_key, specific_label) tuples, grouped by
    broad category; each broad category's own block ends with a
    (broad_key, NONE_LABEL) entry local to that block.
    block_boundaries: positions (0-indexed, matrix-local) after which a
    thick separator line should be drawn (i.e. after each block's None row).
    """
    labels = []
    boundaries = []
    for broad_key, path in broad_files:
        df = pd.read_csv(path, usecols=["hypothesis_label"])
        seen = []
        for v in df["hypothesis_label"]:
            if v not in seen:
                seen.append(v)
        labels.extend((broad_key, v) for v in seen)
        labels.append((broad_key, NONE_LABEL))
        boundaries.append(len(labels) - 1)
    return labels, boundaries


def compute_matrix(broad_files, labels):
    index = {key: i for i, key in enumerate(labels)}
    n = len(labels)
    mat = np.zeros((n, n), dtype=np.int64)

    for broad_key, path in broad_files:
        none_idx = index[(broad_key, NONE_LABEL)]
        df = pd.read_csv(path, usecols=["sentence_id", "hypothesis_label",
                                        "nli_label", "pred_label"])
        for _, g in df.groupby("sentence_id"):
            true_rows = [index[(broad_key, l)]
                        for l in g.loc[g["nli_label"] == 0, "hypothesis_label"]]
            pred_rows = [index[(broad_key, l)]
                        for l in g.loc[g["pred_label"] == 0, "hypothesis_label"]]
            true_rows = true_rows or [none_idx]
            pred_rows = pred_rows or [none_idx]
            for t in true_rows:
                for p in pred_rows:
                    mat[t, p] += 1

    col_names = [f"{b}{SEP}{l}" for b, l in labels]
    return pd.DataFrame(mat, index=col_names, columns=col_names)


def load_or_build_matrix(input_dir, cache_csv):
    if cache_csv.exists():
        print(f"  [cache] loading matrix from {cache_csv}")
        df = pd.read_csv(cache_csv, index_col=0)
        labels = [tuple(idx.split(SEP, 1)) for idx in df.index]
        boundaries = [i for i in range(len(labels) - 1)
                     if labels[i][0] != labels[i + 1][0]]
        return df, labels, boundaries

    broad_files = discover_files(input_dir)
    if not broad_files:
        return None, None, None

    print(f"  reading {len(broad_files)} broad-category files from {input_dir}")
    for broad_key, path in broad_files:
        print(f"    {broad_key:45s} <- {path.name}")

    labels, boundaries = build_label_index(broad_files)
    df = compute_matrix(broad_files, labels)

    cache_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache_csv)
    print(f"  [saved] {cache_csv}")
    return df, labels, boundaries


# ── Metrics ───────────────────────────────────────────────────────────────
def per_category_rates(df, labels):
    mat = df.values.astype(float)
    rows = []
    real = [i for i, (_, l) in enumerate(labels) if l != NONE_LABEL]

    def safe(num, den):
        return num / den if den else float("nan")

    tp_sum = fp_sum = fn_sum = 0.0
    for i in real:
        broad_key, label = labels[i]
        tp = mat[i, i]
        support = mat[i, :].sum()
        predicted = mat[:, i].sum()
        fn = support - tp
        fp = predicted - tp
        precision = safe(tp, tp + fp)
        recall = safe(tp, tp + fn)
        f1 = safe(2 * precision * recall, precision + recall) \
            if not (np.isnan(precision) or np.isnan(recall)) else float("nan")
        rows.append({
            "broad_category": broad_display_name(broad_key),
            "category": label, "support": int(support), "predicted": int(predicted),
            "TP": int(tp), "FP": int(fp), "FN": int(fn),
            "precision": precision, "recall": recall, "f1": f1,
        })
        tp_sum += tp
        fp_sum += fp
        fn_sum += fn

    micro_p = safe(tp_sum, tp_sum + fp_sum)
    micro_r = safe(tp_sum, tp_sum + fn_sum)
    micro_f1 = safe(2 * micro_p * micro_r, micro_p + micro_r) \
        if not (np.isnan(micro_p) or np.isnan(micro_r)) else float("nan")
    macro_p = np.nanmean([r["precision"] for r in rows])
    macro_r = np.nanmean([r["recall"] for r in rows])
    macro_f1 = np.nanmean([r["f1"] for r in rows])

    overall = {
        "n_categories": len(rows),
        "n_sentence_label_pairs_true": int(tp_sum + fn_sum),
        "n_sentence_label_pairs_pred": int(tp_sum + fp_sum),
        "micro_precision": micro_p, "micro_recall": micro_r, "micro_f1": micro_f1,
        "macro_precision": macro_p, "macro_recall": macro_r, "macro_f1": macro_f1,
    }
    return pd.DataFrame(rows), overall


# ── Per-broad-category extraction ───────────────────────────────────────
def broad_keys_in_order(labels):
    seen = []
    for b, _ in labels:
        if b not in seen:
            seen.append(b)
    return seen


def extract_block(df, labels, broad_key):
    idx = [i for i, (b, _) in enumerate(labels) if b == broad_key]
    names = [df.index[i] for i in idx]
    sub_labels = [labels[i] for i in idx]
    sub_df = df.loc[names, names]
    boundaries = [len(sub_labels) - 2] if len(sub_labels) > 1 else []
    return sub_df, sub_labels, boundaries


def write_summary_txt(rows_df, overall, path):
    lines = []
    lines.append("Step 2 — combined confusion matrix, per-category rates")
    lines.append("=" * 60)
    lines.append(f"{'category':45s} {'support':>7s} {'TP':>5s} {'FP':>5s} "
                 f"{'FN':>5s} {'prec':>6s} {'rec':>6s} {'f1':>6s}")
    for _, r in rows_df.iterrows():
        name = f"[{r['broad_category']}] {r['category']}"
        lines.append(f"{name[:45]:45s} {r['support']:7d} {r['TP']:5d} "
                     f"{r['FP']:5d} {r['FN']:5d} {r['precision']:6.3f} "
                     f"{r['recall']:6.3f} {r['f1']:6.3f}")
    lines.append("-" * 60)
    lines.append(f"n categories: {overall['n_categories']}")
    lines.append(f"true label instances (sentence x category): "
                 f"{overall['n_sentence_label_pairs_true']}")
    lines.append(f"predicted label instances (sentence x category): "
                 f"{overall['n_sentence_label_pairs_pred']}")
    lines.append(f"micro  precision={overall['micro_precision']:.4f}  "
                 f"recall={overall['micro_recall']:.4f}  "
                 f"f1={overall['micro_f1']:.4f}")
    lines.append(f"macro  precision={overall['macro_precision']:.4f}  "
                 f"recall={overall['macro_recall']:.4f}  "
                 f"f1={overall['macro_f1']:.4f}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  [saved] {path}")


# ── Plotting ──────────────────────────────────────────────────────────────
def plot_matrix(df, labels, boundaries, title, path):
    mat = df.values.astype(float)
    n = mat.shape[0]
    color = np.log1p(mat)

    fig_side = max(8.0, 0.16 * n)
    fig, ax = plt.subplots(figsize=(fig_side, fig_side))
    im = ax.imshow(color, cmap="Blues", aspect="equal")

    tick_labels = [l for _, l in labels]
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(tick_labels, rotation=90, fontsize=4.5)
    ax.set_yticklabels(tick_labels, fontsize=4.5)
    ax.tick_params(length=0)

    fontsize = 4.5 if n > 40 else 6
    for i in range(n):
        for j in range(n):
            if mat[i, j] > 0:
                ax.text(j, i, f"{int(mat[i, j])}", ha="center", va="center",
                        fontsize=fontsize,
                        color="white" if color[i, j] > color.max() * 0.55 else "black")

    for k in range(n + 1):
        ax.axhline(k - 0.5, color="white", linewidth=0.3)
        ax.axvline(k - 0.5, color="white", linewidth=0.3)
    for b in boundaries:
        ax.axhline(b + 0.5, color="black", linewidth=1.4)
        ax.axvline(b + 0.5, color="black", linewidth=1.4)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.4)

    ax.set_xlabel("Predicted category")
    ax.set_ylabel("True category")
    ax.set_title(title, fontsize=11)

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("log(count + 1)")

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(str(path).replace(".pdf", ".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {path}")


# ── Main ──────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(
        description="Combined confusion matrix for the Step 2 NLI detectors.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--input_dir", type=Path, default=DEFAULT_INPUT_DIR,
                   help="Dir with *_human_vs_model.csv files (one per broad category)")
    p.add_argument("--cache_csv", type=Path, default=DEFAULT_CACHE_CSV,
                   help="Cached matrix cells; reused if present, else (re)built here")
    p.add_argument("--figures_out", type=Path, default=DEFAULT_FIGURES_OUT,
                   help="Output dir for the figure and the rates summary")
    args = p.parse_args()

    print("\n" + "=" * 60)
    print("STEP 2 — combined confusion matrix (per-category NLI detectors)")
    print("=" * 60)

    df, labels, boundaries = load_or_build_matrix(args.input_dir, args.cache_csv)
    if df is None:
        print(f"\n[skip] No Step 2 human_vs_model CSVs found under {args.input_dir}")
        return

    rows_df, overall = per_category_rates(df, labels)
    args.figures_out.mkdir(parents=True, exist_ok=True)
    write_summary_txt(rows_df, overall,
                      args.figures_out / "step_2_confusion_matrix_summary.txt")
    plot_matrix(df, labels, boundaries,
               "Step 2 — combined confusion matrix (grouped by broad category)",
               args.figures_out / "step_2_confusion_matrix.pdf")

    by_broad_dir = args.figures_out / "by_broad_category"
    by_broad_dir.mkdir(parents=True, exist_ok=True)
    for broad_key in broad_keys_in_order(labels):
        sub_df, sub_labels, sub_boundaries = extract_block(df, labels, broad_key)
        sub_rows_df, sub_overall = per_category_rates(sub_df, sub_labels)
        write_summary_txt(
            sub_rows_df, sub_overall,
            by_broad_dir / f"{broad_key}_confusion_matrix_summary.txt")
        plot_matrix(
            sub_df, sub_labels, sub_boundaries,
            f"Step 2 — {broad_display_name(broad_key)} confusion matrix",
            by_broad_dir / f"{broad_key}_confusion_matrix.pdf")

    print(f"\nDone. figures: {args.figures_out}  cache: {args.cache_csv}")


if __name__ == "__main__":
    main()
