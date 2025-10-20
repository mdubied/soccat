# File: figure_performance_summary.py
# One figure with 4 panels (accuracy, precision, recall, f1), paginated vertically.
# - Sorted globally by decreasing F1 mean (best first), then paginated.
# - Best entries appear at the TOP of each page (invert y-axis).
# - Long labels wrapped and measured to allocate left margin (no clipping).
# - No bottom x-axis labels; no overarching left ylabel.
# - Outputs: combined multipage PDF + separate per-page PDFs + PNGs.

import os, math, textwrap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# =======================
# PARAMETERS (edit here)
# =======================
INPUT_FILE   = "data/model_performance/all_cat_box_plot.csv"
# INPUT_FILE   = "data/model_performance/broad_cat_mean_ci.csv"
OUTPUT_DIR   = "figures/performance_summary/all_cat_box_plot"
# OUTPUT_DIR   = "figures/performance_summary/broad_cat_mean_ci"
OUTPUT_BASE_NAME = "perf_all_cat_box_plot"
METRICS      = ["accuracy", "precision_binary", "recall_binary", "f1_binary"]
# METRICS      = ["accuracy", "precision_micro", "recall_micro", "f1_micro"]
# METRICS      = ["accuracy", "precision_binary", "recall_binary", "f1_macro"] 
NAME_CAT = "hypothesis_label"
# NAME_CAT = "model"
BOX_PLOT    = True        # True -> box plots; False -> mean + 95% CI error bars
N_RUNS       = 5            # for 95% CI
FIG_WIDTH_CM = 26.0         # total figure width
MAX_H_CM     = 12.0         # max height per page "block"
ROW_H_IN     = 0.35         # height per row (inches)
WRAP_CHARS   = 40           # initial wrap width (characters)
DEBUG        = False        # True -> print diagnostics

# Fonts 12/10
plt.rcParams.update({
    "font.size": 10,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
})

def wrap_labels(labels, width_chars=40):
    return ["\n".join(textwrap.wrap(str(lbl), width_chars)) for lbl in labels]

def compute_ci95(df, n_runs):
    for m in METRICS:
        if f"{m}_std" not in df.columns or f"{m}_mean" not in df.columns:
            raise ValueError(f"Missing columns for '{m}': need {m}_mean and {m}_std")
        df[f"{m}_ci"] = 1.96 * df[f"{m}_std"] / np.sqrt(n_runs)
    return df

def paginate(df, fig_width_cm, max_h_cm, row_h_in, box_plot=False, label_col=None):
    """
    Returns: (pages, fig_w_in, max_h_in, rows_per_page)
    - If box_plot=False: pages is a list of DataFrames (row-sliced as before).
    - If box_plot=True:  pages is a list of tuples (page_df_subset, raw_labels_for_page),
      where each page contains all rows for a subset of UNIQUE labels, preserving order.
    """
    cm = 1/2.54
    fig_w_in = fig_width_cm * cm
    max_h_in = max_h_cm * cm
    rows_per_page = max(1, int(math.floor(max_h_in / row_h_in)))

    if not box_plot:
        pages = [df.iloc[s:s+rows_per_page].copy()
                 for s in range(0, len(df), rows_per_page)]
        return pages, fig_w_in, max_h_in, rows_per_page

    # box_plot=True:
    if label_col is None or label_col not in df.columns:
        raise ValueError("paginate(...): set box_plot=True and provide label_col present in df.")
    unique_labels = df[label_col].drop_duplicates().tolist()

    pages = []
    for s in range(0, len(unique_labels), rows_per_page):
        page_labels = unique_labels[s:s+rows_per_page]
        subset = df[df[label_col].isin(page_labels)].copy()
        # keep the label order stable as in page_labels
        subset[label_col] = pd.Categorical(subset[label_col], categories=page_labels, ordered=True)
        subset = subset.sort_values(label_col, kind="mergesort")
        pages.append((subset, page_labels))
    return pages, fig_w_in, max_h_in, rows_per_page


def plot_page(axs, page_df, wrapped_labels, box_plot=False, label_col="hypothesis_label", raw_labels=None):
    import numpy as np

    for i, metric in enumerate(METRICS):
        ax = axs[i]

        if not box_plot:
            # Aggregated input: one row per label with *_mean and *_ci
            y = np.arange(len(page_df))
            ax.errorbar(
                page_df[f"{metric}_mean"], y,
                xerr=page_df[f"{metric}_ci"],
                fmt="o", color="black", ecolor="gray",
                elinewidth=1, capsize=2, markersize=4,
            )
            y_for_ticks = y
            shown_wrapped = wrapped_labels
        else:
            if raw_labels is None:
                raise ValueError("plot_page(...): pass raw_labels (unique labels for this page) when box_plot=True.")
            if label_col not in page_df.columns:
                raise KeyError(f"'{label_col}' not in page_df columns.")

            # Build one box per unique label (aggregate folds)
            data = []
            for lbl in raw_labels:
                vals = page_df.loc[page_df[label_col] == lbl, metric].to_numpy()
                if vals.size == 0:
                    vals = np.array([np.nan])
                data.append(vals)

            y = np.arange(len(raw_labels))
            ax.boxplot(
                data,
                vert=False,
                positions=y,
                widths=0.6,
                patch_artist=True,
                boxprops=dict(facecolor="lightgray", color="black"),
                medianprops=dict(color="black", linewidth=1.2),
                whiskerprops=dict(color="black"),
                capprops=dict(color="black"),
                flierprops=dict(marker="o", markersize=3, color="gray", alpha=0.5),
            )
            ax.set_yticks(y)
            ax.set_yticklabels(wrapped_labels)
            y_for_ticks = y
            shown_wrapped = wrapped_labels

        # --- common styling ---
        ax.set_xlim(0, 1)
        ax.set_xticks([0.25, 0.5, 0.75])
        ax.grid(axis="x", linestyle="--", alpha=0.4)
        ax.set_title(metric.capitalize())

        if i == 0:
            ax.set_yticks(y_for_ticks)
            ax.set_yticklabels(shown_wrapped)
            ax.invert_yaxis()
        else:
            ax.tick_params(axis="y", which="both", labelleft=False)

def measure_left_label_width_in(fig, ax_with_labels):
    """Measure widest yticklabel in inches after draw."""
    fig.canvas.draw()
    texts = ax_with_labels.get_yticklabels()
    if not texts:
        return 0.8
    renderer = fig.canvas.get_renderer()
    max_px = max((t.get_window_extent(renderer=renderer).width for t in texts), default=0)
    return max_px / fig.dpi  # inches

def longest_label_info(labels):
    max_len = 0
    max_label = ""
    for l in labels:
        L = len(l.replace("\n", ""))
        if L > max_len:
            max_len = L
            max_label = l
    longest_line_len = max((len(line) for line in max_label.split("\n")), default=0)
    return max_label, max_len, longest_line_len

# =======================
# MAIN
# =======================
os.makedirs(OUTPUT_DIR, exist_ok=True)
df = pd.read_csv(INPUT_FILE)
if NAME_CAT not in df.columns:
    cand = next((c for c in df.columns if "label" in c.lower()), None)
    if cand:
        df = df.rename(columns={cand: NAME_CAT})
    else:
        raise ValueError(f"Expected '{NAME_CAT}' in the input CSV.")

# Compute CI
if BOX_PLOT==False:
    df = compute_ci95(df, N_RUNS)

# ---- Global sort (descending), NaNs last, stable ----
if not BOX_PLOT:
    # same as before: row-level (already aggregated) sort
    f1_key = df[f"{METRICS[3]}_mean"].replace([np.inf, -np.inf], np.nan)
    df["_sort_key"] = f1_key
    df = (df.sort_values("_sort_key", ascending=False, kind="mergesort")
            .drop(columns=["_sort_key"])
            .reset_index(drop=True))
else:
    # BOX PLOT: sort by per-label mean (across folds), then apply that order to all rows
    sort_metric = METRICS[3]  # e.g., "f1_binary"
    tmp = df[[NAME_CAT, sort_metric]].copy()
    tmp[sort_metric] = tmp[sort_metric].replace([np.inf, -np.inf], np.nan)

    label_means = tmp.groupby(NAME_CAT, sort=False)[sort_metric].mean()
    labels_order = (label_means.sort_values(ascending=False, kind="mergesort")
                              .index.tolist())

    # apply this label order to the whole df so folds stay grouped & sorted
    df[NAME_CAT] = pd.Categorical(df[NAME_CAT],
                                  categories=labels_order, ordered=True)
    df = df.sort_values(NAME_CAT, kind="mergesort").reset_index(drop=True)


# Paginate AFTER global sort
pages, fig_w_in, max_h_in, rows_per_page = paginate(
    df, FIG_WIDTH_CM, MAX_H_CM, ROW_H_IN, box_plot=BOX_PLOT, label_col=NAME_CAT
)

combined_pdf_path = os.path.join(OUTPUT_DIR, f"{OUTPUT_BASE_NAME}.pdf")
with PdfPages(combined_pdf_path) as combined_pdf:
    for i, page in enumerate(pages, start=1):
        if BOX_PLOT:
            page_df, raw_labels = page
            wrap_w = WRAP_CHARS
            wrapped = wrap_labels(raw_labels, wrap_w)
            n_rows = len(raw_labels)
        else:
            page_df = page
            wrap_w = WRAP_CHARS
            wrapped = wrap_labels(page_df[NAME_CAT], wrap_w)
            n_rows = len(page_df)

        fig_h_in = min(max_h_in, max(3.0, n_rows * ROW_H_IN))

        # up to 3 passes to relax left margin if needed
        for attempt in range(1, 4):
            fig, axs = plt.subplots(1, 4, figsize=(fig_w_in, fig_h_in), sharey=True)
            if BOX_PLOT:
                plot_page(axs, page_df, wrapped, box_plot=True, label_col=NAME_CAT, raw_labels=raw_labels)
            else:
                plot_page(axs, page_df, wrapped, box_plot=False)

            fig.tight_layout()

            label_w_in = measure_left_label_width_in(fig, axs[0])
            needed_left_in = label_w_in + 0.25
            left_frac = min(0.75, max(0.10, needed_left_in / fig_w_in))
            fig.subplots_adjust(left=left_frac, right=0.98, top=0.90, bottom=0.10)
            fig.canvas.draw()

            re_label_w_in = measure_left_label_width_in(fig, axs[0])
            re_left_frac = (re_label_w_in + 0.25) / fig_w_in
            if re_left_frac > left_frac + 0.02 and attempt < 3:
                plt.close(fig)
                wrap_w = max(20, int(wrap_w * 0.85))
                if BOX_PLOT:
                    wrapped = wrap_labels(raw_labels, wrap_w)
                else:
                    wrapped = wrap_labels(page_df[NAME_CAT], wrap_w)
                continue

            page_pdf = os.path.join(OUTPUT_DIR, f"{OUTPUT_BASE_NAME}_page{i:02d}.pdf")
            page_png = os.path.join(OUTPUT_DIR, f"{OUTPUT_BASE_NAME}_page{i:02d}.png")
            fig.savefig(page_pdf)
            fig.savefig(page_png, dpi=300)
            combined_pdf.savefig(fig)
            plt.close(fig)
            break

print(f"✅ Combined PDF: {combined_pdf_path}")
print(f"✅ Separate per-page PDFs/PNGs in: {OUTPUT_DIR}")
