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
# INPUT_FILE   = "data/model_performance/all_cat_mean_ci.csv"
INPUT_FILE   = "data/model_performance/broad_cat_mean_ci.csv"
OUTPUT_DIR   = "figures/performance_summary/all_cat_mean_ci"
OUTPUT_DIR   = "figures/performance_summary/broad_cat_mean_ci"
OUTPUT_BASE_NAME = "perf_broad_cat_mean_ci"
METRICS      = ["accuracy", "precision_binary", "recall_binary", "f1_binary"]
# METRICS      = ["accuracy", "precision_micro", "recall_micro", "f1_micro"]
# METRICS      = ["accuracy", "precision_binary", "recall_binary", "f1_macro"]
# NAME_CAT = NAME_CAT  
NAME_CAT = "model"
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

def paginate(df, fig_width_cm, max_h_cm, row_h_in):
    cm = 1/2.54
    fig_w_in = fig_width_cm * cm
    max_h_in = max_h_cm * cm
    rows_per_page = max(1, int(math.floor(max_h_in / row_h_in)))
    pages = [df.iloc[s:s+rows_per_page].copy() for s in range(0, len(df), rows_per_page)]
    return pages, fig_w_in, max_h_in, rows_per_page

def plot_page(axs, page_df, wrapped_labels):
    y = np.arange(len(page_df))
    for i, metric in enumerate(METRICS):
        ax = axs[i]
        ax.errorbar(
            page_df[f"{metric}_mean"], y,
            xerr=page_df[f"{metric}_ci"],
            fmt="o", color="black", ecolor="gray",
            elinewidth=1, capsize=2, markersize=4,
        )
        ax.set_xlim(0, 1)
        ax.set_xticks([0.25, 0.5, 0.75])
        ax.grid(axis="x", linestyle="--", alpha=0.4)
        ax.set_title(metric.capitalize())
        # no bottom x-axis label; title above is enough

        if i == 0:
            ax.set_yticks(y)
            ax.set_yticklabels(wrapped_labels)
            ax.invert_yaxis()  # best at TOP
            # no overarching left ylabel
        else:
            # with sharey=True, don't clear shared y labels—just hide visually
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
df = compute_ci95(df, N_RUNS)

# ---- Global sort by F1 mean (descending), NaNs last, stable ----
# Create a sort key to handle NaNs consistently, then stable mergesort
f1_key = df[f"{METRICS[3]}_mean"].replace([np.inf, -np.inf], np.nan)
df["_f1_sort_key"] = f1_key
df = (
    df.sort_values("_f1_sort_key", ascending=False, kind="mergesort")
      .drop(columns=["_f1_sort_key"])
      .reset_index(drop=True)
)

# Paginate AFTER global sort
pages, fig_w_in, max_h_in, rows_per_page = paginate(df, FIG_WIDTH_CM, MAX_H_CM, ROW_H_IN)
if DEBUG:
    print(f"[INFO] Total rows={len(df)}, rows/page≈{rows_per_page}, pages={len(pages)}")

# Combined multipage PDF
combined_pdf_path = os.path.join(OUTPUT_DIR, f"{OUTPUT_BASE_NAME}.pdf")
with PdfPages(combined_pdf_path) as combined_pdf:
    for i, page_df in enumerate(pages, start=1):
        wrap_w = WRAP_CHARS
        wrapped = wrap_labels(page_df[NAME_CAT], wrap_w)
        fig_h_in = min(max_h_in, max(3.0, len(page_df) * ROW_H_IN))

        # iterative: wrap → measure → adjust; up to 3 passes
        for attempt in range(1, 4):
            fig, axs = plt.subplots(1, 4, figsize=(fig_w_in, fig_h_in), sharey=True)
            plot_page(axs, page_df, wrapped)
            fig.tight_layout()

            label_w_in = measure_left_label_width_in(fig, axs[0])
            needed_left_in = label_w_in + 0.25
            left_frac = min(0.75, max(0.10, needed_left_in / fig_w_in))

            if DEBUG:
                longest, tot_chars, longest_line = longest_label_info(wrapped)
                print(f"[PAGE {i} | TRY {attempt}] rows={len(page_df)}, fig=({fig_w_in:.2f}in x {fig_h_in:.2f}in)")
                print(f"   wrap_width={wrap_w} chars, longest_total_chars={tot_chars}, longest_line_chars={longest_line}")
                print(f"   measured_label_width={label_w_in:.2f} in, left_frac={left_frac:.3f}")

            fig.subplots_adjust(left=left_frac, right=0.98, top=0.90, bottom=0.10)
            fig.canvas.draw()

            # quick recheck; if still tight, wrap more and retry
            re_label_w_in = measure_left_label_width_in(fig, axs[0])
            re_left_frac = (re_label_w_in + 0.25) / fig_w_in
            if re_left_frac > left_frac + 0.02 and attempt < 3:
                plt.close(fig)
                wrap_w = max(20, int(wrap_w * 0.85))  # wrap more aggressively
                wrapped = wrap_labels(page_df[NAME_CAT], wrap_w)
                continue

            # Save per-page PDF & PNG
            page_pdf = os.path.join(OUTPUT_DIR, f"{OUTPUT_BASE_NAME}_page{i:02d}.pdf")
            page_png = os.path.join(OUTPUT_DIR, f"{OUTPUT_BASE_NAME}_page{i:02d}.png")
            fig.savefig(page_pdf)           # separate PDF (this page)
            fig.savefig(page_png, dpi=300)  # PNG
            # Also add to combined multipage PDF
            combined_pdf.savefig(fig)
            plt.close(fig)
            break

print(f"✅ Combined PDF: {combined_pdf_path}")
print(f"✅ Separate per-page PDFs/PNGs in: {OUTPUT_DIR}")
