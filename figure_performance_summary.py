# File: figure_performance_summary.py
# One figure with 4 panels (accuracy, precision, recall, f1), paginated vertically.
# - Sorted globally by decreasing F1 mean (best first), then paginated.
# - Best entries appear at the TOP of each page (invert y-axis).
# - Long labels wrapped and measured to allocate left margin (no clipping).
# - No bottom x-axis labels; no overarching left ylabel.
#
# Usage (example):
# python figure_performance_summary.py --broad_cat age_family
# TODO: remove legacy options if no longer needed. + CI calculation if not used

import os, math, textwrap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import argparse

# ===========================================================
# PARSE COMMAND-LINE ARGUMENTS
# ============================================================
parser = argparse.ArgumentParser()
parser.add_argument(
    "--broad_cat",
    type=str,
    default="identity",    # fallback if no argument is passed
    help="Broad category name (e.g., 'age_family')"
)
args = parser.parse_args()

# ============================================================
# PARAMETERS (edit here)
# ============================================================
# currently used:
# SETUP_NAME = "cat_per_broad_cat_box_plot"
SETUP_NAME   = "broad_cat_box_plot"
# legacy options (no gurantee they still work):
# SETUP_NAME = "all_cat_mean_ci"
# SETUP_NAME   = "all_cat_box_plot"
# SETUP_NAME   = "broad_cat_mean_ci"


# Setup-specific paths
if SETUP_NAME == "all_cat_mean_ci": # legacy
    INPUT_FILE   = "data/model_performance/all_cat_mean_ci.csv"
    OUTPUT_DIR   = "figures/performance_summary/all_cat_mean_ci"
    OUTPUT_BASE_NAME = "perf_all_cat_mean_ci"
    NAME_CAT = "hypothesis_label"
    BOX_PLOT    = False
elif SETUP_NAME == "all_cat_box_plot":  #legacy
    INPUT_FILE   = "data/model_performance/all_cat_box_plot.csv"
    OUTPUT_DIR   = "figures/performance_summary/all_cat_box_plot"
    OUTPUT_BASE_NAME = "perf_all_cat_box_plot"
    NAME_CAT = "hypothesis_label"
    BOX_PLOT    = True
elif SETUP_NAME == "broad_cat_mean_ci": #legacy
    INPUT_FILE   = "data/model_performance/broad_cat_mean_ci.csv"
    OUTPUT_DIR   = "figures/performance_summary/broad_cat_mean_ci"
    OUTPUT_BASE_NAME = "perf_broad_cat_mean_ci"
    NAME_CAT = "model"
    BOX_PLOT    = False
elif SETUP_NAME == "cat_per_broad_cat_box_plot":    # current
    BROAD_CAT = args.broad_cat
    INPUT_FILE   = f"data/model_performance/{BROAD_CAT}_per_fold.csv"
    OUTPUT_DIR   = f"figures/performance_summary/cat_per_broad_cat_box_plot"
    OUTPUT_BASE_NAME = f"{BROAD_CAT}_box_plot"
    NAME_CAT = "hypothesis_label"
    BOX_PLOT    = True
    LBL_WIDTH_FRAC  = 0.29   # Fixed fraction of figure width for labels (instead of measuring)
elif SETUP_NAME == "broad_cat_box_plot":    # current
    INPUT_DIR   = "data/model_performance"
    OUTPUT_DIR   = "figures/performance_summary/broad_cat_box_plot"
    OUTPUT_BASE_NAME = "perf_broad_cat_box_plot"
    NAME_CAT = "hypothesis_label"
    BOX_PLOT    = True
    LBL_WIDTH_FRAC  = 0.23   # Fixed fraction of figure width for labels (instead of measuring)
else:
    raise ValueError(f"Unknown SETUP_NAME: {SETUP_NAME}")

# Performance metrics to plot
METRICS      = ["accuracy", "precision_binary", "recall_binary", "f1_binary"]
METRIC_TITLE_MAP = {
    "accuracy": "Accuracy",
    "precision_binary": "Precision",
    "recall_binary": "Recall",
    "f1_binary": "F1 Score",
}

# Broad category listing
BROAD_CAT_LIST = [
    "age_family",
    "identity",
    "labor_market_w_entrepreneurs",
    "profession",
    "real_estate",
    "social_deviance",
    "social_rules_wo_volunteers",
    "socio_economic"
]

# Optional renaming of entries for display
RENAME_DICT = {
    # specific categories
    "entrepreneurs in [specific] sector": "entrepreneurs in specific sector",
    "lgbtqqia+": "LGBTQIA+",
    "people with an immigration background, including immigrants": "people with immigration background",
    "offenders, criminals, prisoners and/or accused people": "offenders, criminals, prisoners, accused people",
    "terrorists, rebels, revolutionaries and/or movements of armed resistance": "terrorists, revolutionaries, rebels, armed resistance",
    # broad categories
    "age_family": "age and family status",
    "business_activity": "business activity",
    "identity": "identity",
    "labor_market_w_entrepreneurs": "labor market position",
    "profession": "profession",
    "real_estate": "real estate",
    "social_deviance": "social deviance",
    "social_rules_wo_volunteers": "social roles",
    "socio_economic": "socio-economic status"
}



# Figure parameters
N_RUNS          = 5      # for 95% CI
FIG_WIDTH_CM    = 14.0   # total figure width
WRAP_CHARS      = 28     # initial wrap width (characters)
ROW_HEIGHT_CM   = 0.6    # Vertical height per row (cm)
TOP_MARGIN_CM   = 0.5    # Reserved vertical space for titles + x-axis annotation (cm)
BOTTOM_MARGIN_CM= 0.5    # Reserved vertical space for x-axis annotation (cm)

# Convert cm to inches, assign to variables
cm = 1/2.54
fig_w_in = FIG_WIDTH_CM * cm
ROW_HEIGHT_IN = ROW_HEIGHT_CM * cm
TOP_MARGIN_IN = TOP_MARGIN_CM * cm
BOTTOM_MARGIN_IN = BOTTOM_MARGIN_CM * cm
wrap_w = WRAP_CHARS

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Latin Modern Sans"],  # or "cmss"
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "text.latex.preamble": r"""
        \usepackage[T1]{fontenc}
        \usepackage{lmodern}
        \renewcommand{\familydefault}{\sfdefault}
    """,
})

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def wrap_labels(labels, width_chars=40):
    return ["\n".join(textwrap.wrap(str(lbl), width_chars)) for lbl in labels]

def compute_ci95(df, n_runs):
    for m in METRICS:
        if f"{m}_std" not in df.columns or f"{m}_mean" not in df.columns:
            raise ValueError(f"Missing columns for '{m}': need {m}_mean and {m}_std")
        df[f"{m}_ci"] = 1.96 * df[f"{m}_std"] / np.sqrt(n_runs)
    return df

def compute_broad_cat_boxplot(df, subcat_col, broad_col):
    """
    Returns a clean dataframe that contains ONLY:
        hypothesis_label (= broad category code),
        fold,
        metrics (weighted),
        n_pos_entail  (sum across subcategories)

    Ready for plotting code.
    """

    assert broad_col in df.columns, f"'{broad_col}' not found in df"
    assert subcat_col in df.columns, f"'{subcat_col}' not found in df"
    assert "fold" in df.columns,     "Missing 'fold' column in df"
    assert "n_pos_entail" in df.columns, "Missing 'n_pos_entail' column"

    # use your global METRICS
    metric_cols = METRICS

    # -------------------------------------------
    # 1) Weighted aggregation per fold & category
    # -------------------------------------------
    def agg_fold(group):
        out = {}
        for m in metric_cols:
            out[m] = np.average(group[m], weights=group["n_pos_entail"])
        out["n_pos_entail"] = group["n_pos_entail"].sum()
        return pd.Series(out)

    df_agg = (
        df.groupby([broad_col, "fold"], observed=False)
        .apply(agg_fold, include_groups=False)
        .reset_index()
    )


    # -------------------------------------------
    # 2) Drop subcategory labels completely
    # -------------------------------------------
    # Rename broad_cat → hypothesis_label
    df_broad = df_agg.rename(columns={broad_col: "hypothesis_label"})

    # Ensure no subcategories remain — this dataframe is ONLY broad categories
    # (just a safety check for debugging)
    assert df_broad["hypothesis_label"].nunique() == df[broad_col].nunique()

    return df_broad

def plot_data(axs, ax_N, df, wrapped_labels, box_plot=False, label_col="hypothesis_label", raw_labels=None):

    for i, metric in enumerate(METRICS):
        ax = axs[i]

        if not box_plot:
            # Aggregated input: one row per label with *_mean and *_ci
            y = np.arange(len(df))
            ax.errorbar(
                df[f"{metric}_mean"], y,
                xerr=df[f"{metric}_ci"],
                fmt="o", color="black", ecolor="gray",
                elinewidth=1, capsize=2, markersize=4,
            )
            y_for_ticks = y
            shown_wrapped = wrapped_labels
        else:
            # Build one box per unique label (aggregate folds)
            data = []
            for lbl in raw_labels:
                vals = df.loc[df[label_col] == lbl, metric].to_numpy()
                if vals.size == 0:
                    vals = np.array([np.nan])
                data.append(vals)

            y = np.arange(len(raw_labels))[::-1]
            ax.boxplot(
                data,
                vert=False,
                positions=y,
                widths=0.8,
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
        ax.set_xticks([0, 0.5, 1])
        ax.set_xticklabels(["0", "0.5", "1"])
        ax.set_xticks([0.25, 0.75], minor=True)
        ax.grid(axis="x", linestyle="--", alpha=0.4)
        ax.grid(axis="x", which="minor", linestyle="--", alpha=0.4)
        ax.set_title(METRIC_TITLE_MAP.get(metric, metric.capitalize()))

        if i == 0:
            ax.set_yticks(y_for_ticks)
            ax.set_yticklabels(shown_wrapped)
        else:
            ax.tick_params(axis="y", which="both", labelleft=False)
            ax.tick_params(axis="y", length=0)  # hide tick marks completely

        # Plot N values if available
        if "n_pos_entail" in df.columns:
            if box_plot:
                N_vals = (
                    df.groupby(label_col, observed=False)["n_pos_entail"]
                    .first()            # or .iloc[0], same effect
                    .reindex(raw_labels)
                )
            else:
                N_vals = df["n_pos_entail"]
            y = np.arange(len(N_vals))[::-1]  # match row order

            # Display text on empty axis
            ax_N.set_title("N")
            for yi, val in zip(y, N_vals):
                ax_N.text(0.5, yi, str(val), ha="center", va="center", fontsize=8)

            ax_N.set_ylim(min(y)-0.5, max(y)+0.5)
            ax_N.set_xlim(0, 1)
            ax_N.set_xticks([])
            ax_N.set_yticks([])
            ax_N.spines['top'].set_visible(False)
            ax_N.spines['bottom'].set_visible(False)
            ax_N.spines['left'].set_visible(False)
            ax_N.spines['right'].set_visible(False)

# ============================================================
# MAIN
# ============================================================

# Load data
os.makedirs(OUTPUT_DIR, exist_ok=True)
if SETUP_NAME == "broad_cat_box_plot":
    dfs = []  # collect all subcategory data

    for bc in BROAD_CAT_LIST:
        input_file = f"{INPUT_DIR}/{bc}_per_fold.csv"
        assert os.path.exists(input_file), f"File missing: {input_file}"

        df_temp = pd.read_csv(input_file)
        df_temp["broad_cat"] = bc  # tag the source
        dfs.append(df_temp)

    # Merge all subcategory data (long format)
    df_raw = pd.concat(dfs, ignore_index=True)

    # Compute broad category aggregation — RETURNS ONLY broad categories
    df = compute_broad_cat_boxplot(
        df_raw,
        subcat_col="hypothesis_label",
        broad_col="broad_cat"
    )

else:
    df = pd.read_csv(INPUT_FILE)
    
assert NAME_CAT in df.columns, f"Missing '{NAME_CAT}' column in CSV."
df[NAME_CAT] = df[NAME_CAT].replace(RENAME_DICT)

# Compute CI
if BOX_PLOT==False:
    df = compute_ci95(df, N_RUNS)

# Global sorting according to F1 mean score
if not BOX_PLOT:
    f1_key = df[f"{METRICS[3]}_mean"].replace([np.inf, -np.inf], np.nan)
    df["_sort_key"] = f1_key
    df = (df.sort_values("_sort_key", ascending=False, kind="mergesort")
            .drop(columns=["_sort_key"])
            .reset_index(drop=True))
else:
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
    df["n_pos_entail"] = df["n_pos_entail"].astype(int)

# Prepare data labeling
unique_labels = df[NAME_CAT].drop_duplicates().tolist()
n_rows = len(unique_labels)
data_to_plot = (df.sort_values(NAME_CAT, kind="mergesort"), unique_labels)


if BOX_PLOT:
    df_plot, raw_labels = data_to_plot
    wrapped = wrap_labels(raw_labels, wrap_w)
    wrapped = [
        "\n".join(
            (lines[0][0].upper() + lines[0][1:]) if idx == 0 and lines[0] else line
            for idx, line in enumerate(lines)
        )
        for lines in [lbl.split("\n") for lbl in wrapped]
    ]

else:
    df_plot, _ = data_to_plot
    wrapped = wrap_labels(df_plot[NAME_CAT], wrap_w)
    wrapped = [
        "\n".join(
            (lines[0][0].upper() + lines[0][1:]) if idx == 0 and lines[0] else line
            for idx, line in enumerate(lines)
        )
        for lines in [lbl.split("\n") for lbl in wrapped]
    ]

    
# Plot figure
fig_h_in = TOP_MARGIN_IN + BOTTOM_MARGIN_IN + n_rows * ROW_HEIGHT_IN
gs = gridspec.GridSpec(
    nrows=1, ncols=5,            
    left=LBL_WIDTH_FRAC, right=0.98,
    bottom=BOTTOM_MARGIN_IN / fig_h_in,
    top=1 - (TOP_MARGIN_IN / fig_h_in),
    wspace=0.15,
    width_ratios=[1, 1, 1, 1, 0.2]  
)

fig = plt.figure(figsize=(fig_w_in, fig_h_in))
axs = [fig.add_subplot(gs[0, i]) for i in range(4)]
ax_N = fig.add_subplot(gs[0, 4]) 

if BOX_PLOT:
    plot_data(axs, ax_N, df_plot, wrapped, box_plot=True, label_col=NAME_CAT, raw_labels=raw_labels)
else:
    plot_data(axs, ax_N, df_plot, wrapped, box_plot=False)

# Save figure
output_path = os.path.join(OUTPUT_DIR, f"{OUTPUT_BASE_NAME}.pdf")
fig.savefig(output_path)

print(f"✅ Saved file as: {output_path}")
