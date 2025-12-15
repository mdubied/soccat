"""
figure_performance_summary.py

Description:
Create the boxplot figures for summary performance. Figures 4, 5, A1-A8 of the paper.

One figure with 4 panels (accuracy, precision, recall, f1), paginated vertically.
 - Sorted globally by decreasing F1 mean (best first), then paginated.
 - Best entries appear at the TOP of each page (invert y-axis).
 - Long labels wrapped and measured to allocate left margin (no clipping).
 - No bottom x-axis labels; no overarching left ylabel.

Outputs:
- PDF boxplot files in "performance_summary/" folder.

Usage (from this directory):
python figure_performance_summary.py --broad_class age_family

(or other broad class name to get all categories within this broad class, or no argument for all broad classes)
"""
import os, textwrap
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
    "--broad_class",
    type=str,
    default="identity",    # fallback if no argument is passed
    help="Broad category name (e.g., 'age_family')"
)
args = parser.parse_args()

# ============================================================
# PARAMETERS (edit here)
# ============================================================
# currently used:
# SETUP_NAME = "cat_per_broad_class_box_plot"
SETUP_NAME   = "broad_class_box_plot"


# Setup-specific paths
if SETUP_NAME == "cat_per_broad_class_box_plot":    # current
    BROAD_CLASS = args.broad_class
    INPUT_FILE   = f"../data/model_performance/{BROAD_CLASS}_per_fold.csv"
    OUTPUT_DIR   = f"performance_summary/cat_per_broad_class_box_plot"
    OUTPUT_BASE_NAME = f"{BROAD_CLASS}_box_plot"
    NAME_CAT = "hypothesis_label"
    LBL_WIDTH_FRAC  = 0.29   # Fixed fraction of figure width for labels (instead of measuring)
elif SETUP_NAME == "broad_class_box_plot":    # current
    INPUT_DIR   = "../data/model_performance"
    OUTPUT_DIR   = "performance_summary/broad_class_box_plot"
    OUTPUT_BASE_NAME = "perf_broad_class_box_plot"
    NAME_CAT = "hypothesis_label"
    LBL_WIDTH_FRAC  = 0.24   # Fixed fraction of figure width for labels (instead of measuring)
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
BROAD_CLASS_LIST = [
    "age_family",
    "identity",
    "labor_market_w_entrepreneurs",
    "profession",
    "real_estate",
    "social_deviance",
    "social_roles",
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
    # broad classes
    "socio_economic": "Socio-economic position",
    "labor_market_w_entrepreneurs": "Labor market position",
    "age_family": "Age and family status",
    "identity": "Identities and minority/ majority status",
    "profession": "Profession",
    "social_roles": "Social roles and behavior",
    "social_deviance": "Social deviance",
    "real_estate": "Real estate ownership",
}

# Score annotations (best fold by F1)
SHOW_BEST_FOLD_SCORES = True
BEST_FOLD_METRIC = "f1_binary"     # fold selector
SCORE_X_DEFAULT = 0.26             # default x position (data coords in [0,1])
SCORE_FMT_MAP = {                  # per-metric formatting if desired
    "accuracy": "{:.2f}",
    "precision_binary": "{:.2f}",
    "recall_binary": "{:.2f}",
    "f1_binary": "{:.2f}",
}

# Optional per-(metric,row) x-position overrides.
# Key: (metric_name, label_string_after_renaming), Value: x-position in [0,1]
SCORE_X_OVERRIDE = {
    # ("precision_binary", "Age and family status"): 0.35,
    # ("f1_binary", "Socio-economic position"): 0.15,
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

def compute_broad_class_boxplot(df, subcat_col, broad_col):
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
    # Rename broad_class → hypothesis_label
    df_broad = df_agg.rename(columns={broad_col: "hypothesis_label"})

    # Ensure no subcategories remain — this dataframe is ONLY broad categories
    # (just a safety check for debugging)
    assert df_broad["hypothesis_label"].nunique() == df[broad_col].nunique()

    return df_broad

def plot_data(axs, ax_N, df, wrapped_labels, label_col="hypothesis_label", raw_labels=None):

    for i, metric in enumerate(METRICS):
        ax = axs[i]

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

        # Annotate "best fold" scores (fold with highest F1 for a label)
        if SHOW_BEST_FOLD_SCORES:
            for yi, lbl in zip(y, raw_labels):
                sub = df.loc[df[label_col] == lbl, ["fold"] + METRICS].copy()
                if sub.empty:
                    continue

                # pick the fold with maximum F1 (BEST_FOLD_METRIC)
                sub = sub.replace([np.inf, -np.inf], np.nan).dropna(subset=[BEST_FOLD_METRIC])
                if sub.empty:
                    continue
                best_idx = sub[BEST_FOLD_METRIC].idxmax()
                best_row = sub.loc[best_idx]

                val = best_row.get(metric, np.nan)
                if pd.isna(val):
                    continue

                # x-position: default or overridden per (metric, label)
                x_pos = SCORE_X_OVERRIDE.get((metric, str(lbl)), SCORE_X_DEFAULT)

                fmt = SCORE_FMT_MAP.get(metric, "{:.2f}")
                s = fmt.format(float(val))
                s = rf"\textsf{{\textit{{{s}}}}}"   # sans-serif + italic
                ax.text(x_pos, yi, s, ha="left", va="center", fontsize=7)




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

            N_vals = (
                df.groupby(label_col, observed=False)["n_pos_entail"]
                .first()            # or .iloc[0], same effect
                .reindex(raw_labels)
            )     
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
if SETUP_NAME == "broad_class_box_plot":
    dfs = []  # collect all subcategory data

    for bc in BROAD_CLASS_LIST:
        input_file = f"{INPUT_DIR}/{bc}_per_fold.csv"
        assert os.path.exists(input_file), f"File missing: {input_file}"

        df_temp = pd.read_csv(input_file)
        df_temp["broad_class"] = bc  # tag the source
        dfs.append(df_temp)

    # Merge all subcategory data (long format)
    df_raw = pd.concat(dfs, ignore_index=True)

    # Compute broad category aggregation — RETURNS ONLY broad categories
    df = compute_broad_class_boxplot(
        df_raw,
        subcat_col="hypothesis_label",
        broad_col="broad_class"
    )

else:
    df = pd.read_csv(INPUT_FILE)
    
assert NAME_CAT in df.columns, f"Missing '{NAME_CAT}' column in CSV."
df[NAME_CAT] = df[NAME_CAT].replace(RENAME_DICT)

# Global sorting according to F1 mean score
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



df_plot, raw_labels = data_to_plot
wrapped = wrap_labels(raw_labels, wrap_w)
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


plot_data(axs, ax_N, df_plot, wrapped, label_col=NAME_CAT, raw_labels=raw_labels)


# Save figure
output_path = os.path.join(OUTPUT_DIR, f"{OUTPUT_BASE_NAME}.pdf")
fig.savefig(output_path)

print(f"✅ Saved file as: {output_path}")
