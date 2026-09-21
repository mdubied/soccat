"""
step_2_boxplot.py

Description:
Create the boxplot figures for summary performance. Figures 4, 5, A1-A8 of the paper.

One figure with 4 panels (binary precision, recall, F1, and macro F1), paginated vertically.
 - Sorted globally by decreasing F1 mean (best first), then paginated.
 - Best entries appear at the TOP of each page (invert y-axis).
 - Long labels wrapped and measured to allocate left margin (no clipping).
 - No bottom x-axis labels; no overarching left ylabel.

Outputs:
- PDF boxplot files in "step_2/boxplots/" folder.

Usage (from this directory):
python step_2_boxplot.py --broad_class age_family

(or other broad class name to get all categories within this broad class, or no argument for all broad classes)
"""
import os, re, glob, textwrap, itertools
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
    default=None,    
    help="Broad category name (e.g., 'age_family')"
)
args = parser.parse_args()

# ============================================================
# PARAMETERS (edit here)
# ============================================================
if args.broad_class is None:
    SETUP_NAME = "broad_class_box_plot"
else:
    SETUP_NAME = "cat_per_broad_class_box_plot"

# Setup-specific paths
STEP_2_DATA_DIR = "../data/model_performance/step_2"

# Each broad class has its own folder directly under STEP_2_DATA_DIR
# (containing per-fold, per-label CSVs); some were renamed in the process.
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

if SETUP_NAME == "cat_per_broad_class_box_plot":
    BROAD_CLASS = args.broad_class
    OUTPUT_DIR   = f"step_2/boxplots/cat_per_broad_class_box_plot"
    OUTPUT_BASE_NAME = f"{BROAD_CLASS}_box_plot"
    NAME_CAT = "hypothesis_label"
    LBL_WIDTH_FRAC  = 0.29   # Fixed fraction of figure width for labels (instead of measuring)
elif SETUP_NAME == "broad_class_box_plot":
    OUTPUT_DIR   = "step_2/boxplots/broad_class_box_plot"
    OUTPUT_BASE_NAME = "perf_broad_class_box_plot"
    NAME_CAT = "hypothesis_label"
    LBL_WIDTH_FRAC  = 0.24   # Fixed fraction of figure width for labels (instead of measuring)
    MERGED_LBL_WIDTH_FRAC = 0.29  # merged all-categories figure needs the wider margin (long category names)
else:
    raise ValueError(f"Unknown SETUP_NAME: {SETUP_NAME}")

# Performance metrics to plot
METRICS      = ["precision_binary", "recall_binary", "f1_binary", "f1_macro"]
METRIC_TITLE_MAP = {
    "precision_binary": "Bin. Precision",
    "recall_binary": "Bin. Recall",
    "f1_binary": "Bin. F1",
    "f1_macro": "Macro F1",
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

# Score annotations (best fold by macro F1 -- this fold's model is the one
# made publicly available, so selection should be robust to class imbalance
# rather than driven by the positive-class-only, small-sample-sensitive
# f1_binary)
SHOW_BEST_FOLD_SCORES = True
BEST_FOLD_METRIC = "f1_macro"     # fold selector
SCORE_X_DEFAULT = 0.04            # default x position (data coords in [0,1])
SCORE_FMT_MAP = {                  # per-metric formatting if desired
    "precision_binary": "{:.2f}",
    "recall_binary": "{:.2f}",
    "f1_binary": "{:.2f}",
    "f1_macro": "{:.2f}",
}

# Optional per-(metric,row) x-position and background color of text overrides,
# keyed per broad class so the merged all-categories figure can reuse every
# broad class's own tuning (union of all of them) instead of just one.
# Key: (metric_name, label_string_after_renaming), Value: x-position in [0,1]
TEXT_BBOX_COLOR_DEFAULT = "white"

CATEGORY_SCORE_X_OVERRIDES = {
    "identity": {
        ("recall_binary", "muslims"): 0.35,
        ("precision_binary", "cisgender and heterosexuals"): 0.23,
        ("precision_binary", "Jews"): 0.3,
        ("precision_binary", "Muslims"): 0.72,
        ("precision_binary", "Christians"): 0.45,
        ("precision_binary", "Ethnic and racial minorities"): 0.72,
        ("precision_binary", "multiple (or other) religious or minority groups"): 0.72,
        ("recall_binary", "Ethnic and racial minorities"): 0.72,
        ("recall_binary", "multiple (or other) religious or minority groups"): 0.72,
        ("f1_binary", "Ethnic and racial minorities"): 0.72,
        ("f1_binary", "multiple (or other) religious or minority groups"): 0.72,
    },
    "age_family": {
        ("recall_binary", "elderly"): 0.02,
        ("recall_binary", "middle-aged and pre-retirement age groups"): 0.58,
    },
}
CATEGORY_BBOX_COLOR_OVERRIDES = {
    "identity": {
        ("recall_binary", "Ethnic and racial minorities"): "lightgray",
        ("recall_binary", "multiple (or other) religious or minority groups"): "lightgray",
        ("precision_binary", "cisgender and heterosexuals"): "lightgray",
        ("precision_binary", "Jews"): "lightgray",
                
    },
    "age_family": {
        ("precision_binary", "middle-aged and pre-retirement age groups"): "lightgray",
        ("f1_binary", "middle-aged and pre-retirement age groups"): "lightgray",
    },
}
# broad-class-LEVEL overrides, for the aggregated (8-broad-class) figure only
AGGREGATE_SCORE_X_OVERRIDE = {}
AGGREGATE_BBOX_COLOR_OVERRIDE = {}

if SETUP_NAME == "cat_per_broad_class_box_plot":
    SCORE_X_OVERRIDE = CATEGORY_SCORE_X_OVERRIDES.get(args.broad_class, {})
    TEXT_BBOX_COLOR_OVERRIDE = CATEGORY_BBOX_COLOR_OVERRIDES.get(args.broad_class, {})
else:  # broad_class_box_plot: figure 1 shows aggregated broad-class rows
    SCORE_X_OVERRIDE = AGGREGATE_SCORE_X_OVERRIDE
    TEXT_BBOX_COLOR_OVERRIDE = AGGREGATE_BBOX_COLOR_OVERRIDE

# union of every broad class's own overrides plus the aggregate-level ones,
# for the merged all-categories figure (which now shows both category rows
# AND each broad class's own rollup row, so both sets of keys can match)
MERGED_SCORE_X_OVERRIDE = {
    **AGGREGATE_SCORE_X_OVERRIDE,
    **{k: v for d in CATEGORY_SCORE_X_OVERRIDES.values() for k, v in d.items()},
}
MERGED_BBOX_COLOR_OVERRIDE = {
    **AGGREGATE_BBOX_COLOR_OVERRIDE,
    **{k: v for d in CATEGORY_BBOX_COLOR_OVERRIDES.values() for k, v in d.items()},
}




# Figure parameters
N_RUNS          = 5      # for 95% CI
FIG_WIDTH_CM    = 14.0   # total figure width
WRAP_CHARS      = 28     # initial wrap width (characters)
ROW_HEIGHT_CM   = 0.5    # Vertical height per row (cm)
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

def compute_y_positions(raw_labels, header_labels=frozenset(), extra_gap_rows=0.0):
    """
    Top-to-bottom row positions (first label highest, decreasing downward),
    normally 1 row-height apart. If extra_gap_rows > 0, an extra gap of that
    many row-heights is inserted above any header label other than the very
    first row -- i.e. between the last category of a broad class and the
    next broad class's header, in the merged all-categories figure.
    """
    positions = np.zeros(len(raw_labels))
    for i in range(1, len(raw_labels)):
        gap = 1.0 + (extra_gap_rows if raw_labels[i] in header_labels else 0.0)
        positions[i] = positions[i - 1] - gap
    return positions

def load_broad_class_df(broad_class):
    """
    Load per-fold, per-label performance data for a broad class: its own
    folder directly under STEP_2_DATA_DIR, with one "fold_X_per_label.csv"
    file per fold (no "fold" column in the file itself, added here).
    """
    folder = FOLDER_NAME_MAP.get(broad_class, broad_class)
    new_dir = f"{STEP_2_DATA_DIR}/{folder}"
    fold_files = sorted(
        glob.glob(f"{new_dir}/fold_*_per_label.csv"),
        key=lambda f: int(re.search(r"fold_(\d+)_per_label", f).group(1))
    )
    assert fold_files, f"No fold_*_per_label.csv found for broad class '{broad_class}' in {new_dir}"

    dfs = []
    for f in fold_files:
        fold_num = int(re.search(r"fold_(\d+)_per_label", f).group(1))
        d = pd.read_csv(f)
        d["fold"] = fold_num
        dfs.append(d)
    return pd.concat(dfs, ignore_index=True)

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

def select_best_fold_row(sub, fixed_fold):
    """
    Pick the row of `sub` (must have a "fold" column and BEST_FOLD_METRIC)
    to annotate/report: either the fixed fold shared across all labels, or
    (if fixed_fold is None) the label's own best fold by BEST_FOLD_METRIC.
    Returns None if no suitable row exists.
    """
    sub = sub.replace([np.inf, -np.inf], np.nan)
    if fixed_fold is not None:
        sub_fold = sub.loc[sub["fold"] == fixed_fold]
        if sub_fold.empty:
            return None
        return sub_fold.iloc[0]

    sub = sub.dropna(subset=[BEST_FOLD_METRIC])
    if sub.empty:
        return None
    best_idx = sub[BEST_FOLD_METRIC].idxmax()
    return sub.loc[best_idx]

def best_fold_for_group(df_group):
    """Fold with the highest n_pos_entail-weighted BEST_FOLD_METRIC across the group's categories."""
    fold_scores = df_group.groupby("fold", observed=False).apply(
        lambda g: np.average(g[BEST_FOLD_METRIC], weights=g["n_pos_entail"]),
        include_groups=False,
    )
    return int(fold_scores.idxmax())

def balanced_contiguous_split(sizes, n_parts):
    """
    Split an ORDERED sequence of group sizes into n_parts contiguous chunks,
    minimizing the largest chunk's total size (order is preserved -- groups
    are never reshuffled, only cut into contiguous runs).
    Returns a list of n_parts lists of indices into `sizes`.
    """
    n = len(sizes)
    if n_parts >= n:
        return [[i] for i in range(n)]

    best_cuts, best_max = None, None
    for cuts in itertools.combinations(range(1, n), n_parts - 1):
        bounds = (0,) + cuts + (n,)
        chunk_sums = [sum(sizes[bounds[k]:bounds[k + 1]]) for k in range(n_parts)]
        worst = max(chunk_sums)
        if best_max is None or worst < best_max:
            best_max, best_cuts = worst, bounds

    return [list(range(best_cuts[k], best_cuts[k + 1])) for k in range(n_parts)]

def plot_data(axs, ax_N, df, wrapped_labels, label_col="hypothesis_label", raw_labels=None,
              fixed_fold=None, header_labels=frozenset(),
              score_x_override=None, bbox_color_override=None, extra_gap_rows=0.0):
    """
    fixed_fold: either a single fold index (int) applied to every label, or a
    dict {label: fold_index} for a per-label fixed fold (used when merging
    categories from several broad classes, each with its own best fold).
    header_labels: labels that start a new section (e.g. a broad class name
    heading its own categories in the merged figure) -- these still get a
    box/annotation/N like any other row (e.g. a broad class's own rollup
    row), the set is only used to add extra spacing above them (see
    extra_gap_rows / compute_y_positions); bold vs. plain label styling is
    handled by the caller via wrapped_labels, not here.
    score_x_override / bbox_color_override: per-(metric,label) annotation
    tweaks (see SCORE_X_OVERRIDE / TEXT_BBOX_COLOR_OVERRIDE / their MERGED_*
    variants); default to the module-level SCORE_X_OVERRIDE / TEXT_BBOX_COLOR_OVERRIDE.
    extra_gap_rows: extra vertical spacing (in row-heights) inserted above
    each header label other than the first row (see compute_y_positions).
    """
    if score_x_override is None:
        score_x_override = SCORE_X_OVERRIDE
    if bbox_color_override is None:
        bbox_color_override = TEXT_BBOX_COLOR_OVERRIDE

    def fold_for(lbl):
        return fixed_fold.get(lbl) if isinstance(fixed_fold, dict) else fixed_fold

    data_labels = list(raw_labels)

    for i, metric in enumerate(METRICS):
        ax = axs[i]

        y = compute_y_positions(raw_labels, header_labels, extra_gap_rows)
        y_by_label = dict(zip(raw_labels, y))

        # Build one box per label (aggregate folds)
        box_positions = []
        data = []
        for lbl in data_labels:
            vals = df.loc[df[label_col] == lbl, metric].to_numpy()
            if vals.size == 0:
                vals = np.array([np.nan])
            box_positions.append(y_by_label[lbl])
            data.append(vals)

        ax.boxplot(
            data,
            vert=False,
            positions=box_positions,
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
        ax.set_ylim(min(y) - 0.5, max(y) + 0.5)
        y_for_ticks = y
        shown_wrapped = wrapped_labels

        # Annotate "best fold" scores (fold with highest F1 for a label)
        if SHOW_BEST_FOLD_SCORES:
            for lbl in data_labels:
                yi = y_by_label[lbl]
                sub = df.loc[df[label_col] == lbl, ["fold"] + METRICS].copy()
                if sub.empty:
                    continue

                best_row = select_best_fold_row(sub, fold_for(lbl))
                if best_row is None:
                    continue

                val = best_row.get(metric, np.nan)
                if pd.isna(val):
                    continue

                x_pos = score_x_override.get((metric, str(lbl)), SCORE_X_DEFAULT)
                fmt = SCORE_FMT_MAP.get(metric, "{:.2f}")
                s = fmt.format(float(val))
                s = rf"\textsf{{\textit{{{s}}}}}"
                bg_color = bbox_color_override.get(
                    (metric, str(lbl)),
                    TEXT_BBOX_COLOR_DEFAULT
                )
                ax.text(x_pos, yi, s,
                        ha="left", va="center",
                        fontsize=7,
                        bbox=dict(
                        facecolor=bg_color,
                        edgecolor="none",
                        pad=0.6)
                )




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

            N_map = (
                df.groupby(label_col, observed=False)["n_pos_entail"]
                .mean()
                .round()
            )

            # Display text on empty axis
            ax_N.set_title("N")
            for lbl in data_labels:
                val = N_map.get(lbl, np.nan)
                if pd.isna(val):
                    continue
                ax_N.text(0.5, y_by_label[lbl], str(int(val)), ha="center", va="center", fontsize=7)

            ax_N.set_ylim(min(y) - 0.5, max(y) + 0.5)
            ax_N.set_xlim(0, 1)
            ax_N.set_xticks([])
            ax_N.set_yticks([])
            ax_N.spines['top'].set_visible(False)
            ax_N.spines['bottom'].set_visible(False)
            ax_N.spines['left'].set_visible(False)
            ax_N.spines['right'].set_visible(False)

def render_boxplot_figure(df_plot, raw_labels, wrapped_labels, fixed_fold, output_path,
                           label_col=None, header_labels=frozenset(), lbl_width_frac=None,
                           score_x_override=None, bbox_color_override=None, extra_gap_rows=0.0):
    """Builds the gridspec + figure for one page of boxplot rows and saves it."""
    if lbl_width_frac is None:
        lbl_width_frac = LBL_WIDTH_FRAC
    y = compute_y_positions(raw_labels, header_labels, extra_gap_rows)
    n_rows_equiv = (y[0] - y[-1]) + 1 if len(y) else 0  # row-height units, incl. any extra gaps
    fig_h_in = TOP_MARGIN_IN + BOTTOM_MARGIN_IN + n_rows_equiv * ROW_HEIGHT_IN
    gs = gridspec.GridSpec(
        nrows=1, ncols=5,
        left=lbl_width_frac, right=0.98,
        bottom=BOTTOM_MARGIN_IN / fig_h_in,
        top=1 - (TOP_MARGIN_IN / fig_h_in),
        wspace=0.15,
        width_ratios=[1, 1, 1, 1, 0.2]
    )

    fig = plt.figure(figsize=(fig_w_in, fig_h_in))
    axs = [fig.add_subplot(gs[0, i]) for i in range(4)]
    ax_N = fig.add_subplot(gs[0, 4])

    plot_data(axs, ax_N, df_plot, wrapped_labels, label_col=label_col, raw_labels=raw_labels,
              fixed_fold=fixed_fold, header_labels=header_labels,
              score_x_override=score_x_override, bbox_color_override=bbox_color_override,
              extra_gap_rows=extra_gap_rows)

    fig.savefig(output_path)
    plt.close(fig)
    print(f"Saved file as: {output_path}")

def wrap_and_capitalize(raw_labels, width_chars):
    """Wrap each label and capitalize the first letter of its first line."""
    wrapped = wrap_labels(raw_labels, width_chars)
    return [
        "\n".join(
            (lines[0][0].upper() + lines[0][1:]) if idx == 0 and lines[0] else line
            for idx, line in enumerate(lines)
        )
        for lines in [lbl.split("\n") for lbl in wrapped]
    ]

# ============================================================
# MAIN
# ============================================================

# Load data
os.makedirs(OUTPUT_DIR, exist_ok=True)
if SETUP_NAME == "broad_class_box_plot":
    dfs = []  # collect all subcategory data

    for bc in BROAD_CLASS_LIST:
        df_temp = load_broad_class_df(bc)
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
    df = load_broad_class_df(BROAD_CLASS)

assert NAME_CAT in df.columns, f"Missing '{NAME_CAT}' column in CSV."
df[NAME_CAT] = df[NAME_CAT].replace(RENAME_DICT)

# Choose ONE fold for all rows (only for cat_per_broad_class_box_plot)
FIXED_FOLD_FOR_ANNOT = None
if SETUP_NAME == "cat_per_broad_class_box_plot":
    # Broad-class best fold: weighted mean F1 across categories in that fold
    fold_scores = (
        df.groupby("fold", observed=False)
        .apply(
            lambda g: np.average(g[BEST_FOLD_METRIC], weights=g["n_pos_entail"]),
            include_groups=False
        )
    )
    FIXED_FOLD_FOR_ANNOT = int(fold_scores.idxmax())

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

# Print F1 median / mean / best-fold per row (binary and macro)
print(f"\n{'Label':<60} {'Bin.F1 med':>10} {'Bin.F1 mean':>11} {'Bin.F1 best':>11} {'Macro F1 best':>14}")
print("-" * 110)
for lbl in raw_labels:
    sub = df_plot.loc[df_plot[NAME_CAT] == lbl, ["fold"] + METRICS]
    median = sub["f1_binary"].median()
    mean = sub["f1_binary"].mean()
    best_row = select_best_fold_row(sub, FIXED_FOLD_FOR_ANNOT)
    best_bin = best_row["f1_binary"] if best_row is not None else np.nan
    best_macro = best_row["f1_macro"] if best_row is not None else np.nan
    print(f"{str(lbl):<60} {median:>10.4f} {mean:>11.4f} {best_bin:>11.4f} {best_macro:>14.4f}")
print()

wrapped = wrap_and_capitalize(raw_labels, wrap_w)

# Plot figure
output_path = os.path.join(OUTPUT_DIR, f"{OUTPUT_BASE_NAME}.pdf")
render_boxplot_figure(df_plot, raw_labels, wrapped, FIXED_FOLD_FOR_ANNOT, output_path,
                       label_col=NAME_CAT,
                       score_x_override=SCORE_X_OVERRIDE, bbox_color_override=TEXT_BBOX_COLOR_OVERRIDE)

# ============================================================
# MERGED ALL-CATEGORIES FIGURE (only for the "no --broad_class" run)
# ============================================================
# One long figure with every category from every broad class, grouped and
# ranked within each broad class (same "rank by mean macro F1, best first"
# rule as everywhere else). Each broad class's own header row is its rollup
# box (same n_pos_entail-weighted aggregation as the figure above), so it
# carries a real box/annotation/N like any other row, just bold. Broad
# classes themselves are sequenced alphabetically. Split into several parts
# (contiguous, group boundaries respected) so each part fits on a page.
if SETUP_NAME == "broad_class_box_plot":
    MERGED_N_PARTS = 2  # bump to 3 if a part still doesn't fit the target page
    MERGED_GROUP_SPACING_CM = 0.3  # extra vertical gap between a broad class's last
                                    # category and the next broad class's header row
    merged_extra_gap_rows = MERGED_GROUP_SPACING_CM / ROW_HEIGHT_CM

    df_cats = df_raw.copy()
    df_cats["hypothesis_label"] = df_cats["hypothesis_label"].replace(RENAME_DICT)
    df_cats["broad_class"] = df_cats["broad_class"].replace(RENAME_DICT)
    df_cats["n_pos_entail"] = df_cats["n_pos_entail"].astype(int)

    # broad-class-level rollup rows (one per (broad_class, fold)): the same
    # weighted aggregation already computed as `df` for the figure above,
    # reused here so each group's header row is a real data row, not just text
    df_broad_rollup = df.copy()
    df_broad_rollup["hypothesis_label"] = df_broad_rollup["hypothesis_label"].astype(str)
    df_cats_full = pd.concat([df_cats, df_broad_rollup], ignore_index=True)

    # broad_class group order = alphabetical (categories within a group are
    # still ranked best-to-worst by mean macro F1, as elsewhere)
    broad_class_order = sorted(df_cats["broad_class"].unique())
    # a broad class's display name doubles as the label of its own rollup row
    header_labels = set(broad_class_order)

    group_labels = {}      # broad_class -> [broad_class_name, cat1, cat2, ...]
    group_fixed_fold = {}  # broad_class -> its own best fold (int)

    for bc in broad_class_order:
        g = df_cats[df_cats["broad_class"] == bc]
        cat_means = g.groupby("hypothesis_label", observed=False)["f1_macro"].mean()
        cats_order = cat_means.sort_values(ascending=False, kind="mergesort").index.tolist()

        group_labels[bc] = [bc] + cats_order
        group_fixed_fold[bc] = best_fold_for_group(g)

    group_sizes = [len(group_labels[bc]) for bc in broad_class_order]
    part_index_groups = balanced_contiguous_split(group_sizes, MERGED_N_PARTS)

    fixed_fold_map = {
        lbl: group_fixed_fold[bc]
        for bc in broad_class_order
        for lbl in group_labels[bc]
    }

    for part_num, idxs in enumerate(part_index_groups, start=1):
        part_bcs = [broad_class_order[i] for i in idxs]
        part_raw_labels = [lbl for bc in part_bcs for lbl in group_labels[bc]]

        part_wrapped = []
        for lbl in part_raw_labels:
            if lbl in header_labels:
                # each wrapped physical line must be independently LaTeX-balanced
                # for matplotlib's usetex rendering (see step_2_roc_pr_curves.py)
                bc_lines = textwrap.wrap(lbl, wrap_w)
                part_wrapped.append("\n".join(rf"\textbf{{{line}}}" for line in bc_lines))
            else:
                part_wrapped.append(wrap_and_capitalize([lbl], wrap_w)[0])

        output_path = os.path.join(OUTPUT_DIR, f"all_categories_merged_part{part_num}.pdf")
        render_boxplot_figure(
            df_cats_full, part_raw_labels, part_wrapped, fixed_fold_map, output_path,
            label_col="hypothesis_label", header_labels=header_labels,
            lbl_width_frac=MERGED_LBL_WIDTH_FRAC,
            score_x_override=MERGED_SCORE_X_OVERRIDE, bbox_color_override=MERGED_BBOX_COLOR_OVERRIDE,
            extra_gap_rows=merged_extra_gap_rows,
        )
