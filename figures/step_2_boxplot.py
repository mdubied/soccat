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

(or other broad class name to get all categories within this broad class, or no argument for all broad classes in one merged figure)
"""
import os, re, glob, json, textwrap, itertools
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
    SETUP_NAME = "broad_class_boxplot"
else:
    SETUP_NAME = "cat_per_broad_class_boxplot"

# Setup-specific paths
STEP_2_DATA_DIR = "../data/model_performance/step_2"

# Each broad class has its own folder directly under STEP_2_DATA_DIR
# (containing per-fold, per-label CSVs); some were renamed in the process.
FOLDER_NAME_MAP = {
    "age_family": "age_family_status",
    "identity": "identities_minority_majority_status",
    "labor_market_position": "labor_market_position",
    "profession": "profession",
    "real_estate": "real_estate_ownership",
    "social_deviance": "social_deviance",
    "social_roles": "social_roles_behavior",
    "socio_economic": "socio_economic_position",
}

if SETUP_NAME == "cat_per_broad_class_boxplot":
    BROAD_CLASS = args.broad_class
    OUTPUT_DIR   = f"step_2/boxplots/cat_per_broad_class_boxplot"
    OUTPUT_BASE_NAME = f"{BROAD_CLASS}_boxplot"
    NAME_CAT = "hypothesis_label"
    LBL_WIDTH_FRAC  = 0.29   # Fixed fraction of figure width for labels (instead of measuring)
elif SETUP_NAME == "broad_class_boxplot":
    OUTPUT_DIR   = "step_2/boxplots"
    OUTPUT_BASE_NAME = "step_2_perf_broad_class_boxplot"
    NAME_CAT = "hypothesis_label"
    LBL_WIDTH_FRAC  = 0.24   # Fixed fraction of figure width for labels (instead of measuring)
    MERGED_LBL_WIDTH_FRAC = 0.27  # merged all-categories figure needs the wider margin (long category names)
else:
    raise ValueError(f"Unknown SETUP_NAME: {SETUP_NAME}")

# Performance metrics to plot
METRICS      = ["precision_binary", "recall_binary", "f1_binary", "f1_macro"]
METRIC_TITLE_MAP = {
    "precision_binary": "Binary Precision",
    "recall_binary": "Binary Recall",
    "f1_binary": "Binary F1",
    "f1_macro": "Macro F1",
}

# Broad category listing
BROAD_CLASS_LIST = [
    "age_family",
    "identity",
    "labor_market_position",
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
    "labor_market_position": "Labor market position",
    "age_family": "Age and family status",
    "identity": "Identities and minority/ majority status",
    "profession": "Profession",
    "social_roles": "Social roles and behavior",
    "social_deviance": "Social deviance",
    "real_estate": "Real estate ownership",
}

# display name -> broad-class code (for looking up fold_*_metrics.json via
# FOLDER_NAME_MAP after a broad class's "hypothesis_label" column has already
# been through RENAME_DICT)
BROAD_CLASS_DISPLAY_TO_CODE = {RENAME_DICT.get(bc, bc): bc for bc in BROAD_CLASS_LIST}

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
        ("recall_binary", "Ethnic and racial minorities"): 0.38,
        ("f1_binary", "Ethnic and racial minorities"): 0.44,
    },
    "labor_market": {
        ("precision_binary", "employers"): 0.38,
        ("precision_binary", "unemployed"): 0.38,
        ("precision_binary", "civil servants"): 0.01,
        ("precision_binary", "CEOs and corporate leaders"): 0.72,
        ("precision_binary", "retirees"): 0.64,
        ("recall_binary", "self-employed and freelancers"): 0.00,
        ("recall_binary", "retirees"): 0.64,
        ("f1_binary", "retirees"): 0.64,
    },
    "age_family": {
        ("recall_binary", "elderly"): 0.02,
        ("recall_binary", "middle-aged and pre-retirement age groups"): 0.58,
    },
    "socio_economic_position": {
        ("recall_binary", "middle class"): 0.0,
        ("recall_binary", "lower class"): 0.0,
    },
    "social_deviance": {
        ("recall_binary", "terrorists, revolutionaries, rebels, armed resistance"): 0.0,
    },
}
CATEGORY_BBOX_COLOR_OVERRIDES = {
    "identity": {
        ("recall_binary", "Ethnic and racial minorities"): "lightgray",
        ("f1_binary", "Ethnic and racial minorities"): "lightgray",
                
    },
    "age_family": {
        ("precision_binary", "middle-aged and pre-retirement age groups"): "lightgray",
        ("f1_binary", "middle-aged and pre-retirement age groups"): "lightgray",
    },
    "labor_market": {
        ("precision_binary", "employers"): "none",
        ("precision_binary", "civil servants"): "none",
        ("precision_binary", "unemployed"): "none",
        ("precision_binary", "CEOs and corporate leaders"): "none",
        ("recall_binary", "self-employed and freelancers"): "none",
        ("recall_binary", "retirees"): "none",
    },
    "socio_economic_position": {
        ("precision_binary", "upper class"): "none",
        ("recall_binary", "middle class"): "none",
        ("recall_binary", "lower class"): "none",
    },
    "social_deviance": {
        ("recall_binary", "terrorists, revolutionaries, rebels, armed resistance"): "none",
    },
}
# broad-class-LEVEL overrides, for the aggregated (8-broad-class) figure only
AGGREGATE_SCORE_X_OVERRIDE = {}
AGGREGATE_BBOX_COLOR_OVERRIDE = {}

if SETUP_NAME == "cat_per_broad_class_boxplot":
    SCORE_X_OVERRIDE = CATEGORY_SCORE_X_OVERRIDES.get(args.broad_class, {})
    TEXT_BBOX_COLOR_OVERRIDE = CATEGORY_BBOX_COLOR_OVERRIDES.get(args.broad_class, {})
else:  # broad_class_boxplot: figure 1 shows aggregated broad-class rows
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
ROW_HEIGHT_CM   = 0.45    # Vertical height per row (cm)
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
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
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

def load_broad_class_fold_metrics(broad_class):
    """
    Load fold_X_metrics.json for a broad class -> {fold: {metric: value}}.
    These come straight from the training notebook's compute_all_metrics(),
    called ONCE per fold on every label's (sentence, hypothesis) pairs
    concatenated together -- i.e. a single POOLED confusion matrix across all
    labels in that broad class/fold, not an average of each label's own
    metrics. This can diverge from our fold_*_per_label.csv-based
    n_pos_entail-weighted average of per-label f1_macro (see
    best_fold_for_group), especially under heavy class imbalance -- both are
    legitimate, just different aggregations (pooled/micro vs. weighted
    average across categories).
    """
    folder = FOLDER_NAME_MAP.get(broad_class, broad_class)
    metrics_dir = f"{STEP_2_DATA_DIR}/{folder}"
    fold_files = sorted(
        glob.glob(f"{metrics_dir}/fold_*_metrics.json"),
        key=lambda f: int(re.search(r"fold_(\d+)_metrics", f).group(1))
    )
    out = {}
    for f in fold_files:
        fold_num = int(re.search(r"fold_(\d+)_metrics", f).group(1))
        with open(f, encoding="utf-8") as fh:
            out[fold_num] = json.load(fh)
    return out

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
                bg_style = bbox_color_override.get(
                    (metric, str(lbl)),
                    TEXT_BBOX_COLOR_DEFAULT
                )
                # bg_style is either a plain color (fully opaque, as before) or
                # a (color, alpha) tuple for a semi-transparent box, e.g.
                # ("white", 0.5) for a half-transparent white background.
                bg_color, bg_alpha = bg_style if isinstance(bg_style, tuple) else (bg_style, 1.0)
                ax.text(x_pos, yi, s,
                        ha="left", va="center",
                        fontsize=7,
                        bbox=dict(
                        facecolor=bg_color,
                        edgecolor="none",
                        alpha=bg_alpha,
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
if SETUP_NAME == "broad_class_boxplot":
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

# Choose ONE fold for all rows (only for cat_per_broad_class_boxplot)
FIXED_FOLD_FOR_ANNOT = None
if SETUP_NAME == "cat_per_broad_class_boxplot":
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

# Print F1 median / mean / best-fold per row (binary and macro), and which
# fold that best-fold value came from (Bin.F1 best and Macro F1 best are read
# off the SAME selected row, so they share one fold number).
#
# For the broad_class_boxplot table, each row IS one broad class, so we can
# also show that broad class's POOLED macro/binary F1 at the same fold,
# straight from the training notebook's own fold_*_metrics.json (one pooled
# confusion matrix across every label's pairs in that fold -- see
# load_broad_class_fold_metrics). This is a genuinely different aggregation
# from "Macro F1 best" (which is our n_pos_entail-weighted average of each
# label's own macro F1) and the two can diverge meaningfully under heavy
# class imbalance -- shown side by side for comparison, not as a discrepancy
# to reconcile.
show_pooled = SETUP_NAME == "broad_class_boxplot"
if show_pooled:
    pooled_metrics_by_code = {
        BROAD_CLASS_DISPLAY_TO_CODE[lbl]: load_broad_class_fold_metrics(BROAD_CLASS_DISPLAY_TO_CODE[lbl])
        for lbl in raw_labels
    }

summary_lines = []
header = f"{'Label':<60} {'Bin.F1 med':>11} {'Bin.F1 mean':>12} {'Bin.F1 best':>12} {'Macro F1 best':>15} {'Fold':>6}"
if show_pooled:
    header += f" {'Pooled Bin.F1':>14} {'Pooled Macro F1':>16}"
summary_lines.append(header)
summary_lines.append("-" * len(header))
medians, means, bests_bin = [], [], []
for lbl in raw_labels:
    sub = df_plot.loc[df_plot[NAME_CAT] == lbl, ["fold"] + METRICS]
    median = sub["f1_binary"].median()
    mean = sub["f1_binary"].mean()
    best_row = select_best_fold_row(sub, FIXED_FOLD_FOR_ANNOT)
    best_bin = best_row["f1_binary"] if best_row is not None else np.nan
    best_macro = best_row["f1_macro"] if best_row is not None else np.nan
    best_fold_num = int(best_row["fold"]) if best_row is not None else -1
    medians.append(median)
    means.append(mean)
    bests_bin.append(best_bin)
    line = (
        f"{str(lbl):<60} {median:>11.5f} {mean:>12.5f} {best_bin:>12.5f} {best_macro:>15.5f} {best_fold_num:>6}"
    )
    if show_pooled:
        fold_metrics = pooled_metrics_by_code[BROAD_CLASS_DISPLAY_TO_CODE[lbl]].get(best_fold_num, {})
        pooled_bin = fold_metrics.get("f1_binary", np.nan)
        pooled_macro = fold_metrics.get("f1_macro", np.nan)
        line += f" {pooled_bin:>14.5f} {pooled_macro:>16.5f}"
    summary_lines.append(line)

if not show_pooled:
    # per-category table: one shared fold across every row (FIXED_FOLD_FOR_ANNOT)
    # -- add the whole broad class's own pooled metrics at that fold as a
    # single reference line, rather than repeating it on every category row
    pooled = load_broad_class_fold_metrics(BROAD_CLASS).get(FIXED_FOLD_FOR_ANNOT, {})
    summary_lines.append("-" * len(header))
    summary_lines.append(
        f"Whole-class pooled metrics at fold {FIXED_FOLD_FOR_ANNOT} (from fold_{FIXED_FOLD_FOR_ANNOT}_metrics.json): "
        f"f1_binary={pooled.get('f1_binary', float('nan')):.5f}, f1_macro={pooled.get('f1_macro', float('nan')):.5f}"
    )

# Count how many INDIVIDUAL (specific) categories clear each binary-F1
# threshold, for median / mean / best-fold -- always at the specific-label
# level (~57 labels across all 8 broad classes), never the 8 broad-class
# rollups themselves, even when this is the broad_class_boxplot (aggregated)
# table above. In that mode we recompute straight from df_raw (per-label,
# pre-rollup data), reading each category's "best" fold as its own broad
# class's selected fold (weighted macro F1 across that class's categories --
# same rule as best_fold_for_group / the per-category tables), since that's
# the fold whose model is actually published for every category in the class.
if SETUP_NAME == "broad_class_boxplot":
    cat_fixed_fold = {
        bc: best_fold_for_group(df_raw[df_raw["broad_class"] == bc])
        for bc in df_raw["broad_class"].unique()
    }
    cat_medians, cat_means, cat_bests = [], [], []
    for cat_lbl, g in df_raw.groupby("hypothesis_label", observed=False):
        bc = g["broad_class"].iloc[0]
        cat_medians.append(g["f1_binary"].median())
        cat_means.append(g["f1_binary"].mean())
        best_row = select_best_fold_row(g[["fold"] + METRICS], cat_fixed_fold[bc])
        cat_bests.append(best_row["f1_binary"] if best_row is not None else np.nan)
    n_categories = len(cat_medians)
else:
    cat_medians, cat_means, cat_bests = medians, means, bests_bin
    n_categories = len(raw_labels)

THRESHOLDS = [0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5]
medians_arr = np.array(cat_medians, dtype=float)
means_arr = np.array(cat_means, dtype=float)
bests_arr = np.array(cat_bests, dtype=float)

summary_lines.append("-" * len(header))
summary_lines.append(f"Count of individual categories with binary F1 > threshold (out of {n_categories} total):")
count_header = f"{'Threshold':>10} {'Median>thr':>11} {'Mean>thr':>9} {'Best>thr':>9}"
summary_lines.append(count_header)
summary_lines.append("-" * len(count_header))
for thr in THRESHOLDS:
    n_med = int(np.nansum(medians_arr > thr))
    n_mean = int(np.nansum(means_arr > thr))
    n_best = int(np.nansum(bests_arr > thr))
    summary_lines.append(f"{thr:>10.2f} {n_med:>11d} {n_mean:>9d} {n_best:>9d}")

summary_text = "\n" + "\n".join(summary_lines) + "\n"
print(summary_text)

summary_path = os.path.join(OUTPUT_DIR, f"{OUTPUT_BASE_NAME}_summary.txt")
with open(summary_path, "w", encoding="utf-8") as f:
    f.write("\n".join(summary_lines) + "\n")
print(f"Saved file as: {summary_path}")

# Negative-class size per INDIVIDUAL category (never the broad-class rollups
# -- same convention as the threshold-count table above): every fold's test
# set is the same total size (n_pairs) regardless of category, so the
# negative class is n_pairs - n_pos_entail, averaged across the 5 folds.
# In broad_class_boxplot mode this is read from df_raw (per-label, pre-rollup
# data for all 8 classes); in cat_per_broad_class_boxplot mode df_plot is
# already per-label for the one selected broad class.
if SETUP_NAME == "broad_class_boxplot":
    neg_source = df_raw.copy()
    label_col_neg = "hypothesis_label"
    neg_source[label_col_neg] = neg_source[label_col_neg].replace(RENAME_DICT)
else:
    neg_source = df_plot
    label_col_neg = NAME_CAT

neg_stats = (
    neg_source.groupby(label_col_neg, observed=False)
    .agg(n_pairs=("n_pairs", "mean"), n_pos=("n_pos_entail", "mean"))
    .reset_index()
)
neg_stats["n_neg"] = neg_stats["n_pairs"] - neg_stats["n_pos"]
neg_stats["pct_neg"] = 100 * neg_stats["n_neg"] / neg_stats["n_pairs"]
neg_stats = neg_stats.sort_values("pct_neg", kind="mergesort").reset_index(drop=True)

neg_lines = []
neg_header = f"{'Category':<60} {'n_pairs':>9} {'n_pos':>8} {'n_neg':>9} {'%neg':>8}"
neg_lines.append(neg_header)
neg_lines.append("-" * len(neg_header))
for _, r in neg_stats.iterrows():
    neg_lines.append(
        f"{str(r[label_col_neg]):<60} {r['n_pairs']:>9.1f} {r['n_pos']:>8.1f} {r['n_neg']:>9.1f} {r['pct_neg']:>7.2f}%"
    )
neg_lines.append("-" * len(neg_header))
neg_lines.append(
    f"Average %neg across {len(neg_stats)} individual categories: {neg_stats['pct_neg'].mean():.2f}%"
)
neg_lines.append(
    f"Median  %neg across {len(neg_stats)} individual categories: {neg_stats['pct_neg'].median():.2f}%"
)

neg_text = "\n" + "\n".join(neg_lines) + "\n"
print(neg_text)

neg_path = os.path.join(OUTPUT_DIR, f"{OUTPUT_BASE_NAME}_neg_class_pct.txt")
with open(neg_path, "w", encoding="utf-8") as f:
    f.write("\n".join(neg_lines) + "\n")
print(f"Saved file as: {neg_path}")

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
# classes themselves are sequenced by the same best-to-worst rule (mean
# macro F1), matching the aggregated figure above -- one consistent ranking
# throughout, at both the group and category level. Split into several
# parts (contiguous, group boundaries respected) so each part fits on a page.
if SETUP_NAME == "broad_class_boxplot":
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

    # broad_class group order = same best-to-worst order as the aggregated
    # figure above (labels_order, by mean macro F1), so both figures read as
    # one consistent ranking and categories within a group follow the same
    # rule as their group (best-to-worst by mean macro F1 throughout)
    broad_class_order = [bc for bc in labels_order if bc in set(df_cats["broad_class"])]
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

        output_path = os.path.join(OUTPUT_DIR, f"step_2_perf_boxplot_part{part_num}.pdf")
        render_boxplot_figure(
            df_cats_full, part_raw_labels, part_wrapped, fixed_fold_map, output_path,
            label_col="hypothesis_label", header_labels=header_labels,
            lbl_width_frac=MERGED_LBL_WIDTH_FRAC,
            score_x_override=MERGED_SCORE_X_OVERRIDE, bbox_color_override=MERGED_BBOX_COLOR_OVERRIDE,
            extra_gap_rows=merged_extra_gap_rows,
        )
