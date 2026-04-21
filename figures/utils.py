"""
soccat_utils.py

Shared utilities for soccat annotated corpus figures:
  - Font configuration (matching step_2_heatmap.py)
  - Data loading and reshaping (wide CSV -> long format)
  - Dot-IQR plot function (used by Figures 1 and 2)

Usage:
  import soccat_utils as su
"""
import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


# ============================================================
# FONT CONFIG — same as step_2_heatmap.py
# ============================================================

def configure_fonts():
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "sans-serif",
        "font.sans-serif": ["Latin Modern Sans"],
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
# OUTLET METADATA
# ============================================================

FRENCH_OUTLETS = {"figaro", "lemonde", "liberation", "mediapart", "mondediplo"}
GERMAN_OUTLETS  = {"bild", "faz", "spiegel", "sz", "welt", "zeit"}

COUNTRY_MAP = {o: "French" for o in FRENCH_OUTLETS}
COUNTRY_MAP.update({o: "German" for o in GERMAN_OUTLETS})

IDEOLOGY_MAP = {
    "liberation": "Left-wing",
    "mediapart":  "Left-wing",
    "mondediplo": "Left-wing",
    "sz":         "Left-wing",
    "lemonde":    "Centrist",
    "spiegel":    "Centrist",
    "zeit":       "Centrist",
    "figaro":     "Conservative",
    "faz":        "Conservative",
    "welt":       "Conservative",
    "bild":       "Populist right-wing",
}


# ============================================================
# DATA LOADING
# ============================================================

def load_and_reshape(filepath, value_col, exclude_outlets=None):
    """
    Load a semicolon-delimited wide CSV and return a long-format DataFrame.

    Parameters
    ----------
    filepath : str
        Path to the CSV file.
    value_col : str
        Name of the label/category column (e.g. 'category' or 'label').
    exclude_outlets : list of str or None
        Outlet names to drop before reshaping.

    Returns
    -------
    DataFrame with columns:
        outlet, {value_col}, year (int), pct (float),
        country_group, ideology_group
    Years 2024 and 2025 are excluded.
    Underscores in {value_col} are replaced by spaces.
    """
    df = pd.read_csv(filepath, sep=";", na_values=["", "NA"])
    if exclude_outlets:
        df = df[~df["outlet"].isin(exclude_outlets)]
    df[value_col] = df[value_col].str.replace("_", " ", regex=False).str.capitalize()

    pct_cols = [
        c for c in df.columns
        if re.match(r"^(19|20)\d{2}_pct$", c)
        and not re.match(r"^202[45]_pct$", c)
    ]

    long = df.melt(
        id_vars=["outlet", value_col],
        value_vars=pct_cols,
        var_name="year_col",
        value_name="pct",
    )
    long["year"] = long["year_col"].str.extract(r"(\d{4})").astype(int)
    long = long.drop(columns=["year_col"])

    long["country_group"]  = long["outlet"].map(COUNTRY_MAP)
    long["ideology_group"] = long["outlet"].map(IDEOLOGY_MAP)

    return long


# ============================================================
# DOT-IQR PLOT  (Figures 1 and 2)
# ============================================================

def plot_dot_iqr(
    df,
    label_col,
    figure_cm=(25, 17),
    save_path=None,
    title="",
    subtitle="",
):
    """
    Horizontal dot plot with IQR line ranges, dodged by country_group.

    Parameters
    ----------
    df : DataFrame
        Must have columns: {label_col} (Categorical, ordered),
        country_group, median_pct, q25, q75.
    label_col : str
        Column that maps to the y-axis rows.
    figure_cm : tuple
        Figure size in centimetres (width, height).
    save_path : str or None
        If provided, save the figure as PDF to this path.
    title : str
    subtitle : str
    """
    configure_fonts()

    fig_w, fig_h = figure_cm[0] / 2.54, figure_cm[1] / 2.54
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    colors  = {"French": "#C0392B", "German": "#909090"}
    markers = {"French": "o",       "German": "s"}
    dodge   = 0.15
    offsets = {"French": dodge, "German": -dodge}
    groups  = ["French", "German"]

    categories  = df[label_col].cat.categories.tolist()
    cat_to_idx  = {c: i for i, c in enumerate(categories)}

    handles = []
    for group in groups:
        g_df = df[df["country_group"] == group]

        # IQR lines
        for _, row in g_df.iterrows():
            yi = cat_to_idx[row[label_col]] + offsets[group]
            ax.plot(
                [row["q25"], row["q75"]], [yi, yi],
                color=colors[group], alpha=0.7, linewidth=1.5, solid_capstyle="round",
            )

        # Median dots
        y_vals = g_df[label_col].map(cat_to_idx).astype(float) + offsets[group]
        h, = ax.plot(
            g_df["median_pct"].values, y_vals.values,
            markers[group], color=colors[group], markersize=4, zorder=3, label=group,
        )
        handles.append(h)

    ax.set_yticks(range(len(categories)))
    ax.set_yticklabels(categories)
    ax.set_xlabel(r"Median share of sentences (\%)")
    ax.set_ylabel("")
    ax.legend(handles=handles, loc="lower right", frameon=False, ncol=2, handlelength=1)

    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.1f}\\%"))
    ax.grid(axis="x", color="grey", alpha=0.25, linewidth=0.4)
    ax.grid(axis="y", visible=False)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)

    if title and subtitle:
        ax.set_title(
            "\\textbf{" + title + "}\n" + subtitle,
            loc="left", pad=8, fontsize=8,
        )
    elif title:
        ax.set_title("\\textbf{" + title + "}", loc="left", pad=8, fontsize=8)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")

    plt.close()
