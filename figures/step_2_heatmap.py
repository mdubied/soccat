"""
step_2_heatmap.py

Description:
Create the heatmap figures for disaggregated Step 2 performance (by outlet,
country, and decade), one figure per metric, Figures A10-A13 of the paper.

Each cell is computed on the predictions POOLED over the 5 cross-validation
test folds (fold_*_human_vs_model.csv): every annotated sentence is in exactly
one test fold, so each cell uses all test sentences of that outlet/country/
decade once, instead of averaging per-fold scores computed on very small
subsets (half of the fold x outlet cells have fewer than 10 positives, and a
fold without positives would score F1 = 0). Within a cell, metrics are
computed on all (sentence, label) pairs of the broad class, i.e. pooled over
its specific group labels. Decades are derived from the publication year
(2 undated sentences are left out of the decade rows).

Metrics: binary (positive-class) precision, recall and F1, and macro F1
(unweighted mean of the positive- and negative-class F1). Accuracy is not
plotted: with 82-99.8% negative pairs it is 0.98-1.00 in every cell.

Each cell also shows N, the number of positive pairs (all 5 folds together).
Cells with fewer than MIN_N positives are greyed out.

Outputs:
- PDF heatmap files in "step_2/heatmaps" folder.
- step_2/heatmaps/outlet_country_decade_summary.txt: the cell values, plus
  the same metrics pooled over all broad classes (numbers for the text).

Usage (from this directory):
python step_2_heatmap.py
"""
import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

# ============================================================
# FONT CONFIG — Political Analysis style (sans-serif)
# ============================================================

def configure_fonts():
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
# PLOTTING (also used by step_1_heatmap.py)
# ============================================================

TEXT_LUM_THRESHOLD = 0.3   # black text on cells lighter than this

def best_text_color(rgba):
    """Black text on light cells, white on the others, based on the relative
    luminance of the cell color (sRGB, as in WCAG 2). The threshold is above
    WCAG's equal-contrast point (~0.18), which would put black text on most
    mid-orange cells; with 0.3, black is used on the light cells only
    (in the Oranges colormap: values below ~0.65 for the 0.1-1 range of the
    Step 2 heatmaps, ~0.76 for the 0.4-1 range of the Step 1 heatmap)."""
    lin = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in rgba[:3]]
    lum = 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]
    return "black" if lum > TEXT_LUM_THRESHOLD else "white"


def plot_heatmap(
    df,
    figure_cm=(20, 12),
    rotation_x=45,
    rotation_y=0,
    decimals=3,
    uniform_range=(0.25, 1.0),
    show_colorbar=True,
    save_path=None,
    separator_after_rows=None,
    anchor_xticklabels="right",
    dark_text_below=None,
    n_df=None,
    min_n=None,
    row_n=None,
):
    # dark_text_below: write values below this threshold in black instead of
    # white (fixed value threshold). None (default) = pick black or white per
    # cell from the luminance of the cell's color (see best_text_color)
    # n_df: optional matrix (same shape as df) of cell sizes, written in small
    # font below each value; cells with n < min_n are greyed out
    # row_n: optional per-row sizes (list/Series in row order), written in an
    # "N" column to the right of the heatmap (when N is the same for a whole row)
    configure_fonts()

    fig_w, fig_h = figure_cm[0] / 2.54, figure_cm[1] / 2.54
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    vmin, vmax = uniform_range
    values = df.values.astype(float)
    small = np.zeros(values.shape, dtype=bool)
    if n_df is not None and min_n is not None:
        small = n_df.values < min_n
    im = ax.imshow(np.where(small, np.nan, values), cmap="Oranges", aspect="auto",
                   norm=Normalize(vmin=vmin, vmax=vmax))
    if small.any():
        grey = np.where(small, 1.0, np.nan)
        ax.imshow(grey, cmap="Greys", vmin=0, vmax=8, aspect="auto")

    ax.set_xticks(np.arange(df.shape[1]))
    ax.set_yticks(np.arange(df.shape[0]))

    ax.set_xticklabels(
        df.columns,
        rotation=rotation_x,
        ha=anchor_xticklabels,
        rotation_mode="anchor"
    )

    ax.set_yticklabels(
        df.index,
        rotation=rotation_y,
        va="center"
    )

    # ---- DRAW SEPARATORS ----
    if separator_after_rows:
        for row_label in separator_after_rows:
            for i, label in enumerate(df.index):
                if row_label == label:
                    ax.axhline(i + 0.5, color="white", linewidth=1.5)
                    break

    # ---- WRITE NUMBERS ----
    for i in range(df.shape[0]):
        for j in range(df.shape[1]):
            val = values[i, j]
            if small[i, j]:
                color = "dimgray"
            else:
                if np.isnan(val):
                    color = "black"
                elif dark_text_below is not None:
                    color = "black" if val < dark_text_below else "white"
                else:
                    color = best_text_color(im.cmap(im.norm(val)))
            if n_df is None:
                if not np.isnan(val):
                    ax.text(j, i, f"{val:.{decimals}f}", ha="center", va="center", color=color)
                continue
            n = n_df.iloc[i, j]
            if not np.isnan(val):
                ax.text(j, i - 0.12, f"{val:.{decimals}f}", ha="center", va="center", color=color)
            ax.text(j, i + 0.24, f"N={int(n)}", ha="center", va="center", color=color, fontsize=5.5)

    if row_n is not None:
        x_n = df.shape[1] - 0.5 + 0.45
        ax.text(x_n, -0.5 - 0.15, "N", ha="center", va="bottom")
        for i, n in enumerate(row_n):
            ax.text(x_n, i, f"{int(n)}", ha="center", va="center", clip_on=False)

    if show_colorbar:
        cbar = plt.colorbar(im)
        cbar.set_label("")

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")

    plt.close()


# ============================================================
# DATA: PREDICTIONS POOLED OVER THE 5 TEST FOLDS
# ============================================================

def load_pooled_predictions(base_dir, broad_class):
    files = sorted(glob.glob(os.path.join(base_dir, broad_class, "fold_*_human_vs_model.csv")))
    if len(files) != 5:
        raise FileNotFoundError(f"Expected 5 fold_*_human_vs_model.csv for {broad_class}, found {len(files)}")
    d = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    # nli_label / pred_label: 0 = entailment (the label applies)
    d["y"] = (d["nli_label"] == 0).astype(int)
    d["p"] = (d["pred_label"] == 0).astype(int)
    # 2 sentences have no date: left out of the decade rows only
    d["all"] = "All"   # whole test data (all folds pooled)
    d["decade"] = (d["year"] // 10 * 10).map(lambda y: f"{int(y)}s" if pd.notna(y) else np.nan)
    return d


def cell_metrics(y, p):
    tp = int(((y == 1) & (p == 1)).sum())
    fp = int(((y == 0) & (p == 1)).sum())
    fn = int(((y == 1) & (p == 0)).sum())
    tn = int(((y == 0) & (p == 0)).sum())

    def f1(a, b, c):  # a = true positives of the class, b = false positives, c = false negatives
        return 2 * a / (2 * a + b + c) if (2 * a + b + c) > 0 else np.nan

    return {
        "precision_binary": tp / (tp + fp) if tp + fp > 0 else np.nan,
        "recall_binary": tp / (tp + fn) if tp + fn > 0 else np.nan,
        "f1_binary": f1(tp, fp, fn),
        "f1_macro": np.nanmean([f1(tp, fp, fn), f1(tn, fn, fp)]),
        "n_pos": tp + fn,
        "n_pairs": tp + fp + fn + tn,
    }


def compute_cells(preds, levels):
    """preds: {broad_class: dataframe}. Returns a long dataframe with one row
    per (broad_class, level, value), plus an "All broad classes" column."""
    rows = []
    groups = dict(preds)
    groups["__all__"] = pd.concat(preds.values(), ignore_index=True)
    for bc, d in groups.items():
        for level in levels:
            for value, g in d.groupby(level):
                rows.append({"broad_class": bc, "level": level, "value": str(value),
                             **cell_metrics(g["y"].values, g["p"].values)})
    return pd.DataFrame(rows)


# ============================================================
# MAIN EXECUTION
# ============================================================
def main():

    base_dir = "../data/model_performance/step_2"
    out_dir = "step_2/heatmaps"

    broad_categories = ["socio_economic_position",
                        "labor_market_position",
                        "age_family_status",
                        "identities_minority_majority_status",
                        "profession",
                        "social_roles_behavior",
                        "social_deviance",
                        "real_estate_ownership"
                        ]

    levels = ["all", "outlet", "country", "decade"]
    metrics = ["precision_binary", "recall_binary", "f1_binary", "f1_macro"]
    MIN_N = 10   # cells with fewer positive pairs are greyed out

    map_broad = {
        "socio_economic_position": "Socio-economic position",
        "labor_market_position": "Labor market position",
        "age_family_status": "Age and family status",
        "identities_minority_majority_status": "Gender, sexuality, and\nsociocultural characteristics",
        "profession": "Profession",
        "social_roles_behavior": "Social roles and behavior",
        "social_deviance": "Social deviance",
        "real_estate_ownership": "Real estate ownership",
    }

    # row order and display names
    map_level = {
        "All": "All",
        "Figaro": "Le Figaro",
        "Monde": "Le Monde",
        "MondeDiplo": "Le Monde\n diplomatique",
        "Parisien": "Le Parisien",
        "Liberation": "Libération",
        "Mediapart": "Mediapart",
        "Bild": "Bild",
        "Spiegel": "Der Spiegel",
        "Welt": "Die Welt",
        "Zeit": "Die Zeit",
        "FAZ": "Frankfurter\n Allgemeine",
        "SZ": "Süddeutsche\n Zeitung",
        "France": "France",
        "Germany": "Germany",
        "1990s": "1990s",
        "2000s": "2000s",
        "2010s": "2010s",
        "2020s": "2020s",
    }

    preds = {bc: load_pooled_predictions(base_dir, bc) for bc in broad_categories}
    cells = compute_cells(preds, levels)

    unknown = set(cells["value"]) - set(map_level)
    if unknown:
        raise ValueError(f"Values without a row in map_level: {unknown}")

    def matrix(col, broad_classes=broad_categories):
        m = (cells[cells["broad_class"].isin(broad_classes)]
             .pivot(index="value", columns="broad_class", values=col)
             .reindex(index=list(map_level), columns=broad_classes))
        m.index = [map_level[v] for v in m.index]
        m.columns = [map_broad.get(c, c) for c in m.columns]
        return m

    n_matrix = matrix("n_pos")
    for metric in metrics:
        out_path = f"{out_dir}/outlet_country_decade_{metric}.pdf"  # name kept (paper references)
        plot_heatmap(
            matrix(metric),
            figure_cm=(14, 18),
            rotation_x=40,
            uniform_range=(0.1, 1.0),
            dark_text_below=0.6,   # same value threshold as step_1_heatmap.py
            show_colorbar=False,
            decimals=2,
            save_path=out_path,
            separator_after_rows=["All", "Süddeutsche\n Zeitung", "Germany"],
            n_df=n_matrix,
            min_n=MIN_N,
        )
        print(f"[saved] {out_path}")

    # ---- summary for the text ----
    lines = ["Step 2 performance by outlet, country and decade, predictions pooled over the 5 test folds.",
             "Pairs pooled over all specific group labels of a broad class; 'All' pools all eight broad classes.",
             f"Heatmap cells with fewer than {MIN_N} positive pairs are greyed out.", ""]
    for metric in metrics + ["n_pos"]:
        m = matrix(metric, broad_categories + ["__all__"])
        m.columns = [c.replace("\n", " ") if c != "__all__" else "All" for c in m.columns]
        m.index = [i.replace("\n ", " ") for i in m.index]
        lines.append(f"=== {metric} ===")
        lines.append(m.to_string(float_format=lambda v: f"{v:.2f}" if metric != "n_pos" else f"{v:.0f}"))
        lines.append("")
    summary_path = f"{out_dir}/outlet_country_decade_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[saved] {summary_path}")


if __name__ == "__main__":
    main()
