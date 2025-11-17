import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable


# ============================================================
# FONT CONFIG — Political Analysis style (sans-serif)
# ============================================================

def configure_fonts():
    plt.rcParams.update({
        "text.usetex": False,
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans", "Liberation Sans"],
        "font.size": 10,
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    })


# ============================================================
# READ FILES (support list of levels)
# ============================================================

def read_fold_files(base_dir, broad_category, levels):
    """
    Read CSV files where levels can be:
       - a single string level  → "fold_*_{level}.csv"
       - a list of levels       → OR of patterns
    """
    if isinstance(levels, str):
        levels = [levels]

    files = []
    for lvl in levels:
        pattern = os.path.join(
            base_dir,
            f"{broad_category}",
            f"fold_*_{lvl}.csv"
        )
        files.extend(glob.glob(pattern))

    if not files:
        raise FileNotFoundError(f"No files found for {broad_category} with levels={levels}")
    else:
        print(f"[info] Found {len(files)} files for {broad_category} with levels={levels}")

    dfs = [pd.read_csv(f) for f in files]
    return pd.concat(dfs, ignore_index=True)


# ============================================================
# AGGREGATION
# ============================================================

def numeric_mean(df):
    numeric_df = df.apply(pd.to_numeric, errors="coerce")
    return numeric_df.mean(skipna=True).to_dict()


def map_single_level_value(value, map_level):
    """Map a single level identifier using map_level if possible."""
    return map_level.get(str(value), str(value))


def map_combined_level(values, map_level):
    """
    Convert a list of raw level values into
    a clean multi-line human-readable label.
    """
    mapped = []
    for v in values:
        v = str(v)
        if v.lower() != "nan" and v.strip() != "":
            mapped.append(map_level.get(v, v))
    return "\n".join(mapped)


def aggregate_performance(
    broad_categories,
    levels,
    base_dir="data/disaggregated_performance"
):
    """
    levels may be string or list.
    Returns nested dict:
    results[broad_category][level_entry][metric] = value
    """
    results = {}

    for bc in broad_categories:
        df_all = read_fold_files(base_dir, bc, levels)

        # Determine all level-column names that actually exist
        if isinstance(levels, str):
            level_cols = [levels] if levels in df_all.columns else []
        else:
            level_cols = [lvl for lvl in levels if lvl in df_all.columns]

        if not level_cols:
            raise ValueError(f"No level columns from {levels} found in CSV for {bc}")

        # Clean concatenation -> remove NaN and blanks
        def combine(row):
            parts = []
            for col in level_cols:
                val = row[col]
                if pd.notna(val) and str(val).strip() != "":
                    parts.append(str(val))
            return "_".join(parts)   # temporary internal identifier

        df_all["_LEVEL_RAW"] = df_all.apply(combine, axis=1)

        level_values = df_all["_LEVEL_RAW"].unique()
        bc_result = {}

        for lv in level_values:
            df_subset = df_all[df_all["_LEVEL_RAW"] == lv]

            # turn "outlet_decade_country" into multi-line label
            components = lv.split("_")
            clean_label = map_combined_level(components, map_level={})

            bc_result[clean_label] = numeric_mean(df_subset)

        results[bc] = bc_result

    return results


# ============================================================
# MATRIX
# ============================================================

def build_matrix(results, metric, rows, cols, map_rows=None, map_cols=None):

    data = {}
    for c in cols:
        col_vals = []
        for r in rows:
            val = results.get(c, {}).get(r, {}).get(metric, np.nan)
            col_vals.append(val)
        data[c] = col_vals

    df = pd.DataFrame(data, index=rows)

    # Apply mappings AFTER combination
    if map_rows:
        df.index = [map_combined_level(x.split("\n"), map_rows) for x in df.index]

    if map_cols:
        df.columns = [map_cols.get(x, x) for x in df.columns]

    return df


# ============================================================
# PLOTTING
# ============================================================

def plot_heatmap(
    df,
    metric,
    figure_cm=(20, 12),
    rotation_x=45,
    rotation_y=0,
    decimals=3,
    uniform_range=(0.25, 1.0),
    show_colorbar=True,
    save_path=None
):
    configure_fonts()

    fig_w, fig_h = figure_cm[0] / 2.54, figure_cm[1] / 2.54
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    # uniform color scale
    vmin, vmax = uniform_range
    im = ax.imshow(df.values, cmap="Oranges", aspect="auto",
                   norm=Normalize(vmin=vmin, vmax=vmax))

    # ticks
    ax.set_xticks(np.arange(df.shape[1]))
    ax.set_yticks(np.arange(df.shape[0]))

    # multi-line centered + right-aligned xticks
    ax.set_xticklabels(
        df.columns,
        rotation=rotation_x,
        ha="right",
        rotation_mode="anchor"
    )

    ax.set_yticklabels(
        df.index,
        rotation=rotation_y,
        va="center"
    )

    # write numbers in cells
    for i in range(df.shape[0]):
        for j in range(df.shape[1]):
            val = df.iloc[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.{decimals}f}",
                        ha="center", va="center", color="white")

    if show_colorbar:
        cbar = plt.colorbar(im)
        cbar.set_label("")  # empty

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")

    plt.close()


# ============================================================
# FULL PIPELINE
# ============================================================

def generate_all_heatmaps(
    broad_categories,
    levels,                    # string or list
    metrics,
    base_dir="data/disaggregated_performance",
    map_broad=None,
    map_level=None,
    columns_are="broad",       # or "level"
    figure_cm=(20, 12),
    rotation_x=45,
    rotation_y=0,
    decimals=3,
    uniform_range=(0.25, 1.0),
    show_colorbar=True
):

    results = aggregate_performance(
        broad_categories=broad_categories,
        levels=levels,
        base_dir=base_dir
    )

    # all level identifiers
    all_levels = []
    seen = set()
    for bc in broad_categories:
        for lv in results[bc].keys():
            if lv not in seen:
                seen.add(lv)
                all_levels.append(lv)

    if columns_are == "broad":
        cols = broad_categories
        rows = all_levels
        map_cols = map_broad
        map_rows = map_level
    else:
        cols = all_levels
        rows = broad_categories
        map_cols = map_level
        map_rows = map_broad

    for metric in metrics:
        df_matrix = build_matrix(
            results,
            metric=metric,
            rows=rows,
            cols=cols,
            map_rows=map_rows,
            map_cols=map_cols
        )

        # filename uses all levels provided
        level_tag = "_".join(levels) if isinstance(levels, list) else levels
        out_path = f"figures/disaggregated_performance/{level_tag}_{metric}.pdf"

        plot_heatmap(
            df_matrix,
            metric=metric,
            figure_cm=figure_cm,
            rotation_x=rotation_x,
            rotation_y=rotation_y,
            decimals=decimals,
            uniform_range=uniform_range,
            show_colorbar=show_colorbar,
            save_path=out_path
        )

        print(f"[saved] {out_path}")


# ============================================================
# MAIN EXECUTION (unchanged)
# ============================================================

broad_categories = ["age_family_status", 
                    "business_activity", 
                    "labor_market_position_entrepreneurs", 
                    "profession", "real_estate_ownership", 
                    "social_deviance", 
                    "social_roles_behavior", 
                    "socio_economic_position"]

levels = ["outlet", "decade", "country"]
metrics = ["accuracy", "precision_binary", "recall_binary", "f1_binary"]

map_broad = {
    "age_family_status": "Age and Family Status",
    "business_activity": "Business Activity",
    "labor_market_position_entrepreneurs": "Labor Market Position \n and Entrepreneurs",
    "profession": "Profession",
    "real_estate_ownership": "Real Estate Ownership",
    "social_deviance": "Social Deviance",
    "social_roles_behavior": "Social Roles and Behavior",
    "socio_economic_position": "Socio-Economic Position"
}

map_level = {
    "1990.0": "1990s",
    "2000.0": "2000s",
    "2010.0": "2010s",
    "2020.0": "2020s",
    "Liberation": "Libération",
    "Monde": "Le Monde",
    "MondeDiplo": "Le Monde \n diplomatique",
    "Welt": "Die Welt",
    "Parisien": "Le Parisien",
    "Spiegel": "Der Spiegel",
    "Zeit": "Die Zeit",
    "Figaro": "Le Figaro",
}

generate_all_heatmaps(
    broad_categories=broad_categories,
    levels=levels,
    metrics=metrics,
    map_broad=map_broad,
    map_level=map_level,
    columns_are="broad",
    figure_cm=(14, 20),
    rotation_x=30,
    uniform_range=(0.0, 1.0),
    show_colorbar=False,
    decimals=2
)
