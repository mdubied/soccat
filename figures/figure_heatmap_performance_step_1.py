"""
figure_heatmap_performance_step_1.py

Description:
Create the heatmap figures for disaggregated performance of the step 1 model.

Outputs:
- PDF heatmap file in "step_1/step_1_outlet_country_decade_performance.pdf".

Usage (from this directory):
python figure_heatmap_performance_step_1.py
"""
import pandas as pd
from figure_heatmap_performance import plot_heatmap

# ============================================================
# DATA REARRANGEMENT FUNCTION
# ============================================================
def extract_result_matrix(df_long, map_level, exclude_rows=None):
    """
    Expects long-format CSV with columns:
      group, Accuracy, Precision, Recall, F1

    Output:
      index   = mapped group names (order from map_level)
      columns = Accuracy, Precision, Recall, F1
    """

    metrics = ["Accuracy", "Precision", "Recall", "F1"]

    needed = {"group", *metrics}
    missing = needed - set(df_long.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    exclude_rows = set(exclude_rows or [])

    # Keep only useful columns
    d = df_long[["group"] + metrics].copy()

    # Remove excluded groups
    if exclude_rows:
        d = d[~d["group"].isin(exclude_rows)]

    # Remove duplicates (safety)
    d = d.drop_duplicates(subset="group", keep="first")

    # Set index
    d = d.set_index("group")

    # Enforce row order using map_level keys
    ordered_raw = [k for k in map_level.keys() if k in d.index]
    rest_raw = [k for k in d.index if k not in ordered_raw]

    d = d.loc[ordered_raw + rest_raw]

    # Apply pretty labels
    d.index = [map_level.get(x, x) for x in d.index]

    return d


# ============================================================
# MAIN EXECUTION
# ============================================================
def main():

    map_level = {
        "Figaro": "Le Figaro",
        "Le Monde": "Le Monde",
        "Le Monde Diplomatique": "Le Monde\n diplomatique",
        "Parisien": "Le Parisien",
        "Libération": "Libération",
        "Médiapart": "Mediapart",
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

    # Read data
    df = pd.read_csv("../data/step_1/performance_all_levels.csv")

    # Extract result matrix from dataframe
    df_matrix = extract_result_matrix(
        df_long=df,
        map_level=map_level,
        exclude_rows=["Unknown", "ALL"],
    )

    # Plot heatmap
    plot_heatmap(
        df_matrix,
        figure_cm=(14, 18),
        rotation_x=0,
        decimals=2,
        uniform_range=(0.6, 1.0),
        show_colorbar=False,
        save_path="step_1/step_1_outlet_country_decade_performance.pdf",
        separator_after_rows=["Süddeutsche\n Zeitung", "Germany"],
        anchor_xticklabels="center",
    )

if __name__ == "__main__":
    main()
