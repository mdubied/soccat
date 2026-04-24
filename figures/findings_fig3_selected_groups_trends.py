"""
findings_fig3_selected_groups_trends.py

Description:
  2x2 faceted time-series plot showing the long-term salience of four social
  groups in French and German newspapers, with annotated historical events
  (Figure 3).

  Replicates the "Focus on single groups" chunk of soccat_analyses_short.Rmd.

Outputs:
  figures/findings/fig3_selected_groups.pdf

Data:
  data/annotated_corpus/label_summary_outlet_year_final.csv

Usage (from the figures/ directory):
  python findings_fig3_selected_groups_trends.py
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import utils as su


DATA_PATH = "../data/annotated_corpus/label_summary_outlet_year_final_v2.csv"
SAVE_PATH = "findings/fig3_selected_groups.pdf"
FIGURE_CM = (14, 14)
SHOW_OUTLET_LINES = True

TITLE_DISPLAY = {
    "People with an immigration background including immigrants":
        r"\shortstack[l]{People with an immigration\\background incl. immigrants}",
}

LABELS_FOCUS = [
    "People with an immigration background including immigrants",
    "Muslims",
    "Patients",
    "Health and care professionals",
]

PANEL_LABELS = ["A", "B", "C", "D"]

# (label, year, event_text, side, y_override, y_offset)
#   side       : "left" | "right" | "top"
#   y_override : None  → y of highest line at that year (auto)
#                float → exact data-coordinate value
#   y_offset   : float added on top of y_override/auto (default 0)
ANNOTATIONS = [
    ("Patients",                          2020, "Covid outbreak",              "left",  None, 0.0),
    ("Health and care professionals",     2020, "Covid outbreak",              "left",  None, 0.0),
    ("People with an immigration background including immigrants", 2015, "Refugee crisis",             "right", None, 0.1),
    ("People with an immigration background including immigrants", 2010, r"\shortstack[c]{Sarrazin debate\\in Germany}", "left",   None, 0.5),
    ("Muslims",                           2015, "Refugee crisis",              "right", None, 0),
    ("Muslims",                           2004, "Veil ban in France",          "right", None, 0.03),
    ("Muslims",                           2010, r"\shortstack[l]{Sarrazin debate\\in Germany}",  "right", None, -0.17),
]


def build_trend_df(long_df):
    df = long_df[
        long_df["pct"].notna()
        & long_df["country_group"].notna()
        & (long_df["year"] <= 2023)
        & (long_df["label"].isin(LABELS_FOCUS))
    ]
    trend = (
        df.groupby(["label", "country_group", "year"])["pct"]
        .mean()
        .reset_index(name="mean_pct")
    )
    outlet_trend = (
        df.groupby(["label", "outlet", "country_group", "year"])["pct"]
        .mean()
        .reset_index(name="mean_pct")
    )
    return trend, outlet_trend


def plot_trends(trend_df, outlet_df, show_outlet_lines=SHOW_OUTLET_LINES,
                figure_cm=FIGURE_CM, save_path=SAVE_PATH):
    su.configure_fonts()

    fig_w, fig_h = figure_cm[0] / 2.54, figure_cm[1] / 2.54
    fig, axes = plt.subplots(2, 2, figsize=(fig_w, fig_h))
    axes = axes.flatten()

    colors     = {"French": "#C0392B", "German": "#909090"}
    linestyles = {"French": "-",       "German": "--"}
    groups = ["French", "German"]

    # Annotation lookup: for each (label, year) -> y position of highest group
    annot_y = {}
    for label, year, _, _side, _y, _dy in ANNOTATIONS:
        sub = trend_df[(trend_df["label"] == label) & (trend_df["year"] == year)]
        if not sub.empty:
            top_row = sub.loc[sub["mean_pct"].idxmax()]
            annot_y[(label, year)] = (top_row["mean_pct"], top_row["country_group"])

    for i, label in enumerate(LABELS_FOCUS):
        ax = axes[i]
        label_df = trend_df[trend_df["label"] == label]

        if show_outlet_lines:
            outlet_label_df = outlet_df[outlet_df["label"] == label]
            for _outlet, o_df in outlet_label_df.groupby("outlet"):
                group = o_df["country_group"].iloc[0]
                o_df = o_df.sort_values("year")
                ax.plot(
                    o_df["year"], o_df["mean_pct"],
                    color=colors[group], linestyle=linestyles[group],
                    linewidth=0.5, alpha=0.3, zorder=1,
                )

        for group in groups:
            g_df = label_df[label_df["country_group"] == group].sort_values("year")
            if g_df.empty:
                continue
            ax.plot(
                g_df["year"], g_df["mean_pct"],
                color=colors[group], linestyle=linestyles[group], linewidth=1.3, label=group,
                zorder=2,
            )

        # Vertical dashed lines and text annotations for this label
        already_annotated_years = set()
        for ann_label, ann_year, ann_text, ann_side, ann_y, ann_dy in ANNOTATIONS:
            if ann_label != label:
                continue

            ax.axvline(
                x=ann_year, color="grey", linestyle=":",
                linewidth=1.0, alpha=0.8, zorder=0,
            )

            if ann_year not in already_annotated_years:
                y_auto, _ = annot_y.get((label, ann_year), (np.nan, None))
                if np.isnan(y_auto):
                    continue
                if ann_y == "top":
                    y_pos = label_df["mean_pct"].max()
                elif ann_y is not None:
                    y_pos = ann_y
                else:
                    y_pos = y_auto
                y_pos += ann_dy
                if ann_side == "top":
                    nudge_x, nudge_y = 0, 0
                    ha, va = "center", "bottom"
                elif ann_side == "left":
                    nudge_x, nudge_y = -1.0, 0
                    ha, va = "right", "center"
                else:
                    nudge_x, nudge_y = 1.0, 0
                    ha, va = "left", "center"
                ax.annotate(
                    ann_text,
                    xy=(ann_year, y_auto),
                    xytext=(ann_year + nudge_x, y_pos + nudge_y),
                    fontsize=8,
                    color="grey",
                    ha=ha, va=va,
                    bbox=dict(
                        boxstyle="round,pad=0.35",
                        fc="white", ec="grey",
                        linewidth=0.3, alpha=0.85,
                    ),
                )
                already_annotated_years.add(ann_year)

        ax.set_xlim(1995, 2025)
        ax.set_xticks(range(1995, 2024, 5))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: str(int(x))))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.2f}\\%"))
        ax.set_ylabel(r"Avg.\ share of sentences (\%)")
        ax.set_xlabel("")

        ax.grid(axis="y", color="grey", alpha=0.3, linewidth=0.3)
        ax.grid(axis="x", visible=False)
        ax.spines[["top", "right"]].set_visible(False)

        title_str = TITLE_DISPLAY.get(label, label)
        panel = PANEL_LABELS[i]
        ax.set_title("\\textbf{" + panel + ".} " + title_str, loc="left", fontsize=9, pad=4)

        ax.legend(frameon=False, fontsize=8, loc="upper left", handlelength=1.5)

    plt.tight_layout()

    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"[saved] {save_path}")


def main():
    long_df  = su.load_and_reshape(DATA_PATH, value_col="label")
    trend_df, outlet_df = build_trend_df(long_df)
    plot_trends(trend_df, outlet_df)


if __name__ == "__main__":
    main()
