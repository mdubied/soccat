"""
findings_fig1_broad_classes.py

Description:
  Horizontal dot plot comparing the median salience of broad social categories
  across French and German outlets, with IQR bars (Figure 1).

  Replicates the "Distribution broad classes" chunk of soccat_analyses_short.Rmd.

Outputs:
  figures/application_to_newspaper_corpus/broad_categories_france_germany_median_iqr.pdf

Data:
  data/annotated_corpus/broad_classes_aggregate_results.csv

Usage (from the figures/ directory):
  python findings_fig1_broad_classes.py
"""
import pandas as pd
import utils as su


DATA_PATH  = "../data/annotated_corpus/broad_classes_aggregate_results.csv"
SAVE_PATH  = "application_to_newspaper_corpus/fig1_broad_classes_median_iqr.pdf"
FIGURE_CM  = (14, 6)


def build_summary(long_df):
    df = long_df[
        long_df["pct"].notna()
        & long_df["country_group"].notna()
        & (long_df["year"] <= 2023)
    ]

    summary = (
        df.groupby(["category", "country_group"])["pct"]
        .agg(
            median_pct=lambda x: x.median(),
            q25=lambda x: x.quantile(0.25),
            q75=lambda x: x.quantile(0.75),
        )
        .reset_index()
    )

    # Order categories by average of the two country medians (ascending)
    order = (
        summary.groupby("category")["median_pct"]
        .mean()
        .sort_values()
        .index.tolist()
    )

    summary["category"] = pd.Categorical(summary["category"], categories=order, ordered=True)
    summary["country_group"] = pd.Categorical(
        summary["country_group"], categories=["French", "German"], ordered=True
    )
    return summary


def main():
    long_df = su.load_and_reshape(DATA_PATH, value_col="category")
    summary  = build_summary(long_df)

    su.plot_dot_iqr(
        df=summary,
        label_col="category",
        figure_cm=FIGURE_CM,
        save_path=SAVE_PATH,
        # title="Salience of broad categories in French and German outlets",
        subtitle="Median and interquartile range",
    )
    print(f"[saved] {SAVE_PATH}")


if __name__ == "__main__":
    main()
