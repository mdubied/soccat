"""
findings_fig2_top_categories.py

Description:
  Horizontal dot plot comparing the median salience of the top-20 social labels
  across French and German outlets, with IQR bars (Figure 2).

  Replicates the "Distribution social categories" chunk of soccat_analyses_short.Rmd.

Outputs:
  figures/findings/fig2_top20_medians_iqr.pdf

Data:
  data/annotated_corpus/label_summary_outlet_year_final.csv

Usage (from the figures/ directory):
  python findings_fig2_top_categories.py
"""
import pandas as pd
import utils as su


DATA_PATH = "../data/annotated_corpus/label_summary_outlet_year_final_v2.csv"
SAVE_PATH = "findings/fig2_top20_medians_iqr.pdf"
FIGURE_CM = (15, 14)


def build_summary(long_df):
    df = long_df[
        long_df["pct"].notna()
        & long_df["country_group"].notna()
        & (long_df["year"] <= 2023)
    ]

    # Top 20 labels by overall median across all observations
    top20 = (
        df.groupby("label")["pct"]
        .median()
        .sort_values(ascending=False)
        .head(20)
        .index.tolist()
    )

    df = df[df["label"].isin(top20)]

    summary = (
        df.groupby(["label", "country_group"])["pct"]
        .agg(
            median_pct=lambda x: x.median(),
            q25=lambda x: x.quantile(0.25),
            q75=lambda x: x.quantile(0.75),
        )
        .reset_index()
    )

    # Order labels by average of the two country medians (ascending)
    order = (
        summary.groupby("label")["median_pct"]
        .mean()
        .sort_values()
        .index.tolist()
    )

    display = {
        "People with an immigration background including immigrants":
            r"\shortstack[l]{People with an immigration\\background incl. immigrants}",
        "Offenders criminals prisoners and or accused people":
            r"\shortstack[l]{Offenders, criminals, prisoners\\and/or accused people}",
    }
    summary["label"] = summary["label"].replace(display)
    order = [display.get(l, l) for l in order]

    summary["label"] = pd.Categorical(summary["label"], categories=order, ordered=True)
    summary["country_group"] = pd.Categorical(
        summary["country_group"], categories=["French", "German"], ordered=True
    )
    return summary


def main():
    long_df = su.load_and_reshape(DATA_PATH, value_col="label")
    summary  = build_summary(long_df)

    su.plot_dot_iqr(
        df=summary,
        label_col="label",
        figure_cm=FIGURE_CM,
        save_path=SAVE_PATH,
        # title="Salience of social labels in French and German outlets",
        subtitle="Median and interquartile range, top 20 labels",
    )
    print(f"[saved] {SAVE_PATH}")


if __name__ == "__main__":
    main()
