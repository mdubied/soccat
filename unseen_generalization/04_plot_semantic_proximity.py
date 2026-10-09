"""
04_plot_semantic_proximity.py

Description:
  Recall of SOCCAT, the LLM and the dictionary baseline as a function of the
  semantic proximity of a test mention to its nearest training mention of
  the same category (panel A), optionally with examples of mentions per
  proximity bin (panel B, SHOW_EXAMPLES; off by default: its text is too
  small in print). Appendix figure of the unseen-mention robustness check.

  Panel A: the non-seen test pairs (partial overlap + no word overlap) are split into
  equal-size proximity bins (Q1 = farthest); seen pairs are shown separately
  as a reference. Error bars are Wilson 95% confidence intervals.
  Panel B: per bin, test mention -> nearest training mention (cosine
  similarity); the dot shows whether SOCCAT detected it (filled) or not
  (hollow). Examples are picked automatically (short mentions, close to the
  bin median, distinct categories) unless listed in EXAMPLE_OVERRIDES.

  One figure per variant of 03_semantic_proximity.py (VARIANTS): pairs of
  interest = partial overlap + no word overlap (main), partial overlap only,
  or no word overlap only; all categories, or only those with SOCCAT median
  F1 >= 0.70 (suffix f1med70). The third tick line shows the bin size N (set
  a variant to "share" in VARIANTS to show the share of no-overlap pairs
  instead). A pooled tick (all of Q1-Q5 together) and the seen reference
  stand right of the bins.

  Plus one combined figure per category filter (COMBINED), at the standard
  width: recall by proximity quintile for partial overlap and for no word
  overlap (Q1-Q5 only, N in the panel titles; the similarity ranges are in
  the single-panel figures), then a narrower panel with the pooled recall of
  both groups and the seen reference side by side.

Outputs (unseen_generalization/output/):
  semantic_proximity.pdf                     main variant, all categories
  semantic_proximity_partial.pdf             partial overlap only
  semantic_proximity_no_overlap.pdf          no word overlap only
  semantic_proximity_f1med70.pdf             main variant, median F1 >= 0.70
  semantic_proximity_partial_f1med70.pdf
  semantic_proximity_no_overlap_f1med70.pdf
  semantic_proximity_partial_no_overlap[_f1med70].pdf   combined, side by side

Data:
  unseen_generalization/output/recall_by_proximity_bin[_<variant>].csv
  unseen_generalization/output/proximity_pairs.csv
  (both from 03_semantic_proximity.py)

Usage (from anywhere):
  python 04_plot_semantic_proximity.py
"""
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "figures"))
import utils as su  # noqa: E402


OUTPUT_DIR = ROOT / "output"
PAIRS_PATH = OUTPUT_DIR / "proximity_pairs.csv"
# file suffix (as written by 03) -> (third tick line: "share" of no-overlap pairs or
# bin size "n", name of the pooled tick = the pairs of interest of that variant)
VARIANTS = {
    "": ("n", "Not seen"),
    "partial": ("n", "Partial overlap"),
    "no_overlap": ("n", "No word overlap"),
    "f1med70": ("n", "Not seen"),
    "partial_f1med70": ("n", "Partial overlap"),
    "no_overlap_f1med70": ("n", "No word overlap"),
}
# combined figures: output suffix -> [(variant, panel title), ...], left to right
COMBINED = {
    "": [("no_overlap", "No word overlap"), ("partial", "Partial overlap")],
    "f1med70": [("no_overlap_f1med70", "No word overlap"), ("partial_f1med70", "Partial overlap")],
}
REFERENCES = ["pooled", "seen"]  # reference ticks, right of the proximity bins
FIGURE_CM = (17, 8.5)         # with the examples panel
FIGURE_CM_SINGLE = (14, 7.5)  # recall panel only; 14 cm wide as the findings_* figures
TICK_FONTSIZE = 8             # as the findings_* figures (figures/utils.py)
WIDTH_RATIOS = (1, 1.1)
SHOW_EXAMPLES = False         # panel B text is ~6 pt at full width: too small for print
SHOW_SHARE_UNSEEN = True      # in panel B headers, or in the tick labels without panel B

MODELS = ["SOCCAT", "LLM", "Dictionary"]
COLORS = {"SOCCAT": "#C0392B", "LLM": "#2F6DB5", "Dictionary": "#909090"}
LINESTYLES = {"SOCCAT": "-", "LLM": "--", "Dictionary": ":"}
MARKERS = {"SOCCAT": "o", "LLM": "s", "Dictionary": "^"}
DODGE = 0.12          # horizontal offset between models, so error bars don't overlap
SEEN_GAP = 0.6        # extra space between the last bin and the seen reference

N_EXAMPLES = 2        # examples per bin in panel B
MAX_EXAMPLE_CHARS = 22
EXAMPLE_FONTSIZE = 6.3
# bin -> list of (sentence_id, label) to show instead of the automatic pick
EXAMPLE_OVERRIDES = {
    # "Q1": [(1234, "elderly")],
}

CATEGORY_SHORT = {
    "people with an immigration background, including immigrants": "immigrants",
    "minors, including children and pupils": "minors",
    "youth, including students and apprentices": "youth",
    "offenders, criminals, prisoners and/or accused people": "offenders",
    "terrorists, rebels, revolutionaries and/or movements of armed resistance": "armed groups",
    "politicians and high-ranking officials": "politicians",
    "capital owners, investors and shareholders": "investors",
    "multiple (or other) religious or minority groups": "other religious/minority",
    "middle-aged and pre-retirement age groups": "middle-aged",
    "health and care professionals": "health professionals",
    "ethnic and racial minorities": "ethnic minorities",
    "parents and families": "families",
    "consumers and clients": "consumers",
    "wage and salary earners": "wage earners",
    "self-employed and freelancers": "self-employed",
    "ceos and corporate leaders": "CEOs",
    "scientists and professors": "scientists",
    "teachers and educators": "teachers",
    "farmers and fishermen": "farmers",
    "authors and artists": "artists",
}


def latex_escape(s):
    for a, b in [("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("$", r"\$"),
                 ("#", r"\#"), ("_", r"\_"), ("{", r"\{"), ("}", r"\}"),
                 ("~", r"\textasciitilde{}"), ("^", r"\textasciicircum{}")]:
        s = s.replace(a, b)
    return s


def bin_order(recall_df):
    present = list(recall_df["bin"].drop_duplicates())
    return [b for b in present if b not in REFERENCES] + [b for b in REFERENCES if b in present]


def x_positions(bins):
    """Proximity bins at 0, 1, ...; the references after an extra SEEN_GAP."""
    n_q = sum(b not in REFERENCES for b in bins)
    return {b: (i if i < n_q else i + SEEN_GAP) for i, b in enumerate(bins)}


def short_num(x):
    return f"{x:.2f}".lstrip("0")


def tick_lines(compact=False):
    if compact:
        return 1
    return 3 if SHOW_SHARE_UNSEEN and not SHOW_EXAMPLES else 2


def tick_label(row, third_line="share", pooled_name="Not seen", compact=False):
    if compact:  # bin name only (combined figure: narrow panels; N in the panel title)
        name = {"pooled": "Pooled", "seen": "Seen"}.get(row["bin"], row["bin"])
        return r"\shortstack{\strut " + name + "}"
    if row["bin"] == "pooled":
        lines = [pooled_name, "(Q1--5 pooled)"]
        if tick_lines() >= 3:
            lines.append(f"{row['share_no_overlap'] * 100:.0f}\\% no overlap" if third_line == "share"
                         else f"N = {row['n']}")
    elif row["bin"] == "seen":
        lines = ["Seen", "(ref.)"]
        if third_line == "n" and tick_lines() >= 3:
            lines.append(f"N = {row['n']}")
    else:
        lines = [row["bin"], f"{short_num(row['prox_min'])}--{short_num(row['prox_max'])}"]
        if tick_lines() >= 3:
            lines.append(f"{row['share_no_overlap'] * 100:.0f}\\% no overlap" if third_line == "share"
                         else f"N = {row['n']}")
    lines += [""] * (tick_lines() - len(lines))  # same line count -> first lines aligned
    # \strut gives every line the same height and depth (equal line spacing
    # whatever the characters); vertical alignment is fixed in plot_recall
    return r"\shortstack{" + r"\\".join(r"\strut " + line for line in lines) + "}"


def bin_header(row):
    """Panel B header: bin name + proximity range, N and share of unseen pairs."""
    if row["bin"] == "seen":
        return rf"\textbf{{Seen}} (N = {row['n']})"
    details = [f"{short_num(row['prox_min'])}--{short_num(row['prox_max'])}", f"N = {row['n']}"]
    if SHOW_SHARE_UNSEEN:
        details.append(f"{row['share_no_overlap'] * 100:.0f}\\% no overlap")
    return rf"\textbf{{{row['bin']}}} (" + ", ".join(details) + ")"


def pick_examples(pairs_df, b):
    sub = pairs_df[pairs_df["bin"] == b].copy()
    if b in EXAMPLE_OVERRIDES:
        keys = EXAMPLE_OVERRIDES[b]
        return pd.concat([sub[(sub["sentence_id"] == s) & (sub["label"] == l)] for s, l in keys])
    sub = sub[(sub["test_mention"].str.len() <= MAX_EXAMPLE_CHARS)
              & (sub["nearest_train_mention"].str.len() <= MAX_EXAMPLE_CHARS)
              & (sub["test_mention"].str.lower() != sub["nearest_train_mention"].str.lower())
              & ~sub["test_mention"].str.endswith("-") & ~sub["nearest_train_mention"].str.endswith("-")]
    sub["dist"] = (sub["proximity"] - sub["proximity"].median()).abs()
    picked, labels, countries = [], set(), set()
    for _, r in sub.sort_values(["dist", "sentence_id"]).iterrows():
        # distinct categories; alternate languages when possible
        if r["label"] in labels or (len(picked) == 1 and r["country"] in countries and
                                    (sub["country"] != r["country"]).any()):
            continue
        picked.append(r)
        labels.add(r["label"])
        countries.add(r["country"])
        if len(picked) == N_EXAMPLES:
            break
    return pd.DataFrame(picked)


def plot_recall(ax, recall_df, third_line="share", pooled_name="Not seen", compact=False, legend=True):
    bins = bin_order(recall_df)
    xs = x_positions(bins)
    n_q = sum(b not in REFERENCES for b in bins)
    for j, model in enumerate(m for m in MODELS if m in set(recall_df["model"])):
        d = recall_df[recall_df["model"] == model].set_index("bin").loc[bins]
        offset = (j - 1) * DODGE
        x = [xs[b] + offset for b in bins]
        # line through the proximity bins only; the references stand apart
        ax.plot(x[:n_q], d["recall"].iloc[:n_q], color=COLORS[model], linestyle=LINESTYLES[model],
                linewidth=1.3, zorder=2)
        ax.vlines(x, d["ci_low"], d["ci_high"], color=COLORS[model], linewidth=0.8, zorder=2)
        ax.plot(x, d["recall"], linestyle="none", marker=MARKERS[model], markersize=4.5,
                color=COLORS[model], markeredgecolor="white", markeredgewidth=0.6, zorder=3)

    ax.axvline(xs[bins[n_q - 1]] + (1 + SEEN_GAP) / 2, color="grey", linestyle=":", linewidth=1.0, alpha=0.8, zorder=0)
    rows = recall_df.drop_duplicates("bin").set_index("bin").loc[bins].reset_index()
    ax.set_xticks([xs[b] for b in bins])
    ax.set_xticklabels([tick_label(r, third_line, pooled_name, compact) for _, r in rows.iterrows()],
                       fontsize=TICK_FONTSIZE, verticalalignment="baseline")
    # baseline alignment: "top" uses the ink extent, which differs for "(ref.)"
    ax.tick_params(axis="x", pad=4 + 9.7 * tick_lines(compact))  # ~line height at TICK_FONTSIZE
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.1f}"))
    ax.set_xlim(xs[bins[0]] - 0.5, xs["seen"] + 0.5)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Recall")
    ax.set_xlabel(r"Similarity (far $\rightarrow$ close)" if compact
                  else r"Similarity to the nearest training mention (far $\rightarrow$ close)")
    # centre the axis title under the proximity bins only (it does not describe
    # the reference ticks); set_x keeps matplotlib's automatic vertical position
    x0, x1 = ax.get_xlim()
    ax.xaxis.label.set_x(((xs[bins[0]] + xs[bins[n_q - 1]]) / 2 - x0) / (x1 - x0))
    ax.grid(axis="y", color="grey", alpha=0.3, linewidth=0.3)
    ax.grid(axis="x", visible=False)
    ax.spines[["top", "right"]].set_visible(False)
    handles = [Line2D([], [], color=COLORS[m], linestyle=LINESTYLES[m], marker=MARKERS[m], markersize=4.5,
                      markeredgecolor="white", markeredgewidth=0.6, linewidth=1.3, label=m)
               for m in MODELS if m in set(recall_df["model"])]
    # above the axes, in one row: inside, it covers low reference markers
    # (e.g. SOCCAT/dictionary on no-overlap pairs, near 0.2/0.0 at the bottom right)
    if legend:
        ax.legend(handles=handles, frameon=False, fontsize=8, loc="lower center",
                  bbox_to_anchor=(0.5, 1.0), ncol=len(handles), handlelength=2.5)
    return handles
    if SHOW_EXAMPLES:  # a single panel needs no title: the LaTeX caption names it
        ax.set_title(r"\textbf{A.} Recall by semantic proximity", loc="left", fontsize=9, pad=4)


def plot_examples(ax, pairs_df, recall_df):
    bins = [b for b in bin_order(recall_df) if b != "pooled"]  # pooled = all bins, no own examples
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(r"\textbf{B.} Test mention $\rightarrow$ nearest training mention",
                 loc="left", fontsize=9, pad=4)
    headers = recall_df.drop_duplicates("bin").set_index("bin")
    row_h = 1 / (len(bins) * (N_EXAMPLES + 1.2) + 1)

    # key for the SOCCAT hit/miss dots, in place of a legend
    y = 1 - row_h * 0.4
    for x, filled, text in [(0.03, True, "detected by SOCCAT"), (0.45, False, "missed by SOCCAT")]:
        ax.plot(x, y, marker="o", markersize=3.5, color=COLORS["SOCCAT"],
                markerfacecolor=COLORS["SOCCAT"] if filled else "white", markeredgewidth=0.8,
                transform=ax.transAxes, clip_on=False)
        ax.text(x + 0.04, y, text, fontsize=EXAMPLE_FONTSIZE, color="grey",
                va="center", transform=ax.transAxes)

    y = 1 - row_h * 1.6
    for b in bins:
        ax.text(0, y, bin_header({**headers.loc[b].to_dict(), "bin": b}), fontsize=7, va="center",
                transform=ax.transAxes)
        y -= row_h
        for _, r in pick_examples(pairs_df, b).iterrows():
            ax.plot(0.03, y, marker="o", markersize=3.5, color=COLORS["SOCCAT"],
                    markerfacecolor=COLORS["SOCCAT"] if r["soccat_hit"] else "white",
                    markeredgewidth=0.8, transform=ax.transAxes, clip_on=False)
            category = CATEGORY_SHORT.get(r["label"].lower(), r["label"])
            text = (r"\textit{" + latex_escape(r["test_mention"]) + r"} $\rightarrow$ \textit{" +
                    latex_escape(r["nearest_train_mention"]) + "}" +
                    rf" ({r['proximity']:.2f}, {latex_escape(category)})")
            ax.text(0.07, y, text, fontsize=EXAMPLE_FONTSIZE, va="center", transform=ax.transAxes)
            y -= row_h
        y -= row_h * 0.2


def plot_proximity(recall_df, pairs_df, save_path, third_line="share", pooled_name="Not seen", total_n=None):
    su.configure_fonts()

    figure_cm = FIGURE_CM if SHOW_EXAMPLES else FIGURE_CM_SINGLE
    fig_w, fig_h = figure_cm[0] / 2.54, figure_cm[1] / 2.54
    if SHOW_EXAMPLES:
        fig, (ax, ax_ex) = plt.subplots(1, 2, figsize=(fig_w, fig_h),
                                        gridspec_kw={"width_ratios": WIDTH_RATIOS})
        plot_examples(ax_ex, pairs_df, recall_df)
    else:
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    plot_recall(ax, recall_df, third_line, pooled_name)

    plt.tight_layout()
    if total_n:
        # share of all test mentions with a seen status, under the pooled and
        # seen ticks, on the same line as the x-axis title
        fig.canvas.draw()
        label_box = ax.xaxis.label.get_window_extent()
        y = fig.transFigure.inverted().transform((0, (label_box.y0 + label_box.y1) / 2))[1]
        xs = x_positions(bin_order(recall_df))
        for b in ("pooled", "seen"):
            n = int(recall_df.loc[recall_df["bin"] == b, "n"].iloc[0])
            x = fig.transFigure.inverted().transform(ax.transData.transform((xs[b], 0)))[0]
            fig.text(x, y, f"{100 * n / total_n:.0f}\\%", ha="center", va="center", fontsize=TICK_FONTSIZE)
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"[saved] {save_path}")


def plot_points(ax, series, connect):
    """series: {model: dataframe with recall / ci_low / ci_high, one row per x
    position 0, 1, ...}; dodged markers, CIs, and a line if connect."""
    models = [m for m in MODELS if m in series]
    for j, model in enumerate(models):
        d = series[model].reset_index(drop=True)
        x = [i + (j - 1) * DODGE for i in range(len(d))]
        if connect:
            ax.plot(x, d["recall"], color=COLORS[model], linestyle=LINESTYLES[model], linewidth=1.3, zorder=2)
        ax.vlines(x, d["ci_low"], d["ci_high"], color=COLORS[model], linewidth=0.8, zorder=2)
        ax.plot(x, d["recall"], linestyle="none", marker=MARKERS[model], markersize=4.5,
                color=COLORS[model], markeredgecolor="white", markeredgewidth=0.6, zorder=3)
    return [Line2D([], [], color=COLORS[m], linestyle=LINESTYLES[m], marker=MARKERS[m], markersize=4.5,
                   markeredgecolor="white", markeredgewidth=0.6, linewidth=1.3, label=m) for m in models]


def plot_combined(panels, save_path):
    """panels: [(recall_df, title), ...]. One panel per group with recall by
    proximity quintile (Q1-Q5 only), then one narrower panel with the pooled
    recall of each group and the seen reference side by side. Shared y axis
    and legend, at the width of the single-panel figure."""
    su.configure_fonts()
    fig_w, fig_h = FIGURE_CM_SINGLE[0] / 2.54, FIGURE_CM_SINGLE[1] / 2.54
    n = len(panels)
    fig, axes = plt.subplots(1, n + 1, figsize=(fig_w, fig_h), sharey=True,
                             gridspec_kw={"width_ratios": [5] * n + [4]})

    for ax, (recall_df, title) in zip(axes[:n], panels):
        bins = [b for b in bin_order(recall_df) if b not in REFERENCES]
        series = {m: g.set_index("bin").loc[bins] for m, g in recall_df.groupby("model")}
        handles = plot_points(ax, series, connect=True)
        n_pooled = int(recall_df.loc[recall_df["bin"] == "pooled", "n"].iloc[0])
        # bins are equal-size quintiles, so N per bin is about N / 5
        ax.set_title(f"{title} (N = {n_pooled})", fontsize=8, pad=3)
        ax.set_xticks(range(len(bins)))
        ax.set_xticklabels(bins, fontsize=TICK_FONTSIZE)
        ax.set_xlim(-0.5, len(bins) - 0.5)
        ax.set_xlabel(r"Similarity (far $\rightarrow$ close)")

    # pooled panel: each group's pooled recall, then the seen reference
    ref_rows = [(df, "pooled", title) for df, title in panels] + [(panels[0][0], "seen", "Seen")]
    series = {m: pd.DataFrame([df[(df["model"] == m) & (df["bin"] == b)].iloc[0] for df, b, _ in ref_rows])
              for m in MODELS if m in set(panels[0][0]["model"])}
    plot_points(axes[n], series, connect=False)
    total = sum(int(df.loc[(df["bin"] == b), "n"].iloc[0]) for df, b, _ in ref_rows)
    axes[n].set_title(f"Pooled (N = {total:,})", fontsize=8, pad=3)  # base of the % in the ticks
    axes[n].set_xticks(range(len(ref_rows)))
    two_lines = {"Partial overlap": r"Partial\\overlap", "No word overlap": r"No word\\overlap",
                 "Seen": r"Seen\\(ref.)"}
    # third line: share of the test mentions with a seen status (seen + the groups)
    sizes = [int(df.loc[(df["bin"] == b), "n"].iloc[0]) for df, b, _ in ref_rows]
    shares = [f"{100 * s / sum(sizes):.0f}\\%" for s in sizes]
    axes[n].set_xticklabels([r"\shortstack{" + two_lines.get(t, t) + r"\\" + share + "}"
                             for (_, _, t), share in zip(ref_rows, shares)], fontsize=TICK_FONTSIZE)
    axes[n].set_xlim(-0.5, len(ref_rows) - 0.5)

    for k, ax in enumerate(axes):
        ax.set_ylim(0, 1)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.1f}"))
        ax.grid(axis="y", color="grey", alpha=0.3, linewidth=0.3)
        ax.spines[["top", "right"]].set_visible(False)
        if k > 0:
            ax.tick_params(axis="y", length=0)
    axes[0].set_ylabel("Recall")
    fig.legend(handles=handles, frameon=False, fontsize=8, loc="lower center",
               bbox_to_anchor=(0.5, 1.0), ncol=len(handles), handlelength=2.5)
    plt.tight_layout(w_pad=0.8)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"[saved] {save_path}")


def total_status_n(variant):
    """All test mentions with a seen status under the variant's category filter
    (seen + partial + no overlap): the base of the shares in the tick labels."""
    filt = "_f1med70" if variant.endswith("f1med70") else ""
    n = lambda f, b: int(pd.read_csv(OUTPUT_DIR / f"recall_by_proximity_bin_{f}{filt}.csv")
                         .query("bin == @b")["n"].iloc[0])
    return n("partial", "seen") + n("partial", "pooled") + n("no_overlap", "pooled")


def main():
    pairs_df = pd.read_csv(PAIRS_PATH)
    for variant, (third_line, pooled_name) in VARIANTS.items():
        suffix = f"_{variant}" if variant else ""
        recall_df = pd.read_csv(OUTPUT_DIR / f"recall_by_proximity_bin{suffix}.csv")
        plot_proximity(recall_df, pairs_df, OUTPUT_DIR / f"semantic_proximity{suffix}.pdf",
                       third_line, pooled_name, total_status_n(variant))
    for suffix, panels in COMBINED.items():
        dfs = [(pd.read_csv(OUTPUT_DIR / f"recall_by_proximity_bin_{v}.csv"), title) for v, title in panels]
        out = OUTPUT_DIR / ("semantic_proximity_partial_no_overlap" + (f"_{suffix}" if suffix else "") + ".pdf")
        plot_combined(dfs, out)


if __name__ == "__main__":
    main()
