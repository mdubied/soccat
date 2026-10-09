"""
cross_language_transfer.py

Description:
Figures for the cross-lingual transfer analysis of the Step 2 NLI models.

Reads only the summary CSVs in ../cross-language_transfer/, produced by
CV_cross_lingual_summary.ipynb (which itself combines the per-category
cross-lingual cross-validation runs). Two conditions are compared, both
predicting every sentence of the target language exactly once (5 folds):
  - within_matched: trained on the target language,
  - cross_matched:  trained on the other language,
with the training set size matched between the two conditions.

Figures:
1. Within-language vs cross-lingual F1 per specific group label, one panel per
   direction, split in two parts (same broad class order, label order and
   part split as step_2_boxplot.py's step_2_perf_boxplot_part1/2.pdf).
   Red circles: within language; grey squares: cross-lingual (the two
   colors of the findings_* figures).
   Point: positive-class F1 pooled over the 5 test folds; bar: 95% bootstrap
   CI (resampled sentences). Bold rows: the whole broad class, F1 micro-averaged
   over its labels with paired-bootstrap CIs. N: positive test sentences.
2. Recall by similarity of each positive test mention to its nearest
   same-label training mention (quartiles, computed per broad class,
   direction and condition), pooled over all labels, with Wilson 95% CIs.

Outputs (in "cross_language/"):
- cross_language_f1_part1.pdf, cross_language_f1_part2.pdf
- cross_language_similarity.pdf
- cross_language_summary.txt (numbers behind the figures)

Usage (from this directory):
python cross_language_transfer.py
"""
import os, re, glob, textwrap, itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D

# ============================================================
# PARAMETERS (edit here)
# ============================================================
XL_DIR = "../cross-language_transfer"
STEP_2_DATA_DIR = "../data/model_performance/step_2"
OUTPUT_DIR = "cross_language"

DIRS = ["fr2de", "de2fr"]
DIR_TITLE = {
    "fr2de": "Trained on French, tested on German",
    "de2fr": "Trained on German, tested on French",
}
DIR_TITLE_2L = {d: t.replace(", ", ",\n") for d, t in DIR_TITLE.items()}  # narrow F1 panels
CONDITIONS = ["within_matched", "cross_matched"]
COND_LABEL = {
    "within_matched": "Within language",
    "cross_matched": "Cross-language",
}
# the two colors/markers of the findings_* figures (utils.py)
COND_STYLE = {
    "within_matched": dict(marker="o", color="#C0392B", offset=+0.17),
    "cross_matched": dict(marker="s", color="#909090", offset=-0.17),
}

# Grey out labels whose smaller French/German positive count is below this
# (the notebook's MIN_POS "reportable" flag). None = no greying.
GREY_BELOW_MIN_POS = None

# Star next to N when the paired 95% CI of the within minus cross-lingual
# F1 difference excludes 0 (crosslingual_paired_gaps.csv for labels,
# crosslingual_category_overview.csv for broad class rows)
SHOW_SIGNIFICANCE = True

# CSV "category" value -> (display name, step 2 results folder)
CATEGORY_MAP = {
    "Socio-economic": ("Socio-economic position", "socio_economic_position"),
    "Labour market": ("Labor market position", "labor_market_position"),
    "Age and family": ("Age and family status", "age_family_status"),
    "Identities": ("Gender, sexuality, and sociocultural characteristics", "identities_minority_majority_status"),
    "Profession": ("Profession", "profession"),
    "Social roles": ("Social roles and behavior", "social_roles_behavior"),
    "Social deviance": ("Social deviance", "social_deviance"),
    "Real estate": ("Real estate ownership", "real_estate_ownership"),
}

# Same display renames as step_2_boxplot.py
RENAME_DICT = {
    "entrepreneurs in [specific] sector": "entrepreneurs in specific sector",
    "lgbtqqia+": "LGBTQIA+",
    "people with an immigration background, including immigrants": "people with immigration background",
    "offenders, criminals, prisoners and/or accused people": "offenders, criminals, prisoners, accused people",
    "terrorists, rebels, revolutionaries and/or movements of armed resistance": "terrorists, revolutionaries, rebels, armed resistance",
}
MANUAL_WRAP = {
    "Gender, sexuality, and sociocultural characteristics":
        ["Gender, sexuality, and", "sociocultural characteristics"],
}

# Figure parameters (as in step_2_boxplot.py)
N_PARTS = 2
FIG_WIDTH_CM = 14.0
WRAP_CHARS = 28
ROW_HEIGHT_CM = 0.45
TOP_MARGIN_CM = 0.8         # two-line panel titles
BOTTOM_MARGIN_CM = 1.1      # room for the legend
GROUP_SPACING_CM = 0.3      # extra gap above each broad class header row
LBL_WIDTH_FRAC = 0.27
SIM_FIG_HEIGHT_CM = 5.5

cm = 1 / 2.54
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Latin Modern Sans"],
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "text.latex.preamble": r"""
        \usepackage[T1]{fontenc}
        \usepackage{lmodern}
        \renewcommand{\familydefault}{\sfdefault}
    """,
})

# ============================================================
# HELPERS
# ============================================================

def tex(s):
    return re.sub(r"([&%_#$])", r"\\\1", str(s))

def wrap_lines(lbl, width_chars=WRAP_CHARS):
    return MANUAL_WRAP.get(str(lbl)) or textwrap.wrap(str(lbl), width_chars)

def label_tick(lbl, header=False, grey=False):
    lines = [tex(l) for l in wrap_lines(lbl)]
    if header:
        return "\n".join(rf"\textbf{{{l}}}" for l in lines)
    lines[0] = lines[0][0].upper() + lines[0][1:]
    if grey:
        return "\n".join(rf"\textcolor[gray]{{0.55}}{{{l}}}" for l in lines)
    return "\n".join(lines)

def balanced_contiguous_split(sizes, n_parts):
    """Same as step_2_boxplot.py: contiguous split minimizing the largest part."""
    n = len(sizes)
    if n_parts >= n:
        return [[i] for i in range(n)]
    best_cuts, best_max = None, None
    for cuts in itertools.combinations(range(1, n), n_parts - 1):
        bounds = (0,) + cuts + (n,)
        worst = max(sum(sizes[bounds[k]:bounds[k + 1]]) for k in range(n_parts))
        if best_max is None or worst < best_max:
            best_max, best_cuts = worst, bounds
    return [list(range(best_cuts[k], best_cuts[k + 1])) for k in range(n_parts)]

def boxplot_order():
    """Broad class and label order of step_2_boxplot.py's merged figure:
    broad classes by mean (over folds) of the n_pos_entail-weighted macro F1,
    labels within a class by their mean macro F1 over folds, best first."""
    class_score, label_order = {}, {}
    for short, (_, folder) in CATEGORY_MAP.items():
        files = glob.glob(f"{STEP_2_DATA_DIR}/{folder}/fold_*_per_label.csv")
        assert files, f"No fold_*_per_label.csv in {STEP_2_DATA_DIR}/{folder}"
        d = pd.concat(
            [pd.read_csv(f).assign(fold=int(re.search(r"fold_(\d+)_", f).group(1))) for f in files],
            ignore_index=True,
        )
        class_score[short] = d.groupby("fold").apply(
            lambda g: np.average(g["f1_macro"], weights=g["n_pos_entail"]), include_groups=False
        ).mean()
        label_order[short] = (d.groupby("hypothesis_label")["f1_macro"].mean()
                               .sort_values(ascending=False, kind="mergesort").index.tolist())
    classes = sorted(class_score, key=lambda c: -class_score[c])
    return classes, label_order

def wilson(h, n, z=1.96):
    p = h / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    hw = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - hw, c + hw

def style_f1_axis(ax):
    ax.set_xlim(-0.03, 1.03)   # keeps F1 = 0 / 1 markers visible
    ax.set_xticks([0, 0.5, 1])
    ax.set_xticklabels(["0", "0.5", "1"])
    ax.set_xticks([0.25, 0.75], minor=True)
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    ax.grid(axis="x", which="minor", linestyle="--", alpha=0.4)

def style_n_axis(ax, ylim):
    ax.set_title("N")
    ax.set_ylim(*ylim)
    ax.set_xlim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)

def legend_handles():
    return [Line2D([], [], ls="", marker=COND_STYLE[c]["marker"], color=COND_STYLE[c]["color"],
                   ms=4, label=COND_LABEL[c]) for c in CONDITIONS]

# ============================================================
# LOAD DATA
# ============================================================
os.makedirs(OUTPUT_DIR, exist_ok=True)

per_label = pd.read_csv(f"{XL_DIR}/crosslingual_per_label_all.csv")
per_label = per_label[per_label["condition"].isin(CONDITIONS)]
overview = pd.read_csv(f"{XL_DIR}/crosslingual_category_overview.csv")
gaps = pd.read_csv(f"{XL_DIR}/crosslingual_paired_gaps.csv")
sim = pd.read_csv(f"{XL_DIR}/crosslingual_similarity_rows_all.csv")
sim = sim[sim["condition"].isin(CONDITIONS)].dropna(subset=["max_sim"])

assert set(per_label["category"]) == set(CATEGORY_MAP), set(per_label["category"]) ^ set(CATEGORY_MAP)

# value lookups: (category, label, direction, condition) -> row
gap_idx = gaps.set_index(["category", "hypothesis_label", "direction"])
pl_idx = per_label.set_index(["category", "hypothesis_label", "direction", "condition"])
ov_idx = overview.set_index(["category", "direction"])
min_both = per_label.groupby(["category", "hypothesis_label"])["min_both"].first()

summary = []

# ============================================================
# FIGURE 1: F1 PER SPECIFIC GROUP LABEL (two parts)
# ============================================================
classes, label_order = boxplot_order()
group_rows = {c: [("header", c)] + [("label", c, l) for l in label_order[c]] for c in classes}
parts = balanced_contiguous_split([len(group_rows[c]) for c in classes], N_PARTS)
gap_rows = GROUP_SPACING_CM / ROW_HEIGHT_CM

def point(row_key, direction, cond):
    """(F1, lo, hi, N) for a label row or a broad class header row."""
    if row_key[0] == "header":
        r = ov_idx.loc[(row_key[1], direction)]
        pre = "microf1_within" if cond == "within_matched" else "microf1_cross"
        return r[pre], r[f"{pre}_lo"], r[f"{pre}_hi"], r["n_test_pos"]
    r = pl_idx.loc[(row_key[1], row_key[2], direction, cond)]
    return r["f1_binary"], r["f1_lo"], r["f1_hi"], r["n_pos_entail"]

def gap_significant(row_key, direction):
    """True if the paired 95% bootstrap CI of the within minus cross F1 gap
    excludes 0 (both conditions predict the same sentences, so the bootstrap
    resamples sentences and recomputes both F1s on each draw)."""
    if row_key[0] == "header":
        r = ov_idx.loc[(row_key[1], direction)]
        lo, hi = r["microf1_gap_lo"], r["microf1_gap_hi"]
    else:
        r = gap_idx.loc[(row_key[1], row_key[2], direction)]
        lo, hi = r["f1_gap_lo"], r["f1_gap_hi"]
    return lo > 0 or hi < 0

for part_num, idxs in enumerate(parts, start=1):
    rows = [rk for i in idxs for rk in group_rows[classes[i]]]

    y = np.zeros(len(rows))
    for i in range(1, len(rows)):
        y[i] = y[i - 1] - 1.0 - (gap_rows if rows[i][0] == "header" else 0.0)
    ylim = (y.min() - 0.5, y.max() + 0.5)

    fig_h_in = (TOP_MARGIN_CM + BOTTOM_MARGIN_CM) * cm + ((y[0] - y[-1]) + 1) * ROW_HEIGHT_CM * cm
    fig = plt.figure(figsize=(FIG_WIDTH_CM * cm, fig_h_in))
    gs = gridspec.GridSpec(
        nrows=1, ncols=4, left=LBL_WIDTH_FRAC, right=0.98,
        bottom=BOTTOM_MARGIN_CM * cm / fig_h_in, top=1 - TOP_MARGIN_CM * cm / fig_h_in,
        wspace=0.08, width_ratios=[1, 0.2, 1, 0.2],
    )

    for k, direction in enumerate(DIRS):
        ax = fig.add_subplot(gs[0, 2 * k])
        ax_n = fig.add_subplot(gs[0, 2 * k + 1])
        for rk, yi in zip(rows, y):
            for cond in CONDITIONS:
                f1, lo, hi, n = point(rk, direction, cond)
                st = COND_STYLE[cond]
                ax.errorbar(f1, yi + st["offset"], xerr=[[f1 - lo], [hi - f1]], fmt=st["marker"],
                            color=st["color"], ms=3, lw=0.8, capsize=0, zorder=3)
            star = r"$^{*}$" if SHOW_SIGNIFICANCE and gap_significant(rk, direction) else r"$^{\phantom{*}}$"
            ax_n.text(0.5, yi, f"{int(round(n))}{star}", ha="center", va="center", fontsize=7,
                      fontweight="bold" if rk[0] == "header" else "normal")
        style_f1_axis(ax)
        ax.set_ylim(*ylim)
        ax.set_title(DIR_TITLE_2L[direction])
        ax.set_yticks(y)
        if k == 0:
            ticks = []
            for rk in rows:
                if rk[0] == "header":
                    ticks.append(label_tick(CATEGORY_MAP[rk[1]][0], header=True))
                else:
                    grey = GREY_BELOW_MIN_POS is not None and min_both[(rk[1], rk[2])] < GREY_BELOW_MIN_POS
                    ticks.append(label_tick(RENAME_DICT.get(rk[2], rk[2]), grey=grey))
            ax.set_yticklabels(ticks)
        else:
            ax.tick_params(axis="y", which="both", labelleft=False, length=0)
        style_n_axis(ax_n, ylim)

    handles = legend_handles()
    if SHOW_SIGNIFICANCE:
        handles.append(Line2D([], [], ls="", marker=r"$*$", color="black", ms=4,
                              label="Significant difference (paired 95\% CI)"))
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False,
               bbox_to_anchor=(0.5, 0.0))
    out = os.path.join(OUTPUT_DIR, f"cross_language_f1_part{part_num}.pdf")
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved file as: {out}")

# numbers for the text
summary.append("F1 PER SPECIFIC GROUP LABEL (within_matched minus cross_matched, positive-class F1)")
summary.append("=" * 84)
for direction in DIRS:
    g = gaps[gaps["direction"] == direction]
    summary.append(
        f"{DIR_TITLE[direction]}: cross-lingual F1 lower for {int((g.f1_gap > 0).sum())} of {len(g)} labels, "
        f"paired CI above 0 for {int((g.f1_gap_lo > 0).sum())}; median gap {g.f1_gap.median():.3f}, "
        f"mean gap {g.f1_gap.mean():.3f}"
    )
summary.append("")
summary.append(f"{'Broad class':<55} {'Direction':<9} {'F1 within':>10} {'F1 cross':>9} {'Gap':>7} {'Gap 95% CI':>16} {'Retained':>9}")
for c in classes:
    for direction in DIRS:
        r = ov_idx.loc[(c, direction)]
        summary.append(
            f"{CATEGORY_MAP[c][0]:<55} {direction:<9} {r.microf1_within:>10.3f} {r.microf1_cross:>9.3f} "
            f"{r.microf1_gap:>7.3f} [{r.microf1_gap_lo:>6.3f}, {r.microf1_gap_hi:>6.3f}] {r.retention:>8.0%}"
        )

# ============================================================
# FIGURE 2: RECALL BY SIMILARITY TO THE NEAREST TRAINING MENTION
# ============================================================
N_QUANT = 4
S = sim.copy()
S["bin"] = (S.groupby(["category", "direction", "condition"])["max_sim"]
             .transform(lambda v: pd.qcut(v.rank(method="first"), N_QUANT, labels=False)))
binned = (S.groupby(["direction", "condition", "bin"])
           .agg(hits=("hit", "sum"), n=("hit", "size"), sim_median=("max_sim", "median"))
           .reset_index())
binned["recall"] = binned["hits"] / binned["n"]
binned[["lo", "hi"]] = binned.apply(lambda r: pd.Series(wilson(r.hits, r.n)), axis=1)

x = np.arange(N_QUANT)
xt = [f"Q{i + 1}" for i in range(N_QUANT)]
xt[0] += "\nleast similar"
xt[-1] += "\nmost similar"

fig, axes = plt.subplots(1, 2, figsize=(FIG_WIDTH_CM * cm, SIM_FIG_HEIGHT_CM * cm), sharey=True)
for ax, direction in zip(axes, DIRS):
    for cond in CONDITIONS:
        st = COND_STYLE[cond]
        off = 0.04 if cond == "within_matched" else -0.04
        b = binned[(binned.direction == direction) & (binned.condition == cond)].set_index("bin").reindex(x)
        ax.errorbar(x + off, b.recall, yerr=[b.recall - b.lo, b.hi - b.recall], color=st["color"],
                    marker=st["marker"], ms=4, lw=1.2, capsize=0)
    ax.set_xticks(x)
    ax.set_xticklabels(xt)
    ax.set_xlim(-0.4, N_QUANT - 0.6)
    ax.set_ylim(0, 1)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_yticklabels(["0", "0.25", "0.5", "0.75", "1"])
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_title(DIR_TITLE[direction])
axes[0].set_ylabel("Recall")
fig.supxlabel("Similarity to the nearest same-label training mention (quartile)", fontsize=8)
fig.legend(handles=legend_handles(), loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.0))
fig.tight_layout(rect=(0, 0, 1, 0.92))
out = os.path.join(OUTPUT_DIR, "cross_language_similarity.pdf")
fig.savefig(out)
plt.close(fig)
print(f"Saved file as: {out}")

summary.append("")
summary.append("RECALL BY SIMILARITY QUARTILE (all labels pooled; quartiles per broad class, direction and condition)")
summary.append("=" * 84)
summary.append(f"{'Direction':<9} {'Condition':<15} {'Bin':<4} {'N':>6} {'Recall':>7} {'95% CI':>16} {'Median sim':>11}")
for _, r in binned.iterrows():
    summary.append(f"{r.direction:<9} {r.condition:<15} Q{int(r.bin) + 1:<3} {int(r.n):>6} {r.recall:>7.3f} "
                   f"[{r.lo:>6.3f}, {r.hi:>6.3f}] {r.sim_median:>11.3f}")

out = os.path.join(OUTPUT_DIR, "cross_language_summary.txt")
with open(out, "w", encoding="utf-8") as f:
    f.write("\n".join(summary) + "\n")
print("\n".join(summary))
print(f"Saved file as: {out}")
