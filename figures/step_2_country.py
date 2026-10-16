"""
step_2_country.py

Description:
Country split (France vs Germany) of the Step 2 performance figures made by
step_2_boxplot.py, which stay unchanged: same two figures (the 8 broad classes,
and all specific categories grouped by broad class, split into 2 parts), same
four panels (binary precision, recall, F1, macro F1) and same row order.

Per (category, country), split per fold, many test folds hold only 1-4 positive
cases, so per-fold scores are too noisy to show as boxplots or fold means.
Instead the 5 folds' held-out predictions are pooled: each sentence is in the
test fold exactly once per broad class, so pooling gives one held-out
prediction per (sentence, category). Each row shows the pooled score per
country (French: red circle, German: grey square, as in the findings_* figures) with a 95% bootstrap confidence
interval (sentences resampled within country, N_BOOT resamples), and a thin black
tick for both countries pooled. Predictions are pred_label (prob_entail > 0.5),
from which fold_*_per_label.csv (used by step_2_boxplot.py) are computed.
Broad-class rows aggregate their categories as in step_2_boxplot.py (average
weighted by the number of positive cases), recomputed in every resample.

N columns: number of mentions (positive cases) per country, pooled over the 5
folds -- i.e. about 5 times the per-fold mean shown as N in step_2_boxplot.py.

Outputs (in step_2/boxplots/country/):
- step_2_perf_broad_class_country.pdf
- step_2_perf_country_part1.pdf, step_2_perf_country_part2.pdf
- step_2_perf_country_metrics.csv     all metrics, both countries + pooled, with CIs
                                      and the France - Germany difference
- step_2_perf_country_table_part{1,2}.tex  F1 per country, difference, N
- step_2_perf_country_summary.txt     rows whose France - Germany difference CI excludes 0

Usage (from this directory):
python step_2_country.py
"""
import itertools
import os
import re
import textwrap
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

# ============================================================
# PARAMETERS (copied from step_2_boxplot.py, which runs on import)
# ============================================================
HERE = Path(__file__).resolve().parent
STEP_2_DATA_DIR = HERE.parent / "data" / "model_performance" / "step_2"
OUTPUT_DIR = HERE / "step_2" / "boxplots" / "country"

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
BROAD_CLASS_LIST = list(FOLDER_NAME_MAP)

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
    "identity": "Gender, sexuality, and sociocultural characteristics",
    "profession": "Profession",
    "social_roles": "Social roles and behavior",
    "social_deviance": "Social deviance",
    "real_estate": "Real estate ownership",
}

METRICS = ["precision_binary", "recall_binary", "f1_binary", "f1_macro"]
METRIC_TITLE_MAP = {
    "precision_binary": "Binary Precision",
    "recall_binary": "Binary Recall",
    "f1_binary": "Binary F1",
    "f1_macro": "Macro F1",
}

COUNTRIES = ["France", "Germany"]
COUNTRY_SHORT = {"France": "FR", "Germany": "DE"}
# same colors, markers, offset and legend labels as the other country comparisons
# (utils.plot_dot_iqr, used by the findings_* figures): French above, German below
COUNTRY_STYLE = {
    "France": dict(color="#C0392B", marker="o"),
    "Germany": dict(color="#909090", marker="s"),
}
COUNTRY_LEGEND = {"France": "French", "Germany": "German"}
POOLED_COLOR = "black"
COUNTRY_OFFSET = 0.15   # vertical offset of each country's marker from the row centre (row units)

N_BOOT = 2000
SEED = 42

FIG_WIDTH_CM = 14.0
WRAP_CHARS = 28
ROW_HEIGHT_CM = 0.45
TOP_MARGIN_CM = 0.5
BOTTOM_MARGIN_CM = 1.0      # room for the legend
LEGEND_BOTTOM_CM = 0.1      # legend baseline above the figure's bottom edge
BROAD_LBL_WIDTH_FRAC = 0.24
MERGED_LBL_WIDTH_FRAC = 0.27
MERGED_N_PARTS = 2
MERGED_GROUP_SPACING_CM = 0.3

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
    "text.latex.preamble": r"""
        \usepackage[T1]{fontenc}
        \usepackage{lmodern}
        \renewcommand{\familydefault}{\sfdefault}
    """,
})

MANUAL_WRAP = {
    "Gender, sexuality, and sociocultural characteristics":
        ["Gender, sexuality, and", "sociocultural characteristics"],
}


# ============================================================
# HELPERS (layout helpers as in step_2_boxplot.py)
# ============================================================
def wrap_lines(lbl, width_chars):
    return MANUAL_WRAP.get(str(lbl)) or textwrap.wrap(str(lbl), width_chars)


def wrap_and_capitalize(lbl, width_chars):
    lines = wrap_lines(lbl, width_chars)
    lines[0] = lines[0][:1].upper() + lines[0][1:]
    return "\n".join(lines)


def compute_y_positions(raw_labels, header_labels=frozenset(), extra_gap_rows=0.0):
    positions = np.zeros(len(raw_labels))
    for i in range(1, len(raw_labels)):
        gap = 1.0 + (extra_gap_rows if raw_labels[i] in header_labels else 0.0)
        positions[i] = positions[i - 1] - gap
    return positions


def balanced_contiguous_split(sizes, n_parts):
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


def load_per_label_folds(bc):
    """fold_*_per_label.csv of one broad class (as step_2_boxplot.py), used for the row order."""
    folder = STEP_2_DATA_DIR / FOLDER_NAME_MAP[bc]
    dfs = []
    for f in sorted(folder.glob("fold_*_per_label.csv")):
        d = pd.read_csv(f)
        d["fold"] = int(re.search(r"fold_(\d+)_per_label", f.name).group(1))
        dfs.append(d)
    return pd.concat(dfs, ignore_index=True)


def row_order():
    """Same order as step_2_boxplot.py: broad classes by mean (over folds) of their
    n_pos_entail-weighted macro F1, categories within a class by mean macro F1."""
    broad_scores, group_labels = {}, {}
    for bc in BROAD_CLASS_LIST:
        d = load_per_label_folds(bc)
        per_fold = d.groupby("fold").apply(
            lambda g: np.average(g["f1_macro"], weights=g["n_pos_entail"]), include_groups=False)
        broad_scores[bc] = per_fold.mean()
        cat_means = d.groupby("hypothesis_label")["f1_macro"].mean()
        group_labels[bc] = cat_means.sort_values(ascending=False, kind="mergesort").index.tolist()
    order = sorted(BROAD_CLASS_LIST, key=lambda bc: -broad_scores[bc])
    return order, group_labels


def load_predictions(bc):
    """Pooled held-out predictions of the 5 folds: Y, P (sentences x categories, 1 =
    entailment), sentence countries, and category names."""
    folder = STEP_2_DATA_DIR / FOLDER_NAME_MAP[bc]
    cols = ["sentence_id", "hypothesis_label", "country", "nli_label", "pred_label"]
    d = pd.concat([pd.read_csv(f, usecols=cols) for f in sorted(folder.glob("fold_*_human_vs_model.csv"))])
    assert not d.duplicated(["sentence_id", "hypothesis_label"]).any(), f"{bc}: a (sentence, category) pair is in several test folds"
    assert set(d["country"]) <= set(COUNTRIES), f"{bc}: unexpected countries {set(d['country'])}"
    d["y"] = (d["nli_label"] == 0).astype(np.int8)   # 0 = entailment (positive class)
    d["p"] = (d["pred_label"] == 0).astype(np.int8)
    labels = sorted(d["hypothesis_label"].unique())
    Y = d.pivot(index="sentence_id", columns="hypothesis_label", values="y")[labels]
    P = d.pivot(index="sentence_id", columns="hypothesis_label", values="p")[labels]
    assert not Y.isna().any().any(), f"{bc}: some sentences lack a pair for some category"
    country = d.drop_duplicates("sentence_id").set_index("sentence_id").loc[Y.index, "country"].to_numpy()
    return Y.to_numpy(), P.to_numpy(), country, labels


def metrics_from_counts(tp, fp, fn, tn):
    """Element-wise binary precision/recall/F1 (positive class) and macro F1;
    0 where undefined (sklearn's zero_division=0)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        prec = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
        rec = np.where(tp + fn > 0, tp / (tp + fn), 0.0)
        f1 = np.where(2 * tp + fp + fn > 0, 2 * tp / (2 * tp + fp + fn), 0.0)
        f1_neg = np.where(2 * tn + fp + fn > 0, 2 * tn / (2 * tn + fp + fn), 0.0)
    return {"precision_binary": prec, "recall_binary": rec, "f1_binary": f1, "f1_macro": (f1 + f1_neg) / 2}


def scope_scores(Y, P, rng):
    """Point estimates and bootstrap samples for one set of sentences.
    Returns {metric: (point[L+1], boot[B, L+1])}; the last column is the broad-class
    aggregate (average over categories weighted by the number of positive cases)."""
    n = len(Y)
    TP, FP, FN = Y & P, (1 - Y) & P, Y & (1 - P)
    TN = (1 - Y) & (1 - P)
    W = rng.multinomial(n, np.full(n, 1 / n), size=N_BOOT)   # bootstrap weights per sentence

    def counts(w):
        return [w @ M for M in (TP, FP, FN, TN)]

    out = {}
    for name, w in (("point", np.ones(n, dtype=np.int64)), ("boot", W)):
        tp, fp, fn, tn = counts(w)
        m = metrics_from_counts(tp, fp, fn, tn)
        npos = tp + fn
        for metric, v in m.items():
            agg = (v * npos).sum(axis=-1) / npos.sum(axis=-1)
            out.setdefault(metric, {})[name] = np.concatenate([v, agg[..., None]], axis=-1)
        out.setdefault("n_pos", {})[name] = np.concatenate([npos, npos.sum(axis=-1)[..., None]], axis=-1)
    return {k: (v["point"], v["boot"]) for k, v in out.items()}


def compute_results():
    """Long table: one row per (row label, metric) with pooled and per-country scores,
    95% CIs and the France - Germany difference."""
    rng = np.random.default_rng(SEED)
    rows = []
    for bc in BROAD_CLASS_LIST:
        Y, P, country, labels = load_predictions(bc)
        row_names = [RENAME_DICT.get(l, l) for l in labels] + [RENAME_DICT[bc]]
        scores = {"All": scope_scores(Y, P, rng)}
        for c in COUNTRIES:
            mask = country == c
            scores[c] = scope_scores(Y[mask], P[mask], rng)
        for metric in METRICS:
            for j, name in enumerate(row_names):
                r = {"broad_class": RENAME_DICT[bc], "label": name, "is_broad_class": j == len(labels),
                     "metric": metric}
                for scope, s in scores.items():
                    key = COUNTRY_SHORT.get(scope, "all")
                    point, boot = s[metric]
                    r[f"{key}"] = point[j]
                    r[f"{key}_lo"], r[f"{key}_hi"] = np.percentile(boot[:, j], [2.5, 97.5])
                    r[f"n_{key}"] = int(s["n_pos"][0][j])
                diff = scores["France"][metric][1][:, j] - scores["Germany"][metric][1][:, j]
                r["diff_FR_DE"] = r["FR"] - r["DE"]
                r["diff_lo"], r["diff_hi"] = np.percentile(diff, [2.5, 97.5])
                r["diff_ci_excludes_0"] = bool(r["diff_lo"] > 0 or r["diff_hi"] < 0)
                rows.append(r)
    return pd.DataFrame(rows)


# ============================================================
# FIGURE
# ============================================================
def render_country_figure(res, raw_labels, wrapped_labels, output_path, lbl_width_frac,
                          header_labels=frozenset(), extra_gap_rows=0.0):
    y = compute_y_positions(raw_labels, header_labels, extra_gap_rows)
    n_rows_equiv = (y[0] - y[-1]) + 1
    fig_h_in = (TOP_MARGIN_CM + BOTTOM_MARGIN_CM + n_rows_equiv * ROW_HEIGHT_CM) * cm
    gs = gridspec.GridSpec(
        nrows=1, ncols=6, left=lbl_width_frac, right=0.98,
        bottom=BOTTOM_MARGIN_CM * cm / fig_h_in, top=1 - TOP_MARGIN_CM * cm / fig_h_in,
        wspace=0.15, width_ratios=[1, 1, 1, 1, 0.27, 0.27],
    )
    fig = plt.figure(figsize=(FIG_WIDTH_CM * cm, fig_h_in))
    axs = [fig.add_subplot(gs[0, i]) for i in range(4)]
    ax_n = {c: fig.add_subplot(gs[0, 4 + k]) for k, c in enumerate(COUNTRIES)}
    by_label = res.set_index(["label", "metric"])

    for i, (ax, metric) in enumerate(zip(axs, METRICS)):
        for lbl, yi in zip(raw_labels, y):
            r = by_label.loc[(lbl, metric)]
            ax.plot([r["all"]] * 2, [yi - 0.32, yi + 0.32], color=POOLED_COLOR, linewidth=0.7, zorder=1)
            for c, sign in zip(COUNTRIES, (1, -1)):
                k = COUNTRY_SHORT[c]
                ax.errorbar(r[k], yi + sign * COUNTRY_OFFSET,
                            xerr=[[r[k] - r[f"{k}_lo"]], [r[f"{k}_hi"] - r[k]]],
                            fmt=COUNTRY_STYLE[c]["marker"], color=COUNTRY_STYLE[c]["color"],
                            markersize=3.5, markeredgecolor="white", markeredgewidth=0.4,
                            elinewidth=0.9, capsize=0, zorder=2)
        ax.set_ylim(min(y) - 0.5, max(y) + 0.5)
        ax.set_xlim(-0.03, 1.03)   # keeps scores of 0 / 1 visible
        ax.set_xticks([0, 0.5, 1])
        ax.set_xticklabels(["0", "0.5", "1"])
        ax.set_xticks([0.25, 0.75], minor=True)
        ax.grid(axis="x", linestyle="--", alpha=0.4)
        ax.grid(axis="x", which="minor", linestyle="--", alpha=0.4)
        ax.set_title(METRIC_TITLE_MAP[metric])
        ax.set_yticks(y)
        if i == 0:
            ax.set_yticklabels(wrapped_labels)
        else:
            ax.tick_params(axis="y", which="both", labelleft=False, length=0)

    for c, ax in ax_n.items():
        k = COUNTRY_SHORT[c]
        ax.set_title(f"N {k}", fontsize=7)
        for lbl, yi in zip(raw_labels, y):
            ax.text(0.5, yi, str(by_label.loc[(lbl, METRICS[0]), f"n_{k}"]), ha="center", va="center", fontsize=7)
        ax.set_ylim(min(y) - 0.5, max(y) + 0.5)
        ax.set_xlim(0, 1)
        ax.set_xticks([])
        ax.set_yticks([])
        for side in ax.spines.values():
            side.set_visible(False)

    handles = [Line2D([], [], linestyle="none", markersize=4, label=COUNTRY_LEGEND[c], **COUNTRY_STYLE[c])
               for c in COUNTRIES]
    handles.append(Line2D([], [], color=POOLED_COLOR, linewidth=0.7, label="Both"))
    # left-aligned with the first panel's left edge (no padding, so the first marker
    # starts exactly at the axis), in the bottom margin below the x tick labels
    fig.legend(handles=handles, loc="lower left", ncol=3, frameon=False, fontsize=7,
               borderpad=0, borderaxespad=0, handlelength=1.2, handletextpad=0.4, columnspacing=1.5,
               bbox_to_anchor=(axs[0].get_position().x0, LEGEND_BOTTOM_CM * cm / fig_h_in))
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Saved file as: {output_path}")


# ============================================================
# TABLES
# ============================================================
def latex_escape(s):
    return str(s).replace("&", r"\&").replace("%", r"\%").replace("_", r"\_")


def build_latex_table(res, part_labels, header_labels, part_num, n_parts):
    f1 = res[res["metric"] == "f1_binary"].set_index("label")
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\scriptsize",
        r"\begin{threeparttable}",
        rf"\caption{{Step 2 binary F1 by country, pooled over the 5 cross-validation folds. Part {part_num}/{n_parts}.}}",
        rf"\label{{tab:step2-country-part{part_num}}}",
        r"\begin{tabular}{l rr r rr}",
        r"\toprule",
        r"Group & F1 FR & F1 DE & FR $-$ DE [95\% CI] & N FR & N DE \\",
        r"\midrule",
    ]
    for lbl in part_labels:
        r = f1.loc[lbl]
        name = latex_escape(lbl[:1].upper() + lbl[1:])
        diff = f"{r['diff_FR_DE']:+.2f} [{r['diff_lo']:+.2f}, {r['diff_hi']:+.2f}]"
        if r["diff_ci_excludes_0"]:
            diff += r"$^{*}$"
        cells = [f"{r['FR']:.2f}", f"{r['DE']:.2f}", diff, f"{r['n_FR']:,}", f"{r['n_DE']:,}"]
        if lbl in header_labels:
            if lbl != part_labels[0]:
                lines.append(r"\addlinespace")
            lines.append(rf"\textbf{{{name}}} & " + " & ".join(rf"\textbf{{{c}}}" for c in cells) + r" \\")
        else:
            lines.append(rf"\quad {name} & " + " & ".join(cells) + r" \\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\begin{tablenotes}[flushleft]",
        r"\footnotesize",
        r"\item \textit{Note:} Held-out predictions of the 5 cross-validation folds pooled, so that each "
        r"sentence is scored once. Bold rows: broad classes (average of their groups' F1, weighted by "
        r"the number of mentions). CI: 95\% bootstrap interval over sentences "
        rf"({N_BOOT:,} resamples). $^{{*}}$: the interval of the difference excludes 0. "
        r"N: number of mentions of the group per country.",
        r"\end{tablenotes}",
        r"\end{threeparttable}",
        r"\end{table}",
        "",
    ]
    return "\n".join(lines)


# ============================================================
# MAIN
# ============================================================
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    res = compute_results()
    res.to_csv(OUTPUT_DIR / "step_2_perf_country_metrics.csv", index=False)
    print(f"Saved file as: {OUTPUT_DIR / 'step_2_perf_country_metrics.csv'}")

    broad_order, group_labels = row_order()
    broad_names = [RENAME_DICT[bc] for bc in broad_order]

    # Figure 1: the 8 broad classes
    render_country_figure(
        res, broad_names, [wrap_and_capitalize(b, WRAP_CHARS) for b in broad_names],
        OUTPUT_DIR / "step_2_perf_broad_class_country.pdf", BROAD_LBL_WIDTH_FRAC)

    # Figure 2: all categories grouped by broad class (header row = broad class), 2 parts
    groups = [[RENAME_DICT[bc]] + [RENAME_DICT.get(l, l) for l in group_labels[bc]] for bc in broad_order]
    header_labels = set(broad_names)
    parts = balanced_contiguous_split([len(g) for g in groups], MERGED_N_PARTS)
    for part_num, idxs in enumerate(parts, start=1):
        part_labels = [lbl for i in idxs for lbl in groups[i]]
        wrapped = [
            "\n".join(rf"\textbf{{{line}}}" for line in wrap_lines(lbl, WRAP_CHARS))
            if lbl in header_labels else wrap_and_capitalize(lbl, WRAP_CHARS)
            for lbl in part_labels
        ]
        render_country_figure(
            res, part_labels, wrapped, OUTPUT_DIR / f"step_2_perf_country_part{part_num}.pdf",
            MERGED_LBL_WIDTH_FRAC, header_labels=header_labels,
            extra_gap_rows=MERGED_GROUP_SPACING_CM / ROW_HEIGHT_CM)
        tex = build_latex_table(res, part_labels, header_labels, part_num, len(parts))
        (OUTPUT_DIR / f"step_2_perf_country_table_part{part_num}.tex").write_text(tex, encoding="utf-8")
        print(f"Saved file as: {OUTPUT_DIR / f'step_2_perf_country_table_part{part_num}.tex'}")

    # Summary: rows whose France - Germany difference CI excludes 0
    lines = [f"France - Germany differences whose 95% bootstrap CI excludes 0 ({N_BOOT:,} resamples, seed {SEED})", ""]
    for metric in METRICS:
        sub = res[(res["metric"] == metric) & res["diff_ci_excludes_0"]]
        n_cat = int((~res.loc[res["metric"] == metric, "is_broad_class"]).sum())
        lines.append(f"{METRIC_TITLE_MAP[metric]}: {int((~sub['is_broad_class']).sum())} of {n_cat} categories, "
                     f"{int(sub['is_broad_class'].sum())} of {len(BROAD_CLASS_LIST)} broad classes")
        for _, r in sub.sort_values(["is_broad_class", "diff_FR_DE"], ascending=[False, True]).iterrows():
            kind = "[broad]" if r["is_broad_class"] else "       "
            lines.append(f"  {kind} {r['label']:<55} FR {r['FR']:.2f}  DE {r['DE']:.2f}  "
                         f"diff {r['diff_FR_DE']:+.2f} [{r['diff_lo']:+.2f}, {r['diff_hi']:+.2f}]  "
                         f"N FR/DE {r['n_FR']}/{r['n_DE']}")
        lines.append("")
    text = "\n".join(lines)
    (OUTPUT_DIR / "step_2_perf_country_summary.txt").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
