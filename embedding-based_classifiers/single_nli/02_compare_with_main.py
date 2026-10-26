#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Single NLI model vs main models (one per broad class)
======================================================
Compares a single NLI model trained on all 57 labels at once (data/) with
the main SOCCAT models, one per broad class
(data/model_performance/step_2/<class>/). Both are evaluated by 5-fold CV on
the same 7,234 sentences with the same gold labels (checked below), so the
comparison is paired: only the model differs.

As in figures/step_2_country.py, the 5 folds' held-out predictions are pooled
(each sentence is scored once per label; prediction = pred_label, i.e.
prob_entail > 0.5) and scored per label: binary precision, recall, F1, and
macro F1. Broad classes and "All labels" average their labels, weighted by the
number of positive cases (as in step_2_boxplot.py). 95% CIs and the
difference single - main come from a paired bootstrap over sentences
(N_BOOT resamples, the same resampled sentences for both models).
Fold-based scores (F1 per fold: mean, median) are added to the CSV for
reference.

The model is named after this script's folder (MODEL; display names in
common.py's SINGLE_NLI_MODELS), which also prefixes every output file and
LaTeX label, so they cannot be mixed up with SOCCAT's own figures.

Outputs (output/comparison/, <MODEL> = this folder's name):
    <MODEL>_vs_main_metrics.csv           all metrics, both models, CIs, differences
    <MODEL>_vs_main_broad_class.pdf       broad classes (+ all labels)
    <MODEL>_vs_main_part1.pdf, _part2.pdf all labels, grouped by broad class
    <MODEL>_vs_main_table_part{1,2}.tex   F1 of both models, difference [CI], N
                                          (LaTeX labels tab:<MODEL>_vs_main_part1, ...)
    <MODEL>_vs_main_summary.txt           overall result and significant differences

Usage: python 02_compare_with_main.py   (run from anywhere)
"""

import sys
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
sys.path.insert(0, str(REPO / "figures"))
sys.path.insert(0, str(ROOT.parent))
import step_2_country as sc  # noqa: E402  (row order, metrics and layout helpers, fonts)
from common import SINGLE_NLI_MODELS  # noqa: E402

MODEL = ROOT.name   # output file prefix
MODEL_TITLE = SINGLE_NLI_MODELS[MODEL]["title"]

SINGLE_DIR = ROOT / "data"
MAIN_DIR = REPO / "data" / "model_performance" / "step_2"
OUTPUT_DIR = ROOT / "output" / "comparison"
N_FOLDS = 5
N_BOOT = 2000
SEED = 42

# broad class as named in the single model's hypothesis_label -> main models' folder
SINGLE_CLASS_FOLDER = {
    "Age and Family Status": "age_family_status",
    "Identities": "identities_minority_majority_status",
    "Labor Market Position": "labor_market_position",
    "Profession": "profession",
    "Real Estate Ownership": "real_estate_ownership",
    "Social Deviance": "social_deviance",
    "Social Roles and Behavior": "social_roles_behavior",
    "Socio-Economic Position": "socio_economic_position",
}
FOLDER_CODE = {folder: code for code, folder in sc.FOLDER_NAME_MAP.items()}

MODELS = ["main", "single"]
MODEL_LABEL = {"main": "One model per broad class (SOCCAT)", "single": MODEL_TITLE}
# the two colours/markers of the other paper comparisons (findings_* and cross-language figures)
MODEL_STYLE = {"main": dict(color="#C0392B", marker="o"), "single": dict(color="#909090", marker="s")}
OFFSET = 0.15
ALL_LABELS = "All labels"


# ── Data ────────────────────────────────────────────────────────────────────
def load_main() -> pd.DataFrame:
    cols = ["sentence_id", "hypothesis_label", "nli_label", "pred_label"]
    parts = []
    for folder in SINGLE_CLASS_FOLDER.values():
        for f in sorted((MAIN_DIR / folder).glob("fold_*_human_vs_model.csv")):
            parts.append(pd.read_csv(f, usecols=cols).assign(folder=folder))
    return pd.concat(parts).rename(columns={"hypothesis_label": "label"})


def load_single() -> pd.DataFrame:
    cols = ["sentence_id", "hypothesis_label", "nli_label", "pred_label"]
    d = pd.concat([pd.read_csv(SINGLE_DIR / f"fold_{k}_human_vs_model.csv", usecols=cols) for k in range(N_FOLDS)])
    parts = d["hypothesis_label"].str.split(" | ", n=1, regex=False)
    d["folder"], d["label"] = parts.str[0].map(SINGLE_CLASS_FOLDER), parts.str[1]
    assert d["folder"].notna().all(), "unknown broad class in the single model's labels"
    return d.drop(columns="hypothesis_label")


def aligned_arrays():
    """Gold Y and predictions P[model] as sentences x labels arrays (1 = entailment),
    with the (folder, label) of each column."""
    keys = ["folder", "label"]
    frames = {"main": load_main(), "single": load_single()}
    for m, d in frames.items():
        assert not d.duplicated(["sentence_id", *keys]).any(), f"{m}: a pair is predicted twice"
    cols = sorted(frames["main"][keys].drop_duplicates().itertuples(index=False, name=None))
    assert cols == sorted(frames["single"][keys].drop_duplicates().itertuples(index=False, name=None)), \
        "the two models do not cover the same labels"

    def pivot(d, value):
        w = d.assign(v=(d[value] == 0).astype(np.int8)).pivot(index="sentence_id", columns=keys, values="v")
        return w[cols]

    Y = pivot(frames["main"], "nli_label")
    assert not Y.isna().any().any(), "main: missing pairs"
    P = {m: pivot(d, "pred_label").loc[Y.index] for m, d in frames.items()}
    assert not P["single"].isna().any().any(), "single: missing pairs"
    Y_single = pivot(frames["single"], "nli_label").loc[Y.index]
    assert (Y_single.to_numpy() == Y.to_numpy()).all(), "the two models were evaluated on different gold labels"
    print(f"{len(Y):,} sentences x {len(cols)} labels; same sentences and gold labels for both models")
    return Y.to_numpy(), {m: p.to_numpy() for m, p in P.items()}, cols


# ── Scores ──────────────────────────────────────────────────────────────────
def scores(Y, P, W, groups):
    """{metric: [B or 1, labels + groups]} for weights W (B x sentences): per-label
    metrics, then one weighted average per group (list of column indices)."""
    TP, FP, FN, TN = Y & P, (1 - Y) & P, Y & (1 - P), (1 - Y) & (1 - P)
    tp, fp, fn, tn = (W @ M for M in (TP, FP, FN, TN))
    m = sc.metrics_from_counts(tp, fp, fn, tn)
    npos = tp + fn
    out = {}
    for metric, v in m.items():
        aggs = [(v[:, g] * npos[:, g]).sum(axis=1) / npos[:, g].sum(axis=1) for g in groups]
        out[metric] = np.column_stack([v, *aggs])
    out["n_pos"] = np.column_stack([npos, *[npos[:, g].sum(axis=1) for g in groups]])
    return out


def fold_f1(cols):
    """Mean / median over the 5 folds of each label's F1, for both models."""
    rows = []
    single = pd.concat([pd.read_csv(SINGLE_DIR / f"fold_{k}_per_label.csv") for k in range(N_FOLDS)])
    parts = single["hypothesis_label"].str.split(" | ", n=1, regex=False)
    single["folder"], single["label"] = parts.str[0].map(SINGLE_CLASS_FOLDER), parts.str[1]
    main = pd.concat([pd.read_csv(f).assign(folder=folder)
                      for folder in SINGLE_CLASS_FOLDER.values()
                      for f in sorted((MAIN_DIR / folder).glob("fold_*_per_label.csv"))])
    main = main.rename(columns={"hypothesis_label": "label"})
    for folder, label in cols:
        r = {}
        for m, d in (("main", main), ("single", single)):
            f1 = d[(d["folder"] == folder) & (d["label"] == label)]["f1_binary"]
            r[f"{m}_f1_fold_mean"], r[f"{m}_f1_fold_median"] = f1.mean(), f1.median()
        rows.append(r)
    return rows


def compute_results():
    Y, P, cols = aligned_arrays()
    folders = list(SINGLE_CLASS_FOLDER.values())
    groups = [[j for j, (f, _) in enumerate(cols) if f == folder] for folder in folders] + [list(range(len(cols)))]
    row_names = ([sc.RENAME_DICT.get(l, l) for _, l in cols]
                 + [sc.RENAME_DICT[FOLDER_CODE[f]] for f in folders] + [ALL_LABELS])
    row_broad = ([sc.RENAME_DICT[FOLDER_CODE[f]] for f, _ in cols]
                 + [sc.RENAME_DICT[FOLDER_CODE[f]] for f in folders] + [ALL_LABELS])
    kind = ["label"] * len(cols) + ["broad_class"] * len(folders) + ["all"]
    assert len(set(row_names)) == len(row_names), "row names are not unique"

    n = len(Y)
    rng = np.random.default_rng(SEED)
    W = rng.multinomial(n, np.full(n, 1 / n), size=N_BOOT)   # same resampled sentences for both models
    ones = np.ones((1, n), dtype=np.int64)
    point = {m: scores(Y, P[m], ones, groups) for m in MODELS}
    boot = {m: scores(Y, P[m], W, groups) for m in MODELS}

    folds = fold_f1(cols) + [{}] * (len(row_names) - len(cols))
    rows = []
    for metric in sc.METRICS:
        for j, name in enumerate(row_names):
            r = {"broad_class": row_broad[j], "label": name, "kind": kind[j], "metric": metric,
                 "n_pos": int(point["main"]["n_pos"][0, j])}
            for m in MODELS:
                r[m] = point[m][metric][0, j]
                r[f"{m}_lo"], r[f"{m}_hi"] = np.percentile(boot[m][metric][:, j], [2.5, 97.5])
            diff = boot["single"][metric][:, j] - boot["main"][metric][:, j]
            r["diff_single_main"] = r["single"] - r["main"]
            r["diff_lo"], r["diff_hi"] = np.percentile(diff, [2.5, 97.5])
            r["diff_ci_excludes_0"] = bool(r["diff_lo"] > 0 or r["diff_hi"] < 0)
            if metric == "f1_binary":
                r.update(folds[j])
            rows.append(r)
    return pd.DataFrame(rows)


# ── Figure ──────────────────────────────────────────────────────────────────
def render_figure(res, raw_labels, wrapped_labels, output_path, lbl_width_frac,
                  header_labels=frozenset(), extra_gap_rows=0.0):
    y = sc.compute_y_positions(raw_labels, header_labels, extra_gap_rows)
    n_rows_equiv = (y[0] - y[-1]) + 1
    fig_h_in = (sc.TOP_MARGIN_CM + sc.BOTTOM_MARGIN_CM + n_rows_equiv * sc.ROW_HEIGHT_CM) * sc.cm
    gs = gridspec.GridSpec(
        nrows=1, ncols=5, left=lbl_width_frac, right=0.98,
        bottom=sc.BOTTOM_MARGIN_CM * sc.cm / fig_h_in, top=1 - sc.TOP_MARGIN_CM * sc.cm / fig_h_in,
        wspace=0.15, width_ratios=[1, 1, 1, 1, 0.2],
    )
    fig = plt.figure(figsize=(sc.FIG_WIDTH_CM * sc.cm, fig_h_in))
    axs = [fig.add_subplot(gs[0, i]) for i in range(4)]
    ax_n = fig.add_subplot(gs[0, 4])
    by_label = res.set_index(["label", "metric"])

    for i, (ax, metric) in enumerate(zip(axs, sc.METRICS)):
        for lbl, yi in zip(raw_labels, y):
            r = by_label.loc[(lbl, metric)]
            for m, sign in zip(MODELS, (1, -1)):
                ax.errorbar(r[m], yi + sign * OFFSET, xerr=[[r[m] - r[f"{m}_lo"]], [r[f"{m}_hi"] - r[m]]],
                            fmt=MODEL_STYLE[m]["marker"], color=MODEL_STYLE[m]["color"],
                            markersize=3.5, markeredgecolor="white", markeredgewidth=0.4,
                            elinewidth=0.9, capsize=0, zorder=2)
        ax.set_ylim(min(y) - 0.5, max(y) + 0.5)
        ax.set_xlim(-0.03, 1.03)   # keeps scores of 0 / 1 visible
        ax.set_xticks([0, 0.5, 1])
        ax.set_xticklabels(["0", "0.5", "1"])
        ax.set_xticks([0.25, 0.75], minor=True)
        ax.grid(axis="x", linestyle="--", alpha=0.4)
        ax.grid(axis="x", which="minor", linestyle="--", alpha=0.4)
        ax.set_title(sc.METRIC_TITLE_MAP[metric])
        ax.set_yticks(y)
        if i == 0:
            ax.set_yticklabels(wrapped_labels)
        else:
            ax.tick_params(axis="y", which="both", labelleft=False, length=0)

    ax_n.set_title("N", fontsize=7)
    for lbl, yi in zip(raw_labels, y):
        ax_n.text(0.5, yi, f"{by_label.loc[(lbl, sc.METRICS[0]), 'n_pos']:,}", ha="center", va="center", fontsize=7)
    ax_n.set_ylim(min(y) - 0.5, max(y) + 0.5)
    ax_n.set_xlim(0, 1)
    ax_n.set_xticks([])
    ax_n.set_yticks([])
    for side in ax_n.spines.values():
        side.set_visible(False)

    handles = [Line2D([], [], linestyle="none", markersize=4, label=MODEL_LABEL[m], **MODEL_STYLE[m]) for m in MODELS]
    # left-aligned with the first panel, as in step_2_country.py
    fig.legend(handles=handles, loc="lower left", ncol=2, frameon=False, fontsize=7,
               borderpad=0, borderaxespad=0, handlelength=1.2, handletextpad=0.4, columnspacing=1.5,
               bbox_to_anchor=(axs[0].get_position().x0, sc.LEGEND_BOTTOM_CM * sc.cm / fig_h_in))
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Saved file as: {output_path}")


# ── Table ───────────────────────────────────────────────────────────────────
def build_latex_table(res, part_labels, header_labels, part_num, n_parts, with_total=False):
    f1 = res[res["metric"] == "f1_binary"].set_index("label")
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\scriptsize",
        r"\begin{threeparttable}",
        rf"\caption{{Step 2 binary F1 of {SINGLE_NLI_MODELS[MODEL]['caption']} and of one model per broad class (SOCCAT). Part {part_num}/{n_parts}.}}",
        rf"\label{{tab:{MODEL}_vs_main_part{part_num}}}",
        r"\begin{tabular}{l rr r r}",
        r"\toprule",
        r"Group & SOCCAT & Single & Single $-$ SOCCAT [95\% CI] & N \\",
        r"\midrule",
    ]

    def row(lbl, bold):
        r = f1.loc[lbl]
        name = sc.latex_escape(lbl[:1].upper() + lbl[1:])
        diff = f"{r['diff_single_main']:+.2f} [{r['diff_lo']:+.2f}, {r['diff_hi']:+.2f}]"
        if r["diff_ci_excludes_0"]:
            diff += r"$^{*}$"
        cells = [f"{r['main']:.2f}", f"{r['single']:.2f}", diff, f"{r['n_pos']:,}"]
        if bold:
            return rf"\textbf{{{name}}} & " + " & ".join(rf"\textbf{{{c}}}" for c in cells) + r" \\"
        return rf"\quad {name} & " + " & ".join(cells) + r" \\"

    for lbl in part_labels:
        if lbl in header_labels:
            if lbl != part_labels[0]:
                lines.append(r"\addlinespace")
            lines.append(row(lbl, bold=True))
        else:
            lines.append(row(lbl, bold=False))
    if with_total:
        lines += [r"\midrule", row(ALL_LABELS, bold=True)]
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\begin{tablenotes}[flushleft]",
        r"\footnotesize",
        r"\item \textit{Note:} Held-out predictions of the 5 cross-validation folds pooled, so that each "
        r"sentence is scored once; both models are evaluated on the same sentences and gold labels. "
        r"Bold rows: broad classes and all labels (average of their groups' F1, weighted by the number "
        r"of mentions). CI: paired 95\% bootstrap interval over sentences "
        rf"({N_BOOT:,} resamples). $^{{*}}$: the interval excludes 0. N: number of mentions.",
        r"\end{tablenotes}",
        r"\end{threeparttable}",
        r"\end{table}",
        "",
    ]
    return "\n".join(lines)


# ── Main ────────────────────────────────────────────────────────────────────
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    res = compute_results()
    res.to_csv(OUTPUT_DIR / f"{MODEL}_vs_main_metrics.csv", index=False)
    print(f"Saved file as: {OUTPUT_DIR / f'{MODEL}_vs_main_metrics.csv'}")

    # same row order and part split as step_2_boxplot.py / step_2_country.py
    broad_order, group_labels = sc.row_order()
    broad_names = [sc.RENAME_DICT[bc] for bc in broad_order]
    render_figure(res, broad_names + [ALL_LABELS],
                  [sc.wrap_and_capitalize(b, sc.WRAP_CHARS) for b in broad_names] + [rf"\textbf{{{ALL_LABELS}}}"],
                  OUTPUT_DIR / f"{MODEL}_vs_main_broad_class.pdf", sc.BROAD_LBL_WIDTH_FRAC)

    groups = [[sc.RENAME_DICT[bc]] + [sc.RENAME_DICT.get(l, l) for l in group_labels[bc]] for bc in broad_order]
    header_labels = set(broad_names)
    parts = sc.balanced_contiguous_split([len(g) for g in groups], sc.MERGED_N_PARTS)
    for part_num, idxs in enumerate(parts, start=1):
        part_labels = [lbl for i in idxs for lbl in groups[i]]
        wrapped = ["\n".join(rf"\textbf{{{line}}}" for line in sc.wrap_lines(lbl, sc.WRAP_CHARS))
                   if lbl in header_labels else sc.wrap_and_capitalize(lbl, sc.WRAP_CHARS)
                   for lbl in part_labels]
        render_figure(res, part_labels, wrapped, OUTPUT_DIR / f"{MODEL}_vs_main_part{part_num}.pdf",
                      sc.MERGED_LBL_WIDTH_FRAC, header_labels=header_labels,
                      extra_gap_rows=sc.MERGED_GROUP_SPACING_CM / sc.ROW_HEIGHT_CM)
        tex = build_latex_table(res, part_labels, header_labels, part_num, len(parts),
                                with_total=part_num == len(parts))
        (OUTPUT_DIR / f"{MODEL}_vs_main_table_part{part_num}.tex").write_text(tex, encoding="utf-8")
        print(f"Saved file as: {OUTPUT_DIR / f'{MODEL}_vs_main_table_part{part_num}.tex'}")

    # Summary
    lines = [f"{MODEL_TITLE} - one model per broad class (SOCCAT); paired bootstrap "
             f"({N_BOOT:,} resamples, seed {SEED}); * = 95% CI excludes 0", ""]
    for metric in sc.METRICS:
        sub = res[res["metric"] == metric]
        lab = sub[sub["kind"] == "label"]
        lines.append(f"{sc.METRIC_TITLE_MAP[metric]}: single better for {int((lab['diff_single_main'] > 0).sum())} "
                     f"of {len(lab)} labels, significantly for {int((lab['diff_ci_excludes_0'] & (lab['diff_single_main'] > 0)).sum())}; "
                     f"significantly worse for {int((lab['diff_ci_excludes_0'] & (lab['diff_single_main'] < 0)).sum())}")
        for _, r in sub[sub["kind"] != "label"].iterrows():
            star = "*" if r["diff_ci_excludes_0"] else " "
            lines.append(f"  {r['label']:<55} SOCCAT {r['main']:.3f}  single {r['single']:.3f}  "
                         f"diff {r['diff_single_main']:+.3f} [{r['diff_lo']:+.3f}, {r['diff_hi']:+.3f}]{star}")
        sig = lab[lab["diff_ci_excludes_0"]].sort_values("diff_single_main")
        if len(sig):
            lines.append("  labels with a significant difference:")
            for _, r in sig.iterrows():
                lines.append(f"    {r['label']:<55} SOCCAT {r['main']:.2f}  single {r['single']:.2f}  "
                             f"diff {r['diff_single_main']:+.2f} [{r['diff_lo']:+.2f}, {r['diff_hi']:+.2f}]  N {r['n_pos']}")
        lines.append("")
    text = "\n".join(lines)
    (OUTPUT_DIR / f"{MODEL}_vs_main_summary.txt").write_text(text, encoding="utf-8")
    print("\n" + text)


if __name__ == "__main__":
    main()
