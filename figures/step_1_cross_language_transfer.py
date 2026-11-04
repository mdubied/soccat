"""
step_1_cross_language_transfer.py

Description:
Figure for the cross-lingual transfer analysis of the Step 1 model
(social group mention yes/no).

Reads the per-run predictions in ../cross-language_transfer/step_1/detailed_results/
(*__predictions.csv, written by CV_cross_lingual_step1.ipynb) and computes the
Step 1 metrics with 95% bootstrap CIs (resampled sentences, as in the notebook).
Three settings are compared, all predicting every sentence of the target
language exactly once (5 folds):
  - within_matched: trained on the target language,
  - cross_matched:  trained on the other language,
    with the training set size (sentences and positives) matched between
    these two settings,
  - pooled_full:    regular cross-validation on both languages (all training
    data; not the released model, which uses the original 80/20 split).
The folds pool the original train and test files (folds.csv), so all
sentences of each language are test sentences once.

Figure:
Accuracy, precision, recall, F1 and macro F1 (threshold 0.5) per setting, one
panel per test set (same panel order, colors and markers as
step_2_cross_language_transfer.py, plus black diamonds for the pooled setting).
Point: score on the pooled test folds; bar: 95% bootstrap CI. Side column:
within minus cross-language gap, with a star when its paired 95% bootstrap CI
excludes 0 (both settings predict the same sentences, so each draw resamples
sentences and recomputes both scores).

Outputs (in "step_1/cross_language/"):
- step_1_cross_language_performance.pdf
- step_1_cross_language_summary.txt (numbers behind the figure and the text)

Usage (from this directory):
python step_1_cross_language_transfer.py
"""
import os, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D

# ============================================================
# PARAMETERS (edit here)
# ============================================================
PRED_DIR = "../cross-language_transfer/step_1/detailed_results"
OUTPUT_DIR = "step_1/cross_language"

SEED = 98        # as in the notebook
N_BOOT = 2000

DIRS = ["de2fr", "fr2de"]   # panel order: French test set, then German
DIR_TITLE = {
    "fr2de": "Trained on French, tested on German",
    "de2fr": "Trained on German, tested on French",
}
# panel titles: all settings of a panel are evaluated on the same test sentences
# (DIR_TITLE is the cross-language model's direction, used in the summary text)
PANEL_TITLE = {"fr2de": "German test set", "de2fr": "French test set"}
CONDITIONS = ["within_matched", "cross_matched", "pooled_full"]
COND_LABEL = {
    "within_matched": "Within language (trained on the test language)",
    "cross_matched": "Cross-language (trained on the other language)",
    "pooled_full": "Regular CV, both languages (all training data)",
}
# within/cross: the two colors/markers of the findings_* figures (utils.py),
# as in step_2_cross_language_transfer.py
COND_STYLE = {
    "within_matched": dict(marker="o", color="#C0392B", offset=+0.22),
    "cross_matched": dict(marker="s", color="#909090", offset=0.0),
    "pooled_full": dict(marker="D", color="#000000", offset=-0.22),
}
# (metric key, display name), top to bottom: the Step 1 metrics
METRICS = [
    ("accuracy", "Accuracy"),
    ("precision", "Precision"),
    ("recall", "Recall"),
    ("f1", "F1"),
    ("macro_f1", "Macro F1"),
]

# Star next to the gap when the paired 95% CI of the within minus
# cross-language difference excludes 0
SHOW_SIGNIFICANCE = True

# Figure parameters (width and margins as in step_2_cross_language_transfer.py)
FIG_WIDTH_CM = 14.0
ROW_HEIGHT_CM = 0.9         # one row holds the three settings
TOP_MARGIN_CM = 0.5
BOTTOM_MARGIN_CM = 2.2      # room for the x label and the two-row legend
LBL_WIDTH_FRAC = 0.12
X_MIN = 0.5                 # all scores and CIs are above 0.5 (0 = as in step 2)

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
# METRICS
# ============================================================

def scores(y, pred):
    """Step 1 metrics of binary labels and predictions (as sklearn, zero_division=0)."""
    tp = int((y & pred).sum()); fp = int((~y & pred).sum()); fn = int((y & ~pred).sum())
    tn = len(y) - tp - fp - fn
    f1 = lambda a, b, c: 2 * a / (2 * a + b + c) if a else 0.0
    return {
        "accuracy": (tp + tn) / len(y),
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "f1": f1(tp, fp, fn),
        "macro_f1": (f1(tp, fp, fn) + f1(tn, fn, fp)) / 2,
        "n": len(y), "n_pos": tp + fn, "n_pred_pos": tp + fp,
    }

def bootstrap(y, preds):
    """95% percentile CIs of each metric of each prediction vector in preds,
    and of the difference preds[0] - preds[1], resampling the same sentences
    for all of them on each draw."""
    r = np.random.RandomState(SEED)
    draws = {j: {k: [] for k, _ in METRICS} for j in range(len(preds))}
    for _ in range(N_BOOT):
        i = r.randint(0, len(y), len(y))
        for j, p in enumerate(preds):
            s = scores(y[i], p[i])
            for k, _ in METRICS:
                draws[j][k].append(s[k])
    ci = lambda v: np.percentile(v, [2.5, 97.5])
    out = [{f"{k}_{b}": v for k, _ in METRICS for b, v in zip(("lo", "hi"), ci(draws[j][k]))}
           for j in range(len(preds))]
    if len(preds) == 2:
        out.append({f"{k}_gap_{b}": v for k, _ in METRICS
                    for b, v in zip(("lo", "hi"), ci(np.subtract(draws[0][k], draws[1][k])))})
    return out

def style_score_axis(ax):
    """As step_2_cross_language_transfer.py's style_f1_axis, from X_MIN to 1."""
    pad = 0.03 * (1 - X_MIN)   # keeps markers at the limits visible
    ax.set_xlim(X_MIN - pad, 1 + pad)
    mid = (X_MIN + 1) / 2
    ax.set_xticks([X_MIN, mid, 1])
    ax.set_xticklabels([f"{X_MIN:g}", f"{mid:g}", "1"])
    ax.set_xticks([(X_MIN + mid) / 2, (mid + 1) / 2], minor=True)
    lo = min(res[f"{k}_lo"].min() for k, _ in METRICS)
    assert lo >= X_MIN, f"A CI starts at {lo:.3f}, below X_MIN = {X_MIN}"
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    ax.grid(axis="x", which="minor", linestyle="--", alpha=0.4)

def style_gap_axis(ax, ylim):
    ax.set_title("Gap")
    ax.set_ylim(*ylim)
    ax.set_xlim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)

def fmt_gap(v):
    return f"$-${abs(v):.2f}" if v < 0 else f"{v:.2f}"

def significant(g, k):
    return g[f"{k}_gap_lo"] > 0 or g[f"{k}_gap_hi"] < 0

# ============================================================
# LOAD PREDICTIONS AND COMPUTE METRICS
# ============================================================
os.makedirs(OUTPUT_DIR, exist_ok=True)

files = sorted(glob.glob(f"{PRED_DIR}/*__predictions.csv"))
assert files, f"No *__predictions.csv in {PRED_DIR}"
P = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

res, gaps = {}, {}
for direction in DIRS:
    tgt = direction[-2:]
    sets = {c: P[(P.direction == direction) & (P.condition == c)] for c in ["within_matched", "cross_matched", "cross_full"]}
    sets["pooled_full"] = P[(P.condition == "pooled_full") & (P.lang == tgt)]
    # all settings predict the same target sentences, each exactly once
    ids = sorted(sets["within_matched"].sentence_id)
    for c, d in sets.items():
        assert d.sentence_id.is_unique and sorted(d.sentence_id) == ids, f"{direction} {c}: test sentences differ"
    sets = {c: d.set_index("sentence_id").loc[ids] for c, d in sets.items()}
    y = sets["within_matched"]["label"].to_numpy(bool)
    pred = {c: (d["prob_group"] > 0.5).to_numpy() for c, d in sets.items()}

    for c in sets:
        if c not in ("within_matched", "cross_matched"):
            res[(direction, c)] = {**scores(y, pred[c]), **bootstrap(y, [pred[c]])[0]}
    ci_w, ci_c, ci_gap = bootstrap(y, [pred["within_matched"], pred["cross_matched"]])
    res[(direction, "within_matched")] = {**scores(y, pred["within_matched"]), **ci_w}
    res[(direction, "cross_matched")] = {**scores(y, pred["cross_matched"]), **ci_c}
    w, c = res[(direction, "within_matched")], res[(direction, "cross_matched")]
    gaps[direction] = {**{f"{k}_gap": w[k] - c[k] for k, _ in METRICS}, **ci_gap, "retention": c["f1"] / w["f1"]}

res = pd.DataFrame.from_dict(res, orient="index")
gaps = pd.DataFrame.from_dict(gaps, orient="index")

# ============================================================
# FIGURE: SCORES PER SETTING AND TEST SET
# ============================================================
y_pos = -np.arange(len(METRICS), dtype=float)
ylim = (y_pos.min() - 0.5, y_pos.max() + 0.5)

fig_h_in = (TOP_MARGIN_CM + BOTTOM_MARGIN_CM + len(METRICS) * ROW_HEIGHT_CM) * cm
fig = plt.figure(figsize=(FIG_WIDTH_CM * cm, fig_h_in))
gs = gridspec.GridSpec(
    nrows=1, ncols=4, left=LBL_WIDTH_FRAC, right=0.98,
    bottom=BOTTOM_MARGIN_CM * cm / fig_h_in, top=1 - TOP_MARGIN_CM * cm / fig_h_in,
    wspace=0.08, width_ratios=[1, 0.2, 1, 0.2],
)

for k, direction in enumerate(DIRS):
    ax = fig.add_subplot(gs[0, 2 * k])
    ax_g = fig.add_subplot(gs[0, 2 * k + 1])
    g = gaps.loc[direction]
    for (key, _), yi in zip(METRICS, y_pos):
        for cond in CONDITIONS:
            r = res.loc[(direction, cond)]
            v, lo, hi = r[key], r[f"{key}_lo"], r[f"{key}_hi"]
            st = COND_STYLE[cond]
            ax.errorbar(v, yi + st["offset"], xerr=[[v - lo], [hi - v]], fmt=st["marker"],
                        color=st["color"], ms=3, lw=0.8, capsize=0, zorder=3)
        star = r"$^{*}$" if SHOW_SIGNIFICANCE and significant(g, key) else r"$^{\phantom{*}}$"
        ax_g.text(0.5, yi, fmt_gap(g[f"{key}_gap"]) + star, ha="center", va="center", fontsize=7)
    style_score_axis(ax)
    ax.set_ylim(*ylim)
    n, n_pos = int(res.loc[(direction, "within_matched"), "n"]), int(res.loc[(direction, "within_matched"), "n_pos"])
    ax.set_title(PANEL_TITLE[direction])
    ax.set_xlabel(f"Score ({n_pos:,} positive of {n:,} sentences)".replace(",", "{,}"))
    ax.set_yticks(y_pos)
    if k == 0:
        ax.set_yticklabels([name for _, name in METRICS])
    else:
        ax.tick_params(axis="y", which="both", labelleft=False, length=0)
    style_gap_axis(ax_g, ylim)

handles = [Line2D([], [], ls="", marker=COND_STYLE[c]["marker"], color=COND_STYLE[c]["color"],
                  ms=4, label=COND_LABEL[c]) for c in CONDITIONS]
if SHOW_SIGNIFICANCE:
    handles.append(Line2D([], [], ls="", marker=r"$*$", color="black", ms=4,
                          label=r"Gap (within $-$ cross-language): significant (paired 95\% CI)"))
fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.0))
out = os.path.join(OUTPUT_DIR, "step_1_cross_language_performance.pdf")
fig.savefig(out)
plt.close(fig)
print(f"Saved file as: {out}")

# ============================================================
# SUMMARY (numbers behind the figure and the text)
# ============================================================
summary = ["NUMBERS IN THE TEXT (within_matched vs cross_matched, paired 95% bootstrap CI)", "=" * 84]
for direction in DIRS:
    g, rw, rc = gaps.loc[direction], res.loc[(direction, "within_matched")], res.loc[(direction, "cross_matched")]
    summary.append(f"{DIR_TITLE[direction]}:")
    summary.append(f"  F1 {rc.f1:.2f} (cross) vs {rw.f1:.2f} (within); gap {g.f1_gap:.2f} "
                   f"[{g.f1_gap_lo:.2f}, {g.f1_gap_hi:.2f}]; {g.retention:.0%} of within-language F1 retained")
    summary.append(f"  Macro F1 {rc.macro_f1:.2f} vs {rw.macro_f1:.2f}; gap {g.macro_f1_gap:.2f} "
                   f"[{g.macro_f1_gap_lo:.2f}, {g.macro_f1_gap_hi:.2f}]")
    summary.append(f"  Precision {rc.precision:.2f} vs {rw.precision:.2f}, recall {rc.recall:.2f} vs {rw.recall:.2f}; "
                   f"predicted positives per true positive (threshold 0.5): "
                   f"{rc.n_pred_pos / rc.n_pos:.2f} (cross) vs {rw.n_pred_pos / rw.n_pos:.2f} (within)")
summary.append("")

summary.append(f"SCORES PER TEST SET AND SETTING (pooled over the 5 test folds, 95% bootstrap CI, {N_BOOT} draws)")
summary.append("=" * 84)
for direction in DIRS:
    r0 = res.loc[(direction, "within_matched")]
    summary.append(f"{PANEL_TITLE[direction]} ({int(r0.n_pos)} positive of {int(r0.n)} sentences)")
    summary.append(f"  {'Setting':<16} " + " ".join(f"{name:>20}" for _, name in METRICS))
    for cond in ["within_matched", "cross_matched", "cross_full", "pooled_full"]:
        r = res.loc[(direction, cond)]
        summary.append(f"  {cond:<16} " + " ".join(
            f"{r[k]:>6.3f} [{r[f'{k}_lo']:.3f}, {r[f'{k}_hi']:.3f}]" for k, _ in METRICS))
    summary.append("")

summary.append("GAP WITHIN_MATCHED MINUS CROSS_MATCHED (paired 95% bootstrap CI; * = CI excludes 0)")
summary.append("=" * 84)
for direction in DIRS:
    g = gaps.loc[direction]
    summary.append(f"{DIR_TITLE[direction]} ({PANEL_TITLE[direction]})")
    for k, name in METRICS:
        summary.append(f"  {name:<10} {g[f'{k}_gap']:>7.3f} [{g[f'{k}_gap_lo']:>6.3f}, {g[f'{k}_gap_hi']:>6.3f}]"
                       f"{'  *' if significant(g, k) else ''}")

out = os.path.join(OUTPUT_DIR, "step_1_cross_language_summary.txt")
with open(out, "w", encoding="utf-8") as f:
    f.write("\n".join(summary) + "\n")
print("\n".join(summary))
print(f"Saved file as: {out}")
