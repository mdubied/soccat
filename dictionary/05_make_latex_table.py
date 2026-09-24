#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build LaTeX tables comparing dictionary-baseline alpha0/z-threshold variants
(see 04_run_alpha0_sweep.py) against SOCCAT's own published NLI models.
=============================================================================
Rows: 57 specific categories, grouped by broad category, split into two
tables (part 1 = first 4 broad categories, part 2 = last 4).

Two variants (see build_table's `metrics` param): primary (F1 only, fits a
normal page) and a full P/R/F1 backup (wide, reference/appendix use only).

SOCCAT columns are read directly from data/model_performance/step_2/<folder>/
fold_*_per_label.csv, not recomputed. Three variants per label (see
load_soccat_metrics): "best fold" (the one actually published), "mean" and
"median" across all 5 CV folds (best-fold alone isn't a fair comparison
against the dictionaries' single train/test split; mean/median remove that
asymmetry).

Requires 04_run_alpha0_sweep.py to have been run first.
Output: dictionary/output/alpha0_sweep_table_part{1,2}.tex       (F1 only)
        dictionary/output/alpha0_sweep_table_full_part{1,2}.tex  (P/R/F1)
Compile with `booktabs` and `threeparttable`.
"""

import json
import math
import re
import sys
from importlib import import_module
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
z_suffix = import_module("02_build_dictionary").z_suffix

OUTPUT_DIR = ROOT / "output"
CATEGORIES_FILE = ROOT.parent / "src" / "step_2" / "categories.json"
STEP_2_DATA_DIR = ROOT.parent / "data" / "model_performance" / "step_2"

MODES = ["low", "medium", "high"]
MODE_LABELS = {"low": "low", "medium": "med", "high": "high"}  # display only
Z_THRESHOLDS = [0.0, 1.96]
Z_LABELS = {0.0: "$z>0$", 1.96: "$z>1.96$"}
LABEL_LINE_WRAP_THRESHOLD = 24  # chars; labels longer than this get split onto 2 lines
LABEL_LINE_GAP = "-3pt"  # tighten \shortstack's default (too loose) inter-line gap
BROAD_CLASS_LINE_GAP = "3pt"  # extra gap between a broad-class name and its first label
GROUP_GAP_HEIGHT = "5pt"  # height of the blank spacer row before each broad class
ARRAY_STRETCH = "0.9"  # \arraystretch factor -- <1 tightens overall row spacing
DECIMALS = 2

# broad category (categories.json "name") -> its own folder directly under
# STEP_2_DATA_DIR, containing fold_*_per_label.csv (one file per fold).
# Matches figures/step_2_boxplot.py's FOLDER_NAME_MAP.
FOLDER_NAME_MAP = {
    "socio_economic_position":   "socio_economic_position",
    "labor_market_position":     "labor_market_position",
    "age_and_family_status":     "age_family_status",
    "identities":                "identities_minority_majority_status",
    "profession":                "profession",
    "social_roles_and_behavior": "social_roles_behavior",
    "social_deviance":           "social_deviance",
    "real_estate_ownership":     "real_estate_ownership",
}

# fold-selection metric -- n_pos_entail-weighted macro F1, matching
# figures/step_2_boxplot.py and tables/step_2_best_fold_table.py (chosen
# there for robustness to the small positive-case counts that make
# per-label f1_binary noisy fold-to-fold).
BEST_FOLD_METRIC = "f1_macro"

# legacy hypothesis_label (as used in the old *_per_fold.csv files) ->
# canonical taxonomy label. All currently empty: the new fold_*_per_label.csv
# files already use canonical label spellings throughout (verified against
# src/step_2/categories.json), so no remapping is needed today -- kept as a
# hook in case a future data refresh reintroduces spelling drift.
SOCCAT_LABEL_REMAP = {
    "age_and_family_status": {},
    "identities": {},
    "labor_market_position": {},
    "profession": {},
    "real_estate_ownership": {},
    "social_deviance": {},
    "social_roles_and_behavior": {},
    "socio_economic_position": {},
}


def escape_latex(text: str) -> str:
    replacements = {
        "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
        "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}", "\\": r"\textbackslash{}",
    }
    for char, repl in replacements.items():
        text = text.replace(char, repl)
    return text


# manual line breaks for specific labels where even the 2-line auto-split
# (see wrap_label) is still too wide for the column
MANUAL_LABEL_BREAKS = {
    "Terrorists, rebels, revolutionaries and/or movements of armed resistance": [
        "Terrorists, rebels,",
        "revolutionaries and/or",
        "movements of armed resistance",
    ],
}


def wrap_label(text: str) -> str:
    """Split long label names onto balanced lines via \\shortstack."""
    text = text[:1].upper() + text[1:]
    if text in MANUAL_LABEL_BREAKS:
        parts = [escape_latex(p) for p in MANUAL_LABEL_BREAKS[text]]
        joined = f"\\\\[{LABEL_LINE_GAP}]".join(parts)
        return f"\\shortstack[l]{{{joined}}}"
    text = escape_latex(text)
    if len(text) <= LABEL_LINE_WRAP_THRESHOLD:
        return text
    spaces = [i for i, c in enumerate(text) if c == " "]
    if not spaces:
        return text
    mid = len(text) / 2
    split_at = min(spaces, key=lambda i: abs(i - mid))
    line1, line2 = text[:split_at].strip(), text[split_at:].strip()
    return f"\\shortstack[l]{{{line1}\\\\[{LABEL_LINE_GAP}]{line2}}}"


def fmt(value) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "--"
    return f"{value:.{DECIMALS}f}"


def load_broad_class_df(broad_category: str) -> pd.DataFrame:
    """All 5 folds' per-label rows for one broad category (fold_*_per_label.csv),
    concatenated with a "fold" column added (mirrors figures/step_2_boxplot.py's
    load_broad_class_df)."""
    folder = FOLDER_NAME_MAP[broad_category]
    fold_files = sorted(
        (STEP_2_DATA_DIR / folder).glob("fold_*_per_label.csv"),
        key=lambda p: int(re.search(r"fold_(\d+)_per_label", p.name).group(1)),
    )
    assert fold_files, f"No fold_*_per_label.csv found for '{broad_category}' in {STEP_2_DATA_DIR / folder}"
    dfs = []
    for f in fold_files:
        fold_num = int(re.search(r"fold_(\d+)_per_label", f.name).group(1))
        d = pd.read_csv(f)
        d["fold"] = fold_num
        dfs.append(d)
    return pd.concat(dfs, ignore_index=True)


def best_fold_for(df: pd.DataFrame) -> int:
    """Fold with the highest n_pos_entail-weighted BEST_FOLD_METRIC across categories."""
    fold_scores = df.groupby("fold").apply(
        lambda g: np.average(g[BEST_FOLD_METRIC], weights=g["n_pos_entail"]),
        include_groups=False,
    )
    return int(fold_scores.idxmax())


SOCCAT_VARIANTS = ["best", "mean", "median"]
# short titles -- "SOCCAT" itself is now the shared super-header spanning all
# three (see build_table), so these only need to distinguish the variant.
SOCCAT_VARIANT_TITLES = {
    "best": "best fold",
    "mean": "mean",
    "median": "med",
}


def load_soccat_metrics(broad_category: str) -> dict:
    """Returns {label: {"best"/"mean"/"median": {precision, recall, f1}}}.
    "best" is read at the automatically selected best fold (highest
    n_pos_entail-weighted macro F1); "mean"/"median" across all 5 CV folds."""
    df = load_broad_class_df(broad_category)
    fold = best_fold_for(df)
    remap = SOCCAT_LABEL_REMAP[broad_category]
    out = {}
    for label_raw, g in df.groupby("hypothesis_label"):
        label = remap.get(label_raw, label_raw)
        best_row = g.loc[g["fold"] == fold]
        out[label] = {
            "best": {
                "precision": best_row["precision_binary"].iloc[0] if len(best_row) else float("nan"),
                "recall": best_row["recall_binary"].iloc[0] if len(best_row) else float("nan"),
                "f1": best_row["f1_binary"].iloc[0] if len(best_row) else float("nan"),
            },
            "mean": {
                "precision": g["precision_binary"].mean(),
                "recall": g["recall_binary"].mean(),
                "f1": g["f1_binary"].mean(),
            },
            "median": {
                "precision": g["precision_binary"].median(),
                "recall": g["recall_binary"].median(),
                "f1": g["f1_binary"].median(),
            },
        }
    return out


METRIC_TITLES = {"precision": "P", "recall": "R", "f1": "F1"}
METRIC_NAMES = {"precision": "precision (P)", "recall": "recall (R)", "f1": "F1 score"}


def build_table(categories_subset, part_num, n_parts, metrics,
                 super_groups, dict_lookup, soccat_variants=SOCCAT_VARIANTS,
                 label_suffix=""):
    """
    metrics: subset/order of ["precision", "recall", "f1"] to show per leaf.
    super_groups: list of (super_label, [(key, leaf_label), ...]) -- top
    header groups leaves under super_label; second row shows each leaf_label.
    dict_lookup: {key: {(category, label): row}}. Keys in SOCCAT_VARIANTS
    are read from that category's soccat dict instead of dict_lookup.

    Third header row (P/R/F1) only added when metrics has more than one entry.
    """
    n_cols = len(metrics)
    leaves = [(key, leaf_label) for _, group_leaves in super_groups for key, leaf_label in group_leaves]
    n_groups = len(leaves)
    lines = []
    lines.append(r"\begin{table}[htb]")
    lines.append(r"\centering")
    lines.append(rf"\renewcommand{{\arraystretch}}{{{ARRAY_STRETCH}}}")
    lines.append(r"\scriptsize")
    lines.append(r"\begin{threeparttable}")
    lines.append(rf"\caption{{Comparison with dictionary-based classifiers. "
                 rf"Part {part_num}/{n_parts}.}}")
    lines.append(rf"\label{{tab:dictionary_alpha0_sweep{label_suffix}_part{part_num}}}")
    lines.append(r"\begin{tabular}{l" + (" " + "c" * n_cols) * n_groups + "}")
    lines.append(r"\toprule")

    header1 = [""]
    col = 2
    cmidrules1 = []
    for super_label, group_leaves in super_groups:
        span = len(group_leaves) * n_cols
        header1.append(rf"\multicolumn{{{span}}}{{c}}{{{super_label}}}")
        cmidrules1.append(rf"\cmidrule(lr){{{col}-{col + span - 1}}}")
        col += span
    lines.append(" & ".join(header1) + r" \\")
    lines.append(" ".join(cmidrules1))

    header2 = ["Specific category" if len(metrics) == 1 else ""]
    for _, leaf_label in leaves:
        header2.append(rf"\multicolumn{{{n_cols}}}{{c}}{{{leaf_label}}}")
    lines.append(" & ".join(header2) + r" \\")

    cmidrules2 = " ".join(
        rf"\cmidrule(lr){{{2 + n_cols*i}-{1 + n_cols*(i + 1)}}}" for i in range(n_groups))
    lines.append(cmidrules2)

    if len(metrics) > 1:
        header3 = ["Specific category"] + [METRIC_TITLES[m] for m in metrics] * n_groups
        lines.append(" & ".join(header3) + r" \\")
    lines.append(r"\midrule")

    # A blank spacer ROW (not \addlinespace) goes before every broad class,
    # including the first -- being an actual row, a template's per-row
    # shading colors it too, instead of leaving a plain white gap.
    total_cols = 1 + n_groups * n_cols
    for cat in categories_subset:
        soccat = load_soccat_metrics(cat["name"])
        blank_row = [rf"\rule{{0pt}}{{{GROUP_GAP_HEIGHT}}}"] + [""] * (total_cols - 1)
        lines.append(" & ".join(blank_row) + r" \\")
        for j, label in enumerate(cat["labels"]):
            label_cell = wrap_label(label)
            if j == 0:
                prefix = rf"\textbf{{{escape_latex(cat['display_name'])}}}"
                if label_cell.startswith(r"\shortstack[l]{"):
                    label_cell = label_cell.replace(
                        r"\shortstack[l]{", rf"\shortstack[l]{{{prefix}\\[{BROAD_CLASS_LINE_GAP}]", 1)
                else:
                    label_cell = rf"\shortstack[l]{{{prefix}\\[{BROAD_CLASS_LINE_GAP}]{label_cell}}}"
            row = [label_cell]
            for key, _ in leaves:
                if key in soccat_variants:
                    m = soccat.get(label, {}).get(key, {})
                else:
                    m = dict_lookup[key].get((cat["name"], label), {})
                row += [fmt(m[metric]) if metric in m else "--" for metric in metrics]
            lines.append(" & ".join(row) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\begin{tablenotes}[flushleft]")
    lines.append(r"\footnotesize")
    if len(metrics) == 1:
        metric_desc = METRIC_NAMES[metrics[0]]
    else:
        metric_desc = ", ".join(METRIC_NAMES[m] for m in metrics[:-1]) + f", and {METRIC_NAMES[metrics[-1]]}"
    lines.append(
        rf"\item \textit{{Note:}} Reported scores are the binary, positive-class {metric_desc}. "
        r"$\alpha_0$ and $z$ are model parameters that yield different dictionary word lists. "
        r"SOCCAT ``best fold''/``mean''/``median (med)'' are computed across the 5 cross-validation folds."
    )
    lines.append(r"\end{tablenotes}")
    lines.append(r"\end{threeparttable}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def main():
    with open(CATEGORIES_FILE, encoding="utf-8") as f:
        categories = json.load(f)

    # Load every (alpha0-mode, z-threshold) variant, keyed by "<mode>_<zsuffix>".
    dict_lookup_all = {}
    for mode in MODES:
        for z in Z_THRESHOLDS:
            variant = f"{mode}_{z_suffix(z)}"
            df = pd.read_csv(OUTPUT_DIR / variant / "per_specific_category_metrics.csv")
            dict_lookup_all[variant] = {(r["category"], r["label"]): r for _, r in df.iterrows()}

    # Split after 4 broad categories -- closest to an even row split.
    split_idx = 4
    parts = [categories[:split_idx], categories[split_idx:]]

    # Primary table: F1 only, one super-group per alpha0 mode (leaves just
    # say "z>0"/"z>1.96") plus one for SOCCAT. Both z-thresholds are shown
    # since the choice isn't negligible (see 02_build_dictionary.py).
    main_super_groups = [
        (rf"Dict., $\alpha_0$={MODE_LABELS[mode]}", [(f"{mode}_{z_suffix(z)}", Z_LABELS[z]) for z in Z_THRESHOLDS])
        for mode in MODES
    ] + [
        ("SOCCAT", [(v, SOCCAT_VARIANT_TITLES[v]) for v in SOCCAT_VARIANTS])
    ]
    for i, subset in enumerate(parts, start=1):
        table_tex = build_table(subset, i, len(parts), metrics=["f1"],
                                 super_groups=main_super_groups, dict_lookup=dict_lookup_all)
        out_path = OUTPUT_DIR / f"alpha0_sweep_table_part{i}.tex"
        out_path.write_text(table_tex, encoding="utf-8")
        print(f"Done. LaTeX table written to: {out_path}")

    # Backup table: full P/R/F1, single z-threshold (1.96) -- wide, reference only.
    full_super_groups = [
        (rf"Dict., $\alpha_0$={MODE_LABELS[mode]}", [(f"{mode}_{z_suffix(1.96)}", "")])
        for mode in MODES
    ] + [
        ("SOCCAT", [(v, SOCCAT_VARIANT_TITLES[v]) for v in SOCCAT_VARIANTS])
    ]
    for i, subset in enumerate(parts, start=1):
        table_tex = build_table(subset, i, len(parts),
                                 metrics=["precision", "recall", "f1"],
                                 super_groups=full_super_groups, dict_lookup=dict_lookup_all,
                                 label_suffix="_full")
        out_path = OUTPUT_DIR / f"alpha0_sweep_table_full_part{i}.tex"
        out_path.write_text(table_tex, encoding="utf-8")
        print(f"Done. LaTeX table written to: {out_path}")

    print("Requires \\usepackage{booktabs} and \\usepackage{threeparttable} "
          "-- \\shortstack needs no extra package.")


if __name__ == "__main__":
    main()
