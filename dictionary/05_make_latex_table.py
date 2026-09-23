#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build LaTeX tables of per-specific-category precision/recall/F1: the three
dictionary-baseline alpha0 modes (see 04_run_alpha0_sweep.py) plus SOCCAT's
own published NLI models, side by side
=============================================================================
Rows: the 57 specific categories (labels), in taxonomy order (grouped by
broad category, with a thin rule between groups for readability), split
into two tables (part 1 = first 4 broad categories, part 2 = last 4) so
each fits on its own page -- 13 columns (label + 4 metric groups x P/R/F1)
was too wide for one.

Columns: precision/recall/F1, once per dictionary alpha0 mode (low/medium/
high) plus once for SOCCAT, each group under its own merged (\\multicolumn)
header.

SOCCAT columns: read directly from the already-computed per-fold, per-label
metrics in data/model_performance/step_2/<folder>/fold_*_per_label.csv (one
file per fold; columns include hypothesis_label, precision_binary,
recall_binary, f1_binary, f1_macro, n_pos_entail) -- NOT recomputed here.
The "best" fold (whose model is the one published) is picked automatically,
same rule as figures/step_2_boxplot.py and tables/step_2_best_fold_table.py:
the fold with the highest n_pos_entail-weighted macro F1 across that broad
category's labels -- robust to the small positive-case counts that make
per-label f1_binary noisy fold-to-fold. See FOLDER_NAME_MAP (which folder
per broad category) and SOCCAT_LABEL_REMAP (only needed if a future data
refresh reintroduces non-canonical label spellings; currently all entries
are no-ops since the new files already use canonical taxonomy label names).

Long label names are split onto two lines within their cell using
\\shortstack (plain LaTeX, no extra package needed), balanced at the space
closest to the middle of the string.

Requires 04_run_alpha0_sweep.py to have been run first.

Output: dictionary/output/alpha0_sweep_table_part1.tex
        dictionary/output/alpha0_sweep_table_part2.tex
Compile with the `booktabs` package (\\toprule/\\midrule/\\bottomrule/\\cmidrule).
"""

import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "output"
CATEGORIES_FILE = ROOT.parent / "src" / "step_2" / "categories.json"
STEP_2_DATA_DIR = ROOT.parent / "data" / "model_performance" / "step_2"

MODES = ["low", "medium", "high"]
LABEL_LINE_WRAP_THRESHOLD = 24  # chars; labels longer than this get split onto 2 lines
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


def wrap_label(text: str) -> str:
    """Split long label names onto two balanced lines via \\shortstack."""
    text = text[:1].upper() + text[1:]
    text = escape_latex(text)
    if len(text) <= LABEL_LINE_WRAP_THRESHOLD:
        return text
    spaces = [i for i, c in enumerate(text) if c == " "]
    if not spaces:
        return text
    mid = len(text) / 2
    split_at = min(spaces, key=lambda i: abs(i - mid))
    line1, line2 = text[:split_at].strip(), text[split_at:].strip()
    return f"\\shortstack[l]{{{line1}\\\\{line2}}}"


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


def load_soccat_metrics(broad_category: str) -> dict:
    """Returns {canonical_label: {precision, recall, f1}} for the automatically
    selected best fold (highest n_pos_entail-weighted macro F1)."""
    df = load_broad_class_df(broad_category)
    fold = best_fold_for(df)
    df = df[df["fold"] == fold]
    remap = SOCCAT_LABEL_REMAP[broad_category]
    out = {}
    for _, r in df.iterrows():
        label = remap.get(r["hypothesis_label"], r["hypothesis_label"])
        out[label] = {
            "precision": r["precision_binary"],
            "recall": r["recall_binary"],
            "f1": r["f1_binary"],
        }
    return out


def build_table(categories_subset, dict_metrics, part_num, n_parts):
    lines = []
    lines.append(r"\begin{table}[htb]")
    lines.append(r"\centering")
    lines.append(r"\scriptsize")
    n_groups = len(MODES) + 1  # + SOCCAT
    lines.append(r"\begin{tabular}{l" + " ccc" * n_groups + "}")
    lines.append(r"\toprule")

    header1 = [""]
    for mode in MODES:
        alpha0 = dict_metrics["alpha0_values"][mode]
        header1.append(rf"\multicolumn{{3}}{{c}}{{$\alpha_0$ = {mode} ({alpha0:,})}}")
    header1.append(r"\multicolumn{3}{c}{SOCCAT}")
    lines.append(" & ".join(header1) + r" \\")

    cmidrules = " ".join(
        rf"\cmidrule(lr){{{2 + 3*i}-{4 + 3*i}}}" for i in range(n_groups))
    lines.append(cmidrules)

    header2 = ["Specific category"] + ["P", "R", "F1"] * n_groups
    lines.append(" & ".join(header2) + r" \\")
    lines.append(r"\midrule")

    for cat in categories_subset:
        soccat = load_soccat_metrics(cat["name"])
        for label in cat["labels"]:
            row = [wrap_label(label)]
            for mode in MODES:
                m = dict_metrics[mode][(cat["name"], label)]
                row += [fmt(m["precision"]), fmt(m["recall"]), fmt(m["f1"])]
            s = soccat.get(label, {})
            row += [fmt(s.get("precision")), fmt(s.get("recall")), fmt(s.get("f1"))]
            lines.append(" & ".join(row) + r" \\")
        lines.append(r"\addlinespace")

    if lines[-1] == r"\addlinespace":
        lines.pop()

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(rf"\caption{{Comparison with dictionary-based classifiers, for different "
                 rf"values of $\alpha_0$, part {part_num}/{n_parts}.}}")
    lines.append(rf"\label{{tab:dictionary_alpha0_sweep_part{part_num}}}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def main():
    with open(CATEGORIES_FILE, encoding="utf-8") as f:
        categories = json.load(f)

    dict_metrics = {"alpha0_values": {}}
    for mode in MODES:
        df = pd.read_csv(OUTPUT_DIR / mode / "per_specific_category_metrics.csv")
        dict_metrics[mode] = {(r["category"], r["label"]): r for _, r in df.iterrows()}
        with open(ROOT / "dictionaries" / mode / "alpha0.json", encoding="utf-8") as f:
            dict_metrics["alpha0_values"][mode] = round(json.load(f)["alpha0"])

    # Split after 4 broad categories -- closest to an even row split (33 vs 24)
    # at a category boundary, out of 57 labels across 8 broad categories.
    split_idx = 4
    parts = [categories[:split_idx], categories[split_idx:]]

    for i, subset in enumerate(parts, start=1):
        table_tex = build_table(subset, dict_metrics, i, len(parts))
        out_path = OUTPUT_DIR / f"alpha0_sweep_table_part{i}.tex"
        out_path.write_text(table_tex, encoding="utf-8")
        print(f"Done. LaTeX table written to: {out_path}")

    print("Requires \\usepackage{booktabs} (and standard LaTeX only otherwise -- "
          "\\shortstack needs no extra package).")


if __name__ == "__main__":
    main()
