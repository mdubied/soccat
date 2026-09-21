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

SOCCAT columns: read directly from the already-computed per-fold metrics
in data/model_performance/step_2/model_performance/<x>_per_fold.csv
(columns include hypothesis_label, fold, precision_binary, recall_binary,
f1_binary) -- NOT recomputed here. Two things vary per broad category and
are kept as explicit, separately-editable lookups below:
  SOCCAT_PER_FOLD_FILE  -- which *_per_fold.csv file
  SOCCAT_BEST_FOLD      -- which fold number counts as "best" for that
                           category's published model (this will change as
                           new folds get selected; update only this dict)
  SOCCAT_LABEL_REMAP    -- these files use the pre-LABEL_MAP legacy label
                           strings (e.g. "minors", "entrepreneur",
                           "prostitutes") rather than the taxonomy's
                           canonical names; mapped here to match the
                           dictionary-baseline label names 1:1.
If a broad category has no best-fold choice yet, its SOCCAT columns are
left blank ("--") rather than guessed.

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
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "output"
CATEGORIES_FILE = ROOT.parent / "src" / "step_2" / "categories.json"
PER_FOLD_DIR = ROOT.parent / "data" / "model_performance" / "step_2" / "model_performance"

MODES = ["low", "medium", "high"]
LABEL_LINE_WRAP_THRESHOLD = 24  # chars; labels longer than this get split onto 2 lines
DECIMALS = 2

SOCCAT_PER_FOLD_FILE = {
    "socio_economic_position":   "socio_economic_per_fold.csv",
    "labor_market_position":     "labor_market_w_entrepreneurs_per_fold.csv",
    "age_and_family_status":     "age_family_per_fold.csv",
    "identities":                "identity_per_fold.csv",
    "profession":                "profession_per_fold.csv",
    "social_roles_and_behavior": "social_roles_per_fold.csv",
    "social_deviance":           "social_deviance_per_fold.csv",
    "real_estate_ownership":     "real_estate_per_fold.csv",
}

# EDIT HERE when the "best" fold selection changes -- nothing else needs to.
# Fold numbers read off the filenames in
# data/annotated_validation_corpus/step_2/best_fold/*_best_foldN_human_vs_model.csv.
# None = no best-fold choice established yet for that broad category.
SOCCAT_BEST_FOLD = {
    "socio_economic_position":   1,
    "labor_market_position":     4,
    "age_and_family_status":     0,
    "identities":                4,
    "profession":                2,
    "social_roles_and_behavior": 1,
    "social_deviance":           3,
    "real_estate_ownership":     3,
}

# legacy hypothesis_label (as used in the *_per_fold.csv files) -> canonical
# taxonomy label. Only entries that actually differ need listing; anything
# else is assumed to already match.
SOCCAT_LABEL_REMAP = {
    "age_and_family_status": {
        "minors": "minors, including children and pupils",
        "youth": "youth, including students and apprentices",
    },
    "identities": {
        "christians": "Christians",
        "jews": "Jews",
        "muslims": "Muslims",
        "lgbtqqia+": "LGBTQIA+",
        "multiple (or other specific) religious or minority groups":
            "multiple (or other) religious or minority groups",
    },
    "labor_market_position": {
        "entrepreneur": "entrepreneurs",
        "housewife and househusband": "housewives and househusbands",
    },
    "profession": {
        "other profession": "other professions",
        "prostitutes": "sex workers",
    },
    "real_estate_ownership": {
        "real-estate owner": "real-estate owners",
    },
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


def load_soccat_metrics(broad_category: str) -> dict:
    """Returns {canonical_label: {precision, recall, f1}}, or {} if no best fold chosen."""
    fold = SOCCAT_BEST_FOLD[broad_category]
    if fold is None:
        return {}
    path = PER_FOLD_DIR / SOCCAT_PER_FOLD_FILE[broad_category]
    df = pd.read_csv(path)
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

    missing_soccat = [cat["name"] for cat in categories if SOCCAT_BEST_FOLD[cat["name"]] is None]
    if missing_soccat:
        print(f"Note: no best-fold choice for {missing_soccat} -- SOCCAT columns left blank ('--')")

    for i, subset in enumerate(parts, start=1):
        table_tex = build_table(subset, dict_metrics, i, len(parts))
        out_path = OUTPUT_DIR / f"alpha0_sweep_table_part{i}.tex"
        out_path.write_text(table_tex, encoding="utf-8")
        print(f"Done. LaTeX table written to: {out_path}")

    print("Requires \\usepackage{booktabs} (and standard LaTeX only otherwise -- "
          "\\shortstack needs no extra package).")


if __name__ == "__main__":
    main()
