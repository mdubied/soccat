#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Per-label F1 table: zero-shot NLI, fine-tuned classifier and single NLI model vs SOCCAT
========================================================================================
Appendix table comparing SOCCAT (one NLI model per broad class) with three
classifiers built on the same base model, in the same layout as the LLM and dictionary
comparisons (dictionary/05_make_latex_table.py's build_table is reused, so the
rows, label wrapping, split into two parts and SOCCAT columns are identical):

- Zero-shot NLI (zero_shot_nli/): SOCCAT's base model without fine-tuning.
- Fine-tuned classifier (fine_tuned_classifier/): the same base model fine-tuned
  as a 57-label multi-label classifier.
- Single NLI model (single_nli/): one NLI model for all 57 labels instead of
  one per broad class, its hypotheses naming the specific label only.

All use the same 7,234 sentences, gold labels and 5-fold CV split
(see common.py). Like SOCCAT's "mean"/"med" columns, their scores are
the mean / median over the 5 folds of each label's positive-class F1
(data/fold_*_per_label.csv). They have no "best fold" column: no fold of theirs
is a released model. As in the LLM table, a value is set in italics when it
beats SOCCAT's mean F1 for that label (and the label is then in italics too).

Outputs (output/):
    baselines_table_part{1,2}.tex    F1 per label (LaTeX, booktabs + threeparttable)
    baselines_per_label_f1.csv       the same numbers, plus SOCCAT's

Usage: python compare_baselines.py   (run from anywhere)
"""

import json
import sys
from importlib import import_module
from pathlib import Path

import pandas as pd

from common import SINGLE_NLI_MODELS

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
OUTPUT_DIR = ROOT / "output"
N_FOLDS = 5

sys.path.insert(0, str(REPO / "dictionary"))
dt = import_module("05_make_latex_table")   # build_table, load_soccat_metrics, categories

MODELS = {   # key -> (folder with fold_*_per_label.csv, column title, name in printed output)
    "zero_shot": (ROOT / "zero_shot_nli" / "data", "Zero-shot NLI", "Zero-shot NLI"),
    "fine_tuned": (ROOT / "fine_tuned_classifier" / "data", "Fine-tuned classifier", "Fine-tuned classifier"),
    **{folder: (ROOT / folder / "data", m["header"], m["title"]) for folder, m in SINGLE_NLI_MODELS.items()},
}
# broad class as in the models' hypothesis_label ("Dimension | label") -> categories.json name
DIMENSION_TO_CATEGORY = {
    "Age and Family Status": "age_and_family_status",
    "Identities": "identities",
    "Labor Market Position": "labor_market_position",
    "Profession": "profession",
    "Real Estate Ownership": "real_estate_ownership",
    "Social Deviance": "social_deviance",
    "Social Roles and Behavior": "social_roles_and_behavior",
    "Socio-Economic Position": "socio_economic_position",
}


def fold_f1(folder: Path) -> pd.DataFrame:
    """Each label's F1 per fold -> mean and median over the folds, keyed by (category, label)."""
    d = pd.concat([pd.read_csv(folder / f"fold_{k}_per_label.csv").assign(fold=k) for k in range(N_FOLDS)])
    parts = d["hypothesis_label"].str.split(" | ", n=1, regex=False)
    d["category"], d["label"] = parts.str[0].map(DIMENSION_TO_CATEGORY), parts.str[1]
    assert d["category"].notna().all(), f"{folder}: unknown broad class"
    return d.groupby(["category", "label"])["f1_binary"].agg(mean="mean", median="median")


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    categories = json.loads(dt.CATEGORIES_FILE.read_text(encoding="utf-8"))

    lookup, rows = {}, []
    for key, (folder, _, _) in MODELS.items():
        stats = fold_f1(folder)
        expected = {(c["name"], l) for c in categories for l in c["labels"]}
        assert set(stats.index) == expected, f"{key}: labels differ from categories.json"
        for stat in ("mean", "median"):
            lookup[f"{key}_{stat}"] = {idx: {"f1": v} for idx, v in stats[stat].items()}
    for c in categories:
        soccat = dt.load_soccat_metrics(c["name"])
        for l in c["labels"]:
            rows.append({"broad_class": c["display_name"], "label": l,
                         **{k: v[(c["name"], l)]["f1"] for k, v in lookup.items()},
                         **{f"soccat_{s}": soccat[l][s]["f1"] for s in dt.SOCCAT_VARIANTS}})
    pd.DataFrame(rows).to_csv(OUTPUT_DIR / "baselines_per_label_f1.csv", index=False)

    super_groups = [(title, [(f"{key}_mean", "mean"), (f"{key}_median", "med")])
                    for key, (_, title, _) in MODELS.items()]
    super_groups.append(("SOCCAT", [(v, dt.SOCCAT_VARIANT_TITLES[v]) for v in dt.SOCCAT_VARIANTS]))
    highlight = [f"{key}_{stat}" for key in MODELS for stat in ("mean", "median")]
    note = (r"Reported scores are the binary, positive-class F1 score. Zero-shot NLI: SOCCAT's base "
            r"NLI model (mDeBERTa-v3-base-mnli-xnli) without fine-tuning; fine-tuned classifier: the "
            r"same model fine-tuned as a multi-label classifier (one output per group label); single "
            r"NLI model: one NLI model for all group labels instead of one per broad class, its "
            r"hypotheses naming the group label only. All "
            r"are evaluated by 5-fold cross-validation on the same sentences and annotations as "
            r"SOCCAT; mean / median (med) across the 5 folds. SOCCAT: best cross-validation fold "
            r"(best, released model), or mean / median across the 5 folds. Italics are used for a "
            r"label when a baseline beats SOCCAT's mean F1, and it shows the beating F1 values.")

    split_idx = 4   # as in the dictionary table: after the first 4 broad classes
    parts = [categories[:split_idx], categories[split_idx:]]
    for i, subset in enumerate(parts, start=1):
        tex = dt.build_table(subset, i, len(parts), metrics=["f1"], super_groups=super_groups,
                             dict_lookup=lookup, caption="Comparison with embedding-based classifiers.",
                             table_label="tab:embedding_baselines", note=note, highlight_keys=highlight)
        out = OUTPUT_DIR / f"baselines_table_part{i}.tex"
        out.write_text(tex, encoding="utf-8")
        print(f"Saved file as: {out}")

    df = pd.DataFrame(rows)
    n = len(df)
    print(f"\nLabels where a baseline beats SOCCAT's mean F1 (of {n}):")
    for key, (_, _, title) in MODELS.items():
        print(f"  {title:<46} mean {int((df[f'{key}_mean'] > df['soccat_mean']).sum()):>2}  "
              f"| average over labels of the fold-mean F1: {df[f'{key}_mean'].mean():.3f}")
    print(f"  {'SOCCAT':<46}         | average over labels of the fold-mean F1: {df['soccat_mean'].mean():.3f}")


if __name__ == "__main__":
    main()
