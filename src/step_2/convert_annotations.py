#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert complete_merged_annotations → NLI pair CSVs per category
=================================================================
Reads the full 9,639-sentence annotation file. For each sentence × label
in each of the 8 taxonomy categories, writes one row:

    nli_label = 0  →  entailment   (sentence mentions this group)
    nli_label = 1  →  not_entailment  (sentence does not)

Sentences with multiple categories appear in every relevant category CSV.
One output CSV per category, named nli_dataset_{category}.csv.
"""

import os
from pathlib import Path

import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────────
INPUT_FILE = Path(__file__).resolve().parents[2] / "data" / "manual_annotations" / "step_2" / "annotations_ground_truth.csv"
OUTPUT_DIR = "nli_pairs_by_category"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Taxonomy ───────────────────────────────────────────────────────────────────
TAXONOMY = {
    "socio_economic_position": [
        "lower class",
        "middle class",
        "upper class",
        "capital owners, investors and shareholders",
        "unskilled or unqualified",
        "skilled or qualified",
    ],
    "labor_market_position": [
        "wage and salary earners",
        "civil servants",
        "CEOs and corporate leaders",
        "employers",
        "entrepreneurs",
        "self-employed and freelancers",
        "unemployed",
        "retirees",
        "housewives and househusbands",
    ],
    "age_and_family_status": [
        "parents and families",
        "minors, including children and pupils",
        "youth, including students and apprentices",
        "middle-aged and pre-retirement age groups",
        "elderly",
        "couples",
        "singles",
    ],
    "identities": [
        "men",
        "women",
        "cisgender and heterosexuals",
        "LGBTQIA+",
        "disabled people",
        "people with an immigration background, including immigrants",
        "Ethnic and racial minorities",
        "Christians",
        "Jews",
        "Muslims",
        "multiple (or other) religious or minority groups",
    ],
    "profession": [
        "athletes",
        "authors and artists",
        "doctors",
        "farmers and fishermen",
        "health and care professionals",
        "journalists",
        "legal professionals",
        "politicians and high-ranking officials",
        "sex workers",
        "scientists and professors",
        "security forces",
        "soldiers",
        "teachers and educators",
        "other professions",
    ],
    "social_roles_and_behavior": [
        "consumers and clients",
        "car drivers",
        "patients",
    ],
    "social_deviance": [
        "extremists",
        "terrorists, rebels, revolutionaries and/or movements of armed resistance",
        "offenders, criminals, prisoners and/or accused people",
        "drug addicts",
    ],
    "real_estate_ownership": [
        "real-estate owners",
        "tenants",
        "homeless",
    ],
}

# All taxonomy labels as a flat set (lowercased for lookup)
ALL_TAXONOMY_LABELS = {lbl.lower(): lbl for cat_labels in TAXONOMY.values() for lbl in cat_labels}

# ── Label normalisation: specific_group_new → taxonomy label ──────────────────
# Maps raw values from specific_group_new to exact taxonomy labels.
# Keys are lowercase-stripped. Values not listed → not in taxonomy (skip).
LABEL_MAP = {
    # socio_economic_position
    "lower class":                          "lower class",
    "middle class":                         "middle class",
    "upper class":                          "upper class",
    "capital owners, investors and shareholders": "capital owners, investors and shareholders",
    "unskilled or unqualified":             "unskilled or unqualified",
    "skilled or qualified":                 "skilled or qualified",

    # labor_market_position
    "wage and salary earners":              "wage and salary earners",
    "civil servants":                       "civil servants",
    "ceos and corporate leaders":           "CEOs and corporate leaders",
    "employers":                            "employers",
    "entrepreneurs":                        "entrepreneurs",
    "entrepreneurs (smes)":                 "entrepreneurs",
    "entrepreneurs in [specific] sector":   "entrepreneurs",
    "entrepreneurs (large enterprises)":    "CEOs and corporate leaders",
    "self-employed and freelancers":        "self-employed and freelancers",
    "unemployed":                           "unemployed",
    "retirees":                             "retirees",
    "housewife and househusband":           "housewives and househusbands",
    "housewives and househusbands":         "housewives and househusbands",

    # age_and_family_status
    "parents and families":                 "parents and families",
    "minors":                               "minors, including children and pupils",
    "minors, including children and pupils": "minors, including children and pupils",
    "youth":                                "youth, including students and apprentices",
    "youth, including students and apprentices": "youth, including students and apprentices",
    "middle-aged and pre-retirement age groups": "middle-aged and pre-retirement age groups",
    "elderly":                              "elderly",
    "couples":                              "couples",
    "singles":                              "singles",

    # identities
    "men":                                  "men",
    "women":                                "women",
    "cisgender and heterosexuals":          "cisgender and heterosexuals",
    "lgbtqia+":                             "LGBTQIA+",
    "lgbtqqia+":                            "LGBTQIA+",
    "disabled people":                      "disabled people",
    "people with an immigration background, including immigrants":
        "people with an immigration background, including immigrants",
    "ethnic and racial minorities":         "Ethnic and racial minorities",
    "christians":                           "Christians",
    "jews":                                 "Jews",
    "muslims":                              "Muslims",
    "multiple (or other specific) religious or minority groups":
        "multiple (or other) religious or minority groups",
    "multiple (or other) religious or minority groups":
        "multiple (or other) religious or minority groups",

    # profession
    "athletes":                             "athletes",
    "authors and artists":                  "authors and artists",
    "doctors":                              "doctors",
    "farmers and fishermen":                "farmers and fishermen",
    "health and care professionals":        "health and care professionals",
    "journalists":                          "journalists",
    "legal professionals":                  "legal professionals",
    "politicians and high-ranking officials": "politicians and high-ranking officials",
    "sex workers":                          "sex workers",
    "prostitutes":                          "sex workers",
    "scientists and professors":            "scientists and professors",
    "security forces":                      "security forces",
    "soldiers":                             "soldiers",
    "teachers and educators":               "teachers and educators",
    "other profession":                     "other professions",
    "other professions":                    "other professions",

    # social_roles_and_behavior
    "consumers and clients":                "consumers and clients",
    "car drivers":                          "car drivers",
    "patients":                             "patients",

    # social_deviance
    "extremists":                           "extremists",
    "terrorists":                           "terrorists, rebels, revolutionaries and/or movements of armed resistance",
    "terrorists, rebels, revolutionaries and/or movements of armed resistance":
        "terrorists, rebels, revolutionaries and/or movements of armed resistance",
    "offenders, criminals, prisoners and/or accused people":
        "offenders, criminals, prisoners and/or accused people",
    "drug addicts":                         "drug addicts",

    # real_estate_ownership
    "real-estate owner":                    "real-estate owners",
    "real-estate owners":                   "real-estate owners",
    "real estate owners":                   "real-estate owners",
    "tenants":                              "tenants",
    "homeless":                             "homeless",
}

# Reverse: taxonomy label → category name
LABEL_TO_CATEGORY = {
    lbl: cat for cat, labels in TAXONOMY.items() for lbl in labels
}


def make_hypothesis(category: str, label: str) -> str:
    cat_display = {
        "socio_economic_position":   "socio-economic position",
        "labor_market_position":     "labor market position",
        "age_and_family_status":     "age and family status",
        "identities":                "identities and minority/majority status",
        "profession":                "profession",
        "social_roles_and_behavior": "social roles and behavior",
        "social_deviance":           "social deviance",
        "real_estate_ownership":     "real estate ownership",
    }[category]
    return (
        f'This sentence refers to {cat_display} as a social group, '
        f'specifically "{label}".'
    )


def parse_labels(raw) -> set:
    """Parse semicolon-separated specific_group_new → set of taxonomy labels."""
    if pd.isna(raw):
        return set()
    mapped = set()
    for part in str(raw).split(";"):
        key = part.strip().lower()
        new_lbl = LABEL_MAP.get(key)
        if new_lbl:
            mapped.add(new_lbl)
    return mapped


# ── Load ───────────────────────────────────────────────────────────────────────
print("Loading data...")
df = pd.read_csv(INPUT_FILE)
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
print(f"  {len(df):,} sentences | outlets: {sorted(df['outlet'].dropna().unique())}")
print(f"  Years: {int(df['year'].min())}–{int(df['year'].max())}")

# ── Parse all labels once ─────────────────────────────────────────────────────
print("Parsing labels...")
df["_labels"] = df["specific_group_new"].apply(parse_labels)

# Track unmapped for transparency
unmapped = {}
for raw in df["specific_group_new"].dropna():
    for part in str(raw).split(";"):
        key = part.strip().lower()
        if key and not LABEL_MAP.get(key):
            unmapped[key] = unmapped.get(key, 0) + 1

# ── Generate NLI pairs per category ───────────────────────────────────────────
print("\nGenerating NLI pairs...")

for cat, labels in TAXONOMY.items():
    rows = []
    for _, row in df.iterrows():
        pos_labels = row["_labels"]
        for label in labels:
            rows.append({
                "sentence_id":      int(row["id"]),
                "premise":          str(row["text"]).strip(),
                "hypothesis":       make_hypothesis(cat, label),
                "nli_label":        0 if label in pos_labels else 1,
                "hypothesis_label": label,
                "outlet":           row["outlet"] if pd.notna(row["outlet"]) else "Unknown",
                "country":          row["country"] if pd.notna(row["country"]) else "Unknown",
                "date":             row["date"].date().isoformat() if pd.notna(row["date"]) else None,
                "year":             int(row["year"]) if pd.notna(row["year"]) else None,
            })

    out = pd.DataFrame(rows)
    n_pos = (out["nli_label"] == 0).sum()
    n_neg = (out["nli_label"] == 1).sum()
    pct   = 100 * n_pos / len(out)
    path  = os.path.join(OUTPUT_DIR, f"nli_dataset_{cat}.csv")
    out.to_csv(path, index=False)
    print(f"  {cat:35s}  {len(out):7,} pairs  "
          f"pos={n_pos:5,} ({pct:.1f}%)  neg={n_neg:6,}")

# ── Unmapped label report ──────────────────────────────────────────────────────
print("\nLabels in specific_group_new not mapped to any taxonomy label:")
for lbl, cnt in sorted(unmapped.items(), key=lambda x: -x[1])[:30]:
    print(f"  {cnt:5d}×  {lbl}")

print("\nDone. Files written to:", OUTPUT_DIR)
