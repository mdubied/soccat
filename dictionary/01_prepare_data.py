#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prepare train/test data for the dictionary-based baseline
===========================================================
Loads the same ground-truth annotations used by the NLI pipeline
(src/step_2/convert_annotations.py), keeps the sentences with a non-empty
ground_truth (target_abroad = 1 sentences are already removed from that file),
applies the same label mapping, and splits sentences 80/20 (random, seed=42)
into train/test. Sentences of step 1's test set are then removed from train
(not from test), since the dictionaries are also evaluated on that set; the
final split is written to data/split_info.json.

train.csv is used by 02_build_dictionary.py to mine word lists per label.
test.csv  is used by 03_evaluate_dictionary.py to evaluate them.

NOTE: TAXONOMY / LABEL_MAP / parse_labels below are copied from
src/step_2/convert_annotations.py (that script is not import-safe — it
executes its conversion at module level). If the taxonomy or label
mapping changes there, mirror the change here.
"""

import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
INPUT_FILE = ROOT / "data" / "manual_annotations" / "step_2" / "annotations_ground_truth.csv"
STEP1_TEST_FILE = ROOT / "data" / "model_performance" / "step_1" / "test_with_all_outlets.json"
OUTPUT_DIR = Path(__file__).resolve().parent / "data"
SEED = 42
TEST_SIZE = 0.2

# ── Taxonomy (copied from src/step_2/convert_annotations.py) ──────────────────
TAXONOMY = {
    "socio_economic_position": [
        "lower class", "middle class", "upper class",
        "capital owners, investors and shareholders",
        "unskilled or unqualified", "skilled or qualified",
    ],
    "labor_market_position": [
        "wage and salary earners", "civil servants", "CEOs and corporate leaders",
        "employers", "entrepreneurs", "self-employed and freelancers",
        "unemployed", "retirees", "housewives and househusbands",
    ],
    "age_and_family_status": [
        "parents and families", "minors, including children and pupils",
        "youth, including students and apprentices",
        "middle-aged and pre-retirement age groups",
        "elderly", "couples", "singles",
    ],
    "identities": [
        "men", "women", "cisgender and heterosexuals", "LGBTQIA+",
        "disabled people",
        "people with an immigration background, including immigrants",
        "Ethnic and racial minorities", "Christians", "Jews", "Muslims",
        "multiple (or other) religious or minority groups",
    ],
    "profession": [
        "athletes", "authors and artists", "doctors", "farmers and fishermen",
        "health and care professionals", "journalists", "legal professionals",
        "politicians and high-ranking officials", "sex workers",
        "scientists and professors", "security forces", "soldiers",
        "teachers and educators", "other professions",
    ],
    "social_roles_and_behavior": [
        "consumers and clients", "car drivers", "patients",
    ],
    "social_deviance": [
        "extremists",
        "terrorists, rebels, revolutionaries and/or movements of armed resistance",
        "offenders, criminals, prisoners and/or accused people", "drug addicts",
    ],
    "real_estate_ownership": [
        "real-estate owners", "tenants", "homeless",
    ],
}

LABEL_MAP = {
    "lower class": "lower class", "middle class": "middle class", "upper class": "upper class",
    "capital owners, investors and shareholders": "capital owners, investors and shareholders",
    "unskilled or unqualified": "unskilled or unqualified",
    "skilled or qualified": "skilled or qualified",

    "wage and salary earners": "wage and salary earners",
    "civil servants": "civil servants",
    "ceos and corporate leaders": "CEOs and corporate leaders",
    "employers": "employers",
    "entrepreneurs": "entrepreneurs",
    "entrepreneurs (smes)": "entrepreneurs",
    "entrepreneurs in [specific] sector": "entrepreneurs",
    "entrepreneurs (large enterprises)": "CEOs and corporate leaders",
    "self-employed and freelancers": "self-employed and freelancers",
    "unemployed": "unemployed",
    "retirees": "retirees",
    "housewife and househusband": "housewives and househusbands",
    "housewives and househusbands": "housewives and househusbands",

    "parents and families": "parents and families",
    "minors": "minors, including children and pupils",
    "minors, including children and pupils": "minors, including children and pupils",
    "youth": "youth, including students and apprentices",
    "youth, including students and apprentices": "youth, including students and apprentices",
    "middle-aged and pre-retirement age groups": "middle-aged and pre-retirement age groups",
    "elderly": "elderly", "couples": "couples", "singles": "singles",

    "men": "men", "women": "women",
    "cisgender and heterosexuals": "cisgender and heterosexuals",
    "lgbtqia+": "LGBTQIA+", "lgbtqqia+": "LGBTQIA+",
    "disabled people": "disabled people",
    "people with an immigration background, including immigrants":
        "people with an immigration background, including immigrants",
    "ethnic and racial minorities": "Ethnic and racial minorities",
    "christians": "Christians", "jews": "Jews", "muslims": "Muslims",
    "multiple (or other specific) religious or minority groups":
        "multiple (or other) religious or minority groups",
    "multiple (or other) religious or minority groups":
        "multiple (or other) religious or minority groups",

    "athletes": "athletes", "authors and artists": "authors and artists",
    "doctors": "doctors", "farmers and fishermen": "farmers and fishermen",
    "health and care professionals": "health and care professionals",
    "journalists": "journalists", "legal professionals": "legal professionals",
    "politicians and high-ranking officials": "politicians and high-ranking officials",
    "sex workers": "sex workers", "prostitutes": "sex workers",
    "scientists and professors": "scientists and professors",
    "security forces": "security forces", "soldiers": "soldiers",
    "teachers and educators": "teachers and educators",
    "other profession": "other professions", "other professions": "other professions",

    "consumers and clients": "consumers and clients",
    "car drivers": "car drivers", "patients": "patients",

    "extremists": "extremists",
    "terrorists": "terrorists, rebels, revolutionaries and/or movements of armed resistance",
    "terrorists, rebels, revolutionaries and/or movements of armed resistance":
        "terrorists, rebels, revolutionaries and/or movements of armed resistance",
    "offenders, criminals, prisoners and/or accused people":
        "offenders, criminals, prisoners and/or accused people",
    "drug addicts": "drug addicts",

    "real-estate owner": "real-estate owners", "real-estate owners": "real-estate owners",
    "real estate owners": "real-estate owners",
    "tenants": "tenants", "homeless": "homeless",
}

LABEL_TO_CATEGORY = {lbl: cat for cat, labels in TAXONOMY.items() for lbl in labels}


def parse_labels(raw) -> set:
    """Parse semicolon-separated specific_group_new -> set of taxonomy labels."""
    if pd.isna(raw):
        return set()
    mapped = set()
    for part in str(raw).split(";"):
        new_lbl = LABEL_MAP.get(part.strip().lower())
        if new_lbl:
            mapped.add(new_lbl)
    return mapped


def normalise_text(text) -> str:
    """Collapse whitespace (line endings differ between the step 1 and step 2 files)."""
    return " ".join(str(text).split())


def load_step1_test_texts() -> set:
    with STEP1_TEST_FILE.open(encoding="utf-8") as f:
        return {normalise_text(json.loads(line)["text"]) for line in f if line.strip()}


def main():
    print("Loading data...")
    df = pd.read_csv(INPUT_FILE)
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    print(f"  {len(df):,} sentences")

    # Keep sentences mentioning at least one social group: the same 7,234 sentences
    # the NLI models are evaluated on (data/model_performance/step_2/*/all_folds_human_vs_model.csv)
    has_group = ~df["ground_truth"].fillna("").str.strip().isin(["", "[]"])
    df = df[has_group].copy()
    print(f"  {len(df):,} with a non-empty ground_truth")

    df["labels"] = df["specific_group_new"].apply(parse_labels)
    df["categories"] = df["labels"].apply(lambda labs: {LABEL_TO_CATEGORY[l] for l in labs})

    n_no_label = (df["labels"].apply(len) == 0).sum()
    print(f"  {n_no_label:,} sentences map to no taxonomy label (e.g. 'others'-only)")

    train_df, test_df = train_test_split(df, test_size=TEST_SIZE, random_state=SEED)
    print(f"  train: {len(train_df):,}  test: {len(test_df):,}")

    # The dictionaries are also evaluated on step 1's test set (06_evaluate_step1.py),
    # so its sentences must not be mined for dictionary words: drop them from train
    step1_texts = load_step1_test_texts()
    in_step1 = train_df["text"].map(normalise_text).isin(step1_texts)
    n_split = len(train_df)
    train_df = train_df[~in_step1]
    print(f"  removed {int(in_step1.sum()):,} step 1 test sentences from train -> "
          f"train: {len(train_df):,}  test: {len(test_df):,}")

    OUTPUT_DIR.mkdir(exist_ok=True)
    split_info = {
        "n_sentences": len(df),
        "n_train_before_step1_filter": n_split,
        "n_removed_step1_test": int(in_step1.sum()),
        "n_train": len(train_df),
        "n_test": len(test_df),
        "step1_test_path": str(STEP1_TEST_FILE.relative_to(ROOT)),
    }
    (OUTPUT_DIR / "split_info.json").write_text(json.dumps(split_info, indent=2), encoding="utf-8")
    for name, split in [("train", train_df), ("test", test_df)]:
        out = split[["id", "text", "outlet", "country", "year"]].copy()
        out["labels"] = split["labels"].apply(lambda s: ";".join(sorted(s)))
        out["categories"] = split["categories"].apply(lambda s: ";".join(sorted(s)))
        out.to_csv(OUTPUT_DIR / f"{name}.csv", index=False)

    print("\nPer-label positive counts (train / test):")
    for cat, labels in TAXONOMY.items():
        for label in labels:
            n_train = train_df["labels"].apply(lambda s: label in s).sum()
            n_test = test_df["labels"].apply(lambda s: label in s).sum()
            print(f"  {cat:28s} {label:60s} train={n_train:4d}  test={n_test:4d}")

    print(f"\nDone. Files written to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
