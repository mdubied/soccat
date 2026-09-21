"""
compare_old_new_step2.py

Throwaway script (delete when done) comparing step 2 performance between:
- OLD model: data/model_performance/step_2/model_performance/<name>_per_fold.csv
- NEW model: data/model_performance/step_2/<folder>/fold_*_per_label.csv

For each broad class, both sources are aggregated the same way (weighted
average of f1_binary across labels, weighted by n_pos_entail, per fold),
then summarized as mean / median / best-fold F1 across the 5 folds.

Usage (from this directory):
python compare_old_new_step2.py
"""
import glob
import re

import numpy as np
import pandas as pd

STEP_2_DATA_DIR = "../data/model_performance/step_2"
LEGACY_DIR = f"{STEP_2_DATA_DIR}/model_performance"

# old_name (legacy "<old_name>_per_fold.csv") -> new_folder (under STEP_2_DATA_DIR)
BROAD_CLASS_MAP = {
    "age_family": "age_family_status",
    "identity": "identities_minority_majority_status",
    "labor_market_w_entrepreneurs": "labor_market_position",
    "profession": "profession",
    "real_estate": "real_estate_ownership",
    "social_deviance": "social_deviance",
    "social_roles": "social_roles_behavior",
    "socio_economic": "socio_economic_position",
}


def per_fold_weighted_f1(df):
    """Weighted (by n_pos_entail) average f1_binary per fold, across labels."""
    return df.groupby("fold").apply(
        lambda g: np.average(g["f1_binary"], weights=g["n_pos_entail"]),
        include_groups=False,
    )


def load_new(folder):
    fold_files = sorted(
        glob.glob(f"{STEP_2_DATA_DIR}/{folder}/fold_*_per_label.csv"),
        key=lambda f: int(re.search(r"fold_(\d+)_per_label", f).group(1)),
    )
    dfs = []
    for f in fold_files:
        fold_num = int(re.search(r"fold_(\d+)_per_label", f).group(1))
        d = pd.read_csv(f)
        d["fold"] = fold_num
        dfs.append(d)
    return pd.concat(dfs, ignore_index=True)


def load_old(old_name):
    return pd.read_csv(f"{LEGACY_DIR}/{old_name}_per_fold.csv")


def summarize(fold_f1):
    return {
        "mean": fold_f1.mean(),
        "median": fold_f1.median(),
        "best_fold": fold_f1.max(),
    }


rows = []
for old_name, new_folder in BROAD_CLASS_MAP.items():
    old_fold_f1 = per_fold_weighted_f1(load_old(old_name))
    new_fold_f1 = per_fold_weighted_f1(load_new(new_folder))

    old_stats = summarize(old_fold_f1)
    new_stats = summarize(new_fold_f1)

    rows.append({
        "broad_class": old_name,
        "old_mean": old_stats["mean"],
        "new_mean": new_stats["mean"],
        "delta_mean": new_stats["mean"] - old_stats["mean"],
        "old_median": old_stats["median"],
        "new_median": new_stats["median"],
        "delta_median": new_stats["median"] - old_stats["median"],
        "old_best_fold": old_stats["best_fold"],
        "new_best_fold": new_stats["best_fold"],
        "delta_best_fold": new_stats["best_fold"] - old_stats["best_fold"],
    })

result = pd.DataFrame(rows).sort_values("delta_mean", ascending=False)

pd.set_option("display.width", 160)
pd.set_option("display.float_format", lambda x: f"{x:.4f}")
print("\nWeighted F1 (per broad class, old vs. new model)\n")
print(result.to_string(index=False))
