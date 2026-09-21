"""
step_2_best_fold_table.py

Description:
Produces one LaTeX table with the final, citable numbers for every category
of every broad class, evaluated on its "best fold" -- the same fold selected
by step_2_boxplot.py (highest n_pos_entail-weighted macro F1 across the
broad class's categories), i.e. the fold whose trained model is the one made
publicly available.

Columns: N (n_pos_entail), Bin. Precision/Recall/F1, Macro F1.
Each broad class gets a bold summary row (weighted average across its
categories, same weighting as the boxplot's broad-class aggregation) followed
by its categories sorted by decreasing macro F1.

ROC-AUC/PR-AUC are intentionally not in this table: they're not standard in
this literature and need the interpretation caveats discussed with the
reviewer response, not the main paper. See step_2_roc_pr_curves.py for those
-- kept as a separate, reviewer-response-only script.

Outputs:
- step_2/tables/best_fold_metrics.tex

Usage (from this directory):
python step_2_best_fold_table.py
"""
import glob
import os
import re

import numpy as np
import pandas as pd

STEP_2_DATA_DIR = "../data/model_performance/step_2"
LEGACY_MODEL_PERFORMANCE_DIR = f"{STEP_2_DATA_DIR}/model_performance"
OUTPUT_DIR = "step_2/tables"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "best_fold_metrics.tex")

BEST_FOLD_METRIC = "f1_macro"

FOLDER_NAME_MAP = {
    "age_family": "age_family_status",
    "identity": "identities_minority_majority_status",
    "labor_market_w_entrepreneurs": "labor_market_position",
    "profession": "profession",
    "real_estate": "real_estate_ownership",
    "social_deviance": "social_deviance",
    "social_roles": "social_roles_behavior",
    "socio_economic": "socio_economic_position",
}
BROAD_CLASS_LIST = list(FOLDER_NAME_MAP.keys())

RENAME_DICT = {
    "entrepreneurs in [specific] sector": "entrepreneurs in specific sector",
    "lgbtqqia+": "LGBTQIA+",
    "people with an immigration background, including immigrants": "people with immigration background",
    "offenders, criminals, prisoners and/or accused people": "offenders, criminals, prisoners, accused people",
    "terrorists, rebels, revolutionaries and/or movements of armed resistance": "terrorists, revolutionaries, rebels, armed resistance",
    "socio_economic": "Socio-economic position",
    "labor_market_w_entrepreneurs": "Labor market position",
    "age_family": "Age and family status",
    "identity": "Identities and minority/majority status",
    "profession": "Profession",
    "social_roles": "Social roles and behavior",
    "social_deviance": "Social deviance",
    "real_estate": "Real estate ownership",
}

METRIC_COLS = ["precision_binary", "recall_binary", "f1_binary", "f1_macro"]


def load_broad_class_df(broad_class):
    """Same logic as step_2_boxplot.py's loader (duplicated to keep this script standalone)."""
    folder = FOLDER_NAME_MAP.get(broad_class, broad_class)
    new_dir = f"{STEP_2_DATA_DIR}/{folder}"
    fold_files = sorted(
        glob.glob(f"{new_dir}/fold_*_per_label.csv"),
        key=lambda f: int(re.search(r"fold_(\d+)_per_label", f).group(1))
    )
    if fold_files:
        dfs = []
        for f in fold_files:
            fold_num = int(re.search(r"fold_(\d+)_per_label", f).group(1))
            d = pd.read_csv(f)
            d["fold"] = fold_num
            dfs.append(d)
        return pd.concat(dfs, ignore_index=True)

    legacy_file = f"{LEGACY_MODEL_PERFORMANCE_DIR}/{broad_class}_per_fold.csv"
    assert os.path.exists(legacy_file), f"No data found for broad class '{broad_class}'"
    return pd.read_csv(legacy_file)


def best_fold_for(df):
    """Fold with the highest n_pos_entail-weighted BEST_FOLD_METRIC across categories."""
    fold_scores = df.groupby("fold").apply(
        lambda g: np.average(g[BEST_FOLD_METRIC], weights=g["n_pos_entail"]),
        include_groups=False,
    )
    return int(fold_scores.idxmax())


def latex_escape(s):
    return str(s).replace("_", r"\_").replace("&", r"\&").replace("%", r"\%")


def fmt(x):
    return f"{x:.2f}" if pd.notna(x) else "--"


def broad_class_summary_row(df_best_fold):
    """n_pos_entail-weighted average across categories, same weighting as compute_broad_class_boxplot."""
    w = df_best_fold["n_pos_entail"]
    row = {m: np.average(df_best_fold[m], weights=w) for m in METRIC_COLS}
    row["n_pos_entail"] = int(w.sum())
    return row


# ============================================================
# MAIN
# ============================================================
os.makedirs(OUTPUT_DIR, exist_ok=True)

lines = []
lines.append(r"\begin{table}[t]")
lines.append(r"\centering")
lines.append(r"\small")
lines.append(r"\begin{tabular}{l r r r r r}")
lines.append(r"\toprule")
lines.append(r"Category & N & Bin. Prec. & Bin. Rec. & Bin. F1 & Macro F1 \\")
lines.append(r"\midrule")

for bc in BROAD_CLASS_LIST:
    df = load_broad_class_df(bc)
    fold = best_fold_for(df)
    df_best = df[df["fold"] == fold].copy()
    df_best["hypothesis_label"] = df_best["hypothesis_label"].replace(RENAME_DICT)

    bc_title = RENAME_DICT.get(bc, bc)
    summary = broad_class_summary_row(df_best)

    lines.append(r"\addlinespace")
    lines.append(
        rf"\textbf{{{latex_escape(bc_title)}}} (fold {fold}) & "
        rf"\textbf{{{summary['n_pos_entail']}}} & "
        rf"\textbf{{{fmt(summary['precision_binary'])}}} & "
        rf"\textbf{{{fmt(summary['recall_binary'])}}} & "
        rf"\textbf{{{fmt(summary['f1_binary'])}}} & "
        rf"\textbf{{{fmt(summary['f1_macro'])}}} \\"
    )

    df_best = df_best.sort_values("f1_macro", ascending=False)
    for _, row in df_best.iterrows():
        lines.append(
            rf"\quad {latex_escape(row['hypothesis_label'])} & "
            rf"{int(row['n_pos_entail'])} & "
            rf"{fmt(row['precision_binary'])} & "
            rf"{fmt(row['recall_binary'])} & "
            rf"{fmt(row['f1_binary'])} & "
            rf"{fmt(row['f1_macro'])} \\"
        )

lines.append(r"\bottomrule")
lines.append(r"\end{tabular}")
lines.append(
    r"\caption{Performance per category, evaluated on each broad class's best fold "
    r"(highest $n_\text{pos}$-weighted macro F1 across its categories) -- the fold "
    r"whose model is publicly released. Bold rows are the $n_\text{pos}$-weighted "
    r"average across categories within a broad class.}"
)
lines.append(r"\label{tab:step2_best_fold_metrics}")
lines.append(r"\end{table}")

tex = "\n".join(lines)
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    f.write(tex)

print(tex)
print(f"\nSaved file as: {OUTPUT_FILE}")
