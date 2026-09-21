"""
step_2_best_fold_table.py

Description:
Produces one LaTeX table with the final, citable numbers for every category
of every broad class, evaluated on its "best fold" -- the same fold selected
by figures/step_2_boxplot.py (highest n_pos_entail-weighted macro F1 across
the broad class's categories), i.e. the fold whose trained model is the one
made publicly available.

Columns: N (n_pos_entail), Bin. Precision/Recall/F1, Macro F1.
Each broad class gets a bold summary row (weighted average across its
categories, same weighting as the boxplot's broad-class aggregation) followed
by its categories sorted by decreasing macro F1.

ROC-AUC/PR-AUC are intentionally not in this table: they're not standard in
this literature and need the interpretation caveats discussed with the
reviewer response, not the main paper. See figures/step_2_roc_pr_curves.py
for those -- kept as a separate, reviewer-response-only script.

Outputs:
- step_2_best_fold_metrics_part1.tex, step_2_best_fold_metrics_part2.tex
  (split in two, row-count-balanced, so each fits on one page)
- step_2_best_fold_used.txt (which fold was selected per broad class, for reference)

Usage (from this directory):
python step_2_best_fold_table.py
"""
import glob
import re

import numpy as np
import pandas as pd

STEP_2_DATA_DIR = "../data/model_performance/step_2"
OUTPUT_FILE = "step_2_best_fold_metrics.tex"
FOLD_LOG_FILE = "step_2_best_fold_used.txt"

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
    """Same logic as figures/step_2_boxplot.py's loader (duplicated to keep this script standalone)."""
    folder = FOLDER_NAME_MAP.get(broad_class, broad_class)
    new_dir = f"{STEP_2_DATA_DIR}/{folder}"
    fold_files = sorted(
        glob.glob(f"{new_dir}/fold_*_per_label.csv"),
        key=lambda f: int(re.search(r"fold_(\d+)_per_label", f).group(1))
    )
    assert fold_files, f"No fold_*_per_label.csv found for broad class '{broad_class}' in {new_dir}"

    dfs = []
    for f in fold_files:
        fold_num = int(re.search(r"fold_(\d+)_per_label", f).group(1))
        d = pd.read_csv(f)
        d["fold"] = fold_num
        dfs.append(d)
    return pd.concat(dfs, ignore_index=True)


def best_fold_for(df):
    """Fold with the highest n_pos_entail-weighted BEST_FOLD_METRIC across categories."""
    fold_scores = df.groupby("fold").apply(
        lambda g: np.average(g[BEST_FOLD_METRIC], weights=g["n_pos_entail"]),
        include_groups=False,
    )
    return int(fold_scores.idxmax())


def latex_escape(s):
    return str(s).replace("_", r"\_").replace("&", r"\&").replace("%", r"\%")


def capitalize_first(s):
    s = str(s)
    return s[0].upper() + s[1:] if s else s


def fmt(x):
    return f"{x:.2f}" if pd.notna(x) else "--"


def broad_class_summary_row(df_best_fold):
    """n_pos_entail-weighted average across categories, same weighting as compute_broad_class_boxplot."""
    w = df_best_fold["n_pos_entail"]
    row = {m: np.average(df_best_fold[m], weights=w) for m in METRIC_COLS}
    row["n_pos_entail"] = int(w.sum())
    return row


def broad_class_rows(bc, fold_used):
    """Returns the list of table lines (summary row + category rows) for one broad class."""
    df = load_broad_class_df(bc)
    fold = best_fold_for(df)
    fold_used[bc] = fold
    df_best = df[df["fold"] == fold].copy()
    df_best["hypothesis_label"] = df_best["hypothesis_label"].replace(RENAME_DICT)

    bc_title = RENAME_DICT.get(bc, bc)
    summary = broad_class_summary_row(df_best)

    rows = [r"\addlinespace"]
    rows.append(
        rf"\textbf{{{latex_escape(bc_title)}}} & "
        rf"\textbf{{{summary['n_pos_entail']}}} & "
        rf"\textbf{{{fmt(summary['precision_binary'])}}} & "
        rf"\textbf{{{fmt(summary['recall_binary'])}}} & "
        rf"\textbf{{{fmt(summary['f1_binary'])}}} & "
        rf"\textbf{{{fmt(summary['f1_macro'])}}} \\"
    )

    df_best = df_best.sort_values("f1_macro", ascending=False)
    for _, row in df_best.iterrows():
        rows.append(
            rf"\quad {latex_escape(capitalize_first(row['hypothesis_label']))} & "
            rf"{int(row['n_pos_entail'])} & "
            rf"{fmt(row['precision_binary'])} & "
            rf"{fmt(row['recall_binary'])} & "
            rf"{fmt(row['f1_binary'])} & "
            rf"{fmt(row['f1_macro'])} \\"
        )
    return rows


def build_table(rows, part_num):
    lines = [r"\begin{table}[htb]", r"\centering", r"\scriptsize"]
    lines.append(r"\begin{tabular}{l r r r r r}")
    lines.append(r"\toprule")
    lines.append(r"Category & N & Bin. Prec. & Bin. Rec. & Bin. F1 & Macro F1 \\")
    lines.append(r"\midrule")
    lines.extend(rows)
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(
        r"\caption{Performance per category for each broad class' best fold. "
        r"N indicates the number of positive cases for the category in the test set "
        rf"(Part {part_num}/2).}}"
    )
    lines.append(rf"\label{{tab:step2_best_fold_metrics_{part_num}}}")
    lines.append(r"\end{table}")
    return lines


# ============================================================
# MAIN
# ============================================================
fold_used = {}
rows_by_bc = {bc: broad_class_rows(bc, fold_used) for bc in BROAD_CLASS_LIST}

# split broad classes across the two tables balancing by row count, not by
# number of broad classes, since categories per broad class vary a lot
# (e.g. profession has 14 categories, real_estate has 3). Pick the prefix
# split point that minimizes the row-count imbalance between the two halves.
row_counts = [len(rows_by_bc[bc]) for bc in BROAD_CLASS_LIST]
total_rows = sum(row_counts)
prefix_sums = np.cumsum(row_counts)
split_idx = int(np.argmin(np.abs(2 * prefix_sums - total_rows))) + 1

part1_bcs = BROAD_CLASS_LIST[:split_idx]
part2_bcs = BROAD_CLASS_LIST[split_idx:]

part1_rows = [line for bc in part1_bcs for line in rows_by_bc[bc]]
part2_rows = [line for bc in part2_bcs for line in rows_by_bc[bc]]

tex1 = "\n".join(build_table(part1_rows, 1))
tex2 = "\n".join(build_table(part2_rows, 2))

output_file_1 = OUTPUT_FILE.replace(".tex", "_part1.tex")
output_file_2 = OUTPUT_FILE.replace(".tex", "_part2.tex")
with open(output_file_1, "w", encoding="utf-8") as f:
    f.write(tex1)
with open(output_file_2, "w", encoding="utf-8") as f:
    f.write(tex2)

fold_log = "\n".join(
    f"{RENAME_DICT.get(bc, bc)} ({bc}): fold {fold}"
    for bc, fold in fold_used.items()
)
with open(FOLD_LOG_FILE, "w", encoding="utf-8") as f:
    f.write(fold_log + "\n")

print(tex1)
print()
print(tex2)
print(f"\nSaved file as: {output_file_1}")
print(f"Saved file as: {output_file_2}")
print(f"Saved file as: {FOLD_LOG_FILE}")
