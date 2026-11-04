#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 2: initial specification vs current main specification (response to reviewers)
=====================================================================================
In the initial version, the training and test sets of a broad class held only the
sentences annotated with a label of that broad class; sentences mentioning groups of
other broad classes were left out. The current specification uses all 7,234
annotated sentences for every broad class, so each model also meets sentences about
other groups (hard negatives).

Three rows, positive-class scores pooled over all (sentence, label) pairs of the
five test folds and all broad classes:
  1. initial model, initial test sets      (data/model_performance/step_2/00_initial_performance/
                                            *_per_fold.csv; only per-label summaries survive, so
                                            TP/FP/FN are rebuilt from recall, accuracy and counts)
  2. current model, test sets restricted the initial way: per broad class, only the
     sentences with a gold label of that class (current fold_*_human_vs_model.csv)
  3. current model, full test sets          (same files, all sentences)

Rows 1 and 2 use the same kind of test set, so their similarity shows that the
models perform alike and that the drop in row 3 comes from the test set.

Outputs: tables/step_2_initial_vs_current.tex / .csv (per broad class and overall)
Usage:   python step_2_initial_vs_current.py   (run from anywhere)
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
STEP_2 = ROOT.parent / "data" / "model_performance" / "step_2"
INITIAL = STEP_2 / "00_initial_performance"
OUT_TEX, OUT_CSV = ROOT / "step_2_initial_vs_current.tex", ROOT / "step_2_initial_vs_current.csv"

# current broad-class folder -> prefix of the initial *_per_fold.csv
CLASSES = {
    "socio_economic_position": "socio_economic",
    "labor_market_position": "labor_market_w_entrepreneurs",
    "age_family_status": "age_family",
    "identities_minority_majority_status": "identity",
    "profession": "profession",
    "social_roles_behavior": "social_roles",
    "social_deviance": "social_deviance",
    "real_estate_ownership": "real_estate",
}
ROWS = {
    "initial": ("Initial", "Sentences of the broad class only"),
    "current_restricted": ("Current", "Sentences of the broad class only"),
    "current_full": ("Current", "All sentences"),
}


def initial_counts(prefix: str) -> dict:
    """TP/FP/FN and test sentences of one broad class, from the initial per-label summaries."""
    d = pd.read_csv(INITIAL / f"{prefix}_per_fold.csv")
    tp = (d["recall_binary"] * d["n_pos_entail"]).round()
    fn = d["n_pos_entail"] - tp
    fp = ((1 - d["accuracy"]) * d["n_pairs"] - fn).round()
    sentences = int(d.groupby("fold")["n_pairs"].first().sum())  # pairs per label = test sentences
    return {"tp": tp.sum(), "fp": fp.sum(), "fn": fn.sum(), "sentences": sentences}


def current_counts(folder: str) -> dict:
    """TP/FP/FN of one broad class on the full and on the restricted test sets."""
    d = pd.concat([pd.read_csv(f, usecols=["sentence_id", "nli_label", "pred_label"])
                   for f in sorted((STEP_2 / folder).glob("fold_*_human_vs_model.csv"))])
    y, p = d["nli_label"].eq(0), d["pred_label"].eq(0)   # 0 = entailment (label applies)
    in_class = d["sentence_id"].isin(set(d.loc[y, "sentence_id"]))
    out = {}
    for key, m in (("current_full", np.ones(len(d), bool)), ("current_restricted", in_class.to_numpy())):
        out[key] = {"tp": (y & p)[m].sum(), "fp": (~y & p)[m].sum(), "fn": (y & ~p)[m].sum(),
                    "sentences": d.loc[m, "sentence_id"].nunique()}
    return out


def scores(c: dict) -> dict:
    tp, fp, fn = c["tp"], c["fp"], c["fn"]
    return {"precision": tp / (tp + fp), "recall": tp / (tp + fn), "f1": 2 * tp / (2 * tp + fp + fn)}


def main():
    rows = []
    for folder, prefix in CLASSES.items():
        per = {"initial": initial_counts(prefix), **current_counts(folder)}
        for key, c in per.items():
            rows.append({"broad_class": folder, "version": key, **c})
    df = pd.DataFrame(rows)
    total = df.groupby("version")[["tp", "fp", "fn", "sentences"]].sum().reset_index().assign(broad_class="All")
    df = pd.concat([df, total], ignore_index=True)
    df = pd.concat([df, pd.DataFrame([scores(r) for _, r in df.iterrows()])], axis=1)
    df.to_csv(OUT_CSV, index=False)

    allrows = df[df["broad_class"] == "All"].set_index("version")
    lines = [
        r"\begin{table}[htb]",
        r"\centering",
        r"\scriptsize",
        r"\begin{threeparttable}",
        r"\caption{Step 2 performance of the initial and the current specification.}",
        r"\label{tab:step-2-initial-vs-current}",
        r"\begin{tabular}{l l r r r r}",
        r"\toprule",
        r"Model & Test set & Test sentences & Precision & Recall & F1 \\",
        r"\midrule",
    ]
    for key, (model, test) in ROWS.items():
        r = allrows.loc[key]
        lines.append(f"{model} & {test} & {int(r['sentences']):,} & {r['precision']:.2f} & "
                     f"{r['recall']:.2f} & {r['f1']:.2f} \\\\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\begin{tablenotes}[flushleft]",
        r"\footnotesize",
        r"\item \textit{Note:} Positive-class scores pooled over all (sentence, group label) pairs of the "
        r"five cross-validation test folds and all eight broad classes. Test sentences: summed over the "
        r"eight broad classes. In the initial specification, the training and test sets of a broad class "
        r"contained only the sentences annotated with a label of that class; in the current specification, "
        r"they contain all annotated sentences. The second row scores the current model on test sets "
        r"restricted in the initial way.",
        r"\end{tablenotes}",
        r"\end{threeparttable}",
        r"\end{table}",
        "",
    ]
    OUT_TEX.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved file as: {OUT_TEX}\nSaved file as: {OUT_CSV}\n")
    with pd.option_context("display.width", 200):
        print(df[["broad_class", "version", "sentences", "precision", "recall", "f1"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
