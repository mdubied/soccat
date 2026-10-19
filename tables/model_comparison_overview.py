#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Overview table: Step 2 performance of all compared methods
===========================================================
One row per method, all scored the same way, so the rows are comparable:

- gold labels: the 57 specific labels as used to train and evaluate SOCCAT
  (nli_label of data/model_performance/step_2/<class>/fold_*_human_vs_model.csv);
  the residual "others" category, which only the LLM predicts, is left out;
- precision, recall and F1: pooled over all (sentence, label) pairs of the
  method's test sentences (one prediction per pair, P > 0.5 for the models);
- macro F1 (column "Macro F1"): each label's F1 on those sentences, averaged over the
  labels with at least one mention among them.

Methods and test sentences:
- SOCCAT (one NLI model per broad class), single NLI model, fine-tuned classifier,
  zero-shot NLI: pooled out-of-fold predictions of the 5-fold CV, all 7,234 sentences;
- Claude Sonnet-5 (high thinking effort, long prompt, the best LLM configuration):
  its random sample of 1,446 sentences (llm/classification/output/step_2/);
- dictionary (alpha0 = high, z > 1.96, the best variant): its own held-out split of
  1,447 sentences (dictionary/data/test.csv), predictions recomputed here exactly as
  in dictionary/03_evaluate_dictionary.py.
The LLM and the dictionary are evaluated on fewer sentences; the CSV also gives
SOCCAT's scores on exactly the same sentences (not shown in the LaTeX table).

Outputs: tables/model_comparison_overview.tex / .csv
Usage:   python model_comparison_overview.py   (run from anywhere)
"""

import json
import sys
from importlib import import_module
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
MAIN_DIR = REPO / "data" / "model_performance" / "step_2"
EMB_DIR = REPO / "embedding-based_classifiers"
LLM_RUN = REPO / "llm" / "classification" / "output" / "step_2" / "claude-sonnet-5-high__long" / "predictions.csv"
DICT_DIR = REPO / "dictionary"
DICT_VARIANT = ("high", 1.96)
OUT_TEX, OUT_CSV = ROOT / "model_comparison_overview.tex", ROOT / "model_comparison_overview.csv"
CLASS_FOLDERS = ["age_family_status", "identities_minority_majority_status", "labor_market_position", "profession",
                 "real_estate_ownership", "social_deviance", "social_roles_behavior", "socio_economic_position"]


# ── Gold labels and predictions as sentence x label 0/1 matrices ────────────
def read_pairs(files, split_dimension=False):
    d = pd.concat([pd.read_csv(f, usecols=["sentence_id", "hypothesis_label", "nli_label", "pred_label"]) for f in files])
    if split_dimension:   # "Dimension | label" -> label
        d["hypothesis_label"] = d["hypothesis_label"].str.split(" | ", n=1, regex=False).str[1]
    d["label"] = d["hypothesis_label"].str.lower()
    assert not d.duplicated(["sentence_id", "label"]).any()
    return d


def matrix(d, value, index, labels):
    m = (d.assign(v=(d[value] == 0).astype(np.int8))
         .pivot(index="sentence_id", columns="label", values="v").reindex(index=index, columns=labels))
    assert not m.isna().any().any(), "missing (sentence, label) pairs"
    return m.to_numpy()


def from_sets(pred_sets: dict, index, labels):
    col = {l: j for j, l in enumerate(labels)}
    m = np.zeros((len(index), len(labels)), dtype=np.int8)
    for i, sid in enumerate(index):
        for l in pred_sets.get(sid, ()):
            if l in col:
                m[i, col[l]] = 1
    return m


def llm_predictions():
    p = pd.read_csv(LLM_RUN, usecols=["id", "pred_categories"])
    p = p[p["pred_categories"] != "ERROR"]
    return {int(r.id): {s.lower() for _, s in json.loads(r.pred_categories)} for r in p.itertuples()}


def dictionary_predictions():
    """As in dictionary/03_evaluate_dictionary.py: a label is predicted if any word of
    its dictionary is a token of the sentence."""
    sys.path.insert(0, str(DICT_DIR))
    from tokenizer import tokenize
    z_suffix = import_module("02_build_dictionary").z_suffix
    load_dictionaries = import_module("03_evaluate_dictionary").load_dictionaries
    dicts = load_dictionaries(DICT_DIR / "dictionaries" / f"{DICT_VARIANT[0]}_{z_suffix(DICT_VARIANT[1])}")
    test = pd.read_csv(DICT_DIR / "data" / "test.csv", keep_default_na=False)
    preds = {}
    for r in test.itertuples():
        toks = set(tokenize(r.text, r.country))
        preds[int(r.id)] = {label.lower() for (_, label), words in dicts.items() if toks & words}
    return preds


# ── Scores ──────────────────────────────────────────────────────────────────
def scores(G, P):
    tp, fp, fn = (G & P).sum(0), ((1 - G) & P).sum(0), (G & (1 - P)).sum(0)
    T, F, N = tp.sum(), fp.sum(), fn.sum()
    has_pos = (tp + fn) > 0
    f1_label = np.where(2 * tp + fp + fn > 0, 2 * tp / np.maximum(2 * tp + fp + fn, 1), 0.0)
    return {"precision": T / (T + F) if T + F else 0.0, "recall": T / (T + N) if T + N else 0.0,
            "f1": 2 * T / (2 * T + F + N), "mean_f1_groups": f1_label[has_pos].mean(), "n_groups": int(has_pos.sum())}


def main():
    main_pairs = read_pairs([f for c in CLASS_FOLDERS for f in sorted((MAIN_DIR / c).glob("fold_*_human_vs_model.csv"))])
    labels = sorted(main_pairs["label"].unique())
    assert len(labels) == 57
    sentences = np.array(sorted(main_pairs["sentence_id"].unique()))
    G = matrix(main_pairs, "nli_label", sentences, labels)
    P_soccat = matrix(main_pairs, "pred_label", sentences, labels)
    pos = {sid: i for i, sid in enumerate(sentences)}

    # rows named by formulation: the first four all use mDeBERTa-v3-base (see the note)
    methods = [("SOCCAT (NLI, one model per broad class)", P_soccat, sentences)]
    for title, folder in [("NLI, single model (all groups)", "single_nli"),
                          ("Multi-label classifier (fine-tuned)", "fine_tuned_classifier"),
                          ("NLI, zero-shot (no fine-tuning)", "zero_shot_nli")]:
        d = read_pairs(sorted((EMB_DIR / folder / "data").glob("fold_*_human_vs_model.csv")), split_dimension=True)
        assert (matrix(d, "nli_label", sentences, labels) == G).all(), f"{folder}: different gold labels"
        methods.append((title, matrix(d, "pred_label", sentences, labels), sentences))
    for title, preds in [("Claude Sonnet-5 (high thinking, long prompt)", llm_predictions()),
                         (r"Dictionary ($\alpha_0$ = high, $z > 1.96$)", dictionary_predictions())]:
        ids = np.array(sorted(set(preds) & set(pos)))
        missing = set(preds) - set(pos)
        assert not missing, f"{title}: {len(missing)} test sentences not among SOCCAT's"
        methods.append((title, from_sets(preds, ids, labels), ids))

    rows = []
    for title, P, ids in methods:
        idx = [pos[s] for s in ids]
        s = scores(G[idx], P)
        s_soccat = scores(G[idx], P_soccat[idx])
        rows.append({"method": title, "n_sentences": len(ids), **s, "soccat_f1_same_sentences": s_soccat["f1"],
                     "soccat_mean_f1_groups_same_sentences": s_soccat["mean_f1_groups"]})
    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)

    lines = [
        r"\begin{table}[htb]",
        r"\centering",
        r"\scriptsize",
        r"\begin{threeparttable}",
        r"\caption{Step 2 performance of SOCCAT and the compared methods.}",
        r"\label{tab:model-comparison-overview}",
        r"\begin{tabular}{l r r r r r}",
        r"\toprule",
        r"Method & Sentences & Precision & Recall & F1 & Macro F1 \\",
        r"\midrule",
    ]
    for i, r in df.iterrows():
        lines.append(f"{r['method']} & {r['n_sentences']:,} & {r['precision']:.2f} & {r['recall']:.2f} & "
                     f"{r['f1']:.2f} & {r['mean_f1_groups']:.2f} \\\\")
        if i == 0:
            lines.append(r"\midrule")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\begin{tablenotes}[flushleft]",
        r"\footnotesize",
        r"\item \textit{Note:} Positive-class scores over the 57 group labels (\textit{others} excluded), pooled "
        r"over all (sentence, group label) pairs; macro F1: unweighted mean of the groups' F1. SOCCAT, the "
        r"single NLI model, the multi-label classifier and zero-shot NLI all use mDeBERTa-v3-base. Held-out "
        r"predictions of 5-fold cross-validation, except for Claude Sonnet-5 (random sample) and the "
        r"dictionary (held-out split).",
        r"\end{tablenotes}",
        r"\end{threeparttable}",
        r"\end{table}",
        "",
    ]
    OUT_TEX.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved file as: {OUT_TEX}\nSaved file as: {OUT_CSV}\n")
    with pd.option_context("display.width", 200):
        print(df.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
