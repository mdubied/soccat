#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LaTeX table for the seen/unseen recall robustness check
========================================================
Reads output/recall_by_seen_status.csv (from 01_seen_unseen_recall.py) and
writes one appendix table, output/recall_by_seen_status.tex: one block per
broad class (N row + one row per model), then a pooled Total block with
Wilson 95% CIs.

Columns: All / Seen / Partial overlap / Unseen (see 01's docstring for the
definitions). Cells with fewer than MIN_N pairs are shown as "--": recall on
e.g. 2 pairs is not interpretable. Any model present in the CSV is included,
in MODEL_ORDER, so an extra model (e.g. an LLM) only needs to be added to
the CSV.

Compile with `booktabs` and `threeparttable`.
Usage: python 02_make_latex_table.py   (run from anywhere)
"""

import sys
from importlib import import_module
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
DISPLAY_NAMES = import_module("01_seen_unseen_recall").DISPLAY_NAMES

INPUT_FILE = ROOT / "output" / "recall_by_seen_status.csv"
OUTPUT_FILE = ROOT / "output" / "recall_by_seen_status.tex"

MIN_N = 10
MODEL_ORDER = ["SOCCAT", "LLM", "Dictionary"]
STATUSES = [("all", "All"), ("seen", "Seen"), ("partial", "Partial overlap"), ("unseen", "Unseen")]

CAPTION = "Recall on mentions seen and unseen in the training data (best fold of each broad class)."
NOTE = (
    r"\textit{Note:} Reported scores are the recall of the positive class; precision and F1 "
    r"cannot be split by seen status, as false positives have no annotated mention. "
    r"A test mention is \emph{seen} if its content words (stopwords removed, stemmed) match "
    r"those of a training mention of the same category, \emph{partial overlap} if it shares at "
    r"least one content word or compound part with one, and \emph{unseen} otherwise. "
    r"\emph{All} also includes a few mentions with no content words (e.g.\ pronouns). "
    rf"Cells with fewer than {MIN_N} cases are not reported (--). "
    r"95\% Wilson confidence intervals in brackets."
)


def fmt_recall(row, with_ci=False):
    if row is None or row["n"] < MIN_N:
        return "--"
    s = f"{row['recall']:.2f}"
    if with_ci:
        s += rf" {{\tiny [{row['ci_low']:.2f}, {row['ci_high']:.2f}]}}"
    return s


def cell(df, broad_class, status, model):
    sub = df[(df["broad_class"] == broad_class) & (df["status"] == status) & (df["model"] == model)]
    return sub.iloc[0] if len(sub) else None


def n_for(df, broad_class, status):
    sub = df[(df["broad_class"] == broad_class) & (df["status"] == status)]
    return int(sub["n"].iloc[0]) if len(sub) else 0


def block(df, broad_class, models, title, with_ci):
    rows = [rf"\textbf{{{title}}} & " +
            " & ".join(rf"\textit{{N = {n_for(df, broad_class, s)}}}" for s, _ in STATUSES) + r" \\"]
    for model in models:
        rows.append(rf"\quad {model} & " +
                    " & ".join(fmt_recall(cell(df, broad_class, s, model), with_ci) for s, _ in STATUSES) +
                    r" \\")
    return rows


def main():
    df = pd.read_csv(INPUT_FILE)
    models = [m for m in MODEL_ORDER if m in set(df["model"])]

    body = []
    for bc in [b for b in DISPLAY_NAMES if b in set(df["broad_class"])]:
        body += [r"\addlinespace"] + block(df, bc, models, DISPLAY_NAMES[bc], with_ci=False)
    body += [r"\midrule"] + block(df, "Total", models, "Total", with_ci=True)

    tex = "\n".join([
        r"\begin{table}[htb]", r"\centering", r"\scriptsize",
        r"\begin{threeparttable}",
        rf"\caption{{{CAPTION}}}",
        r"\label{tab:unseen_recall}",
        r"\begin{tabular}{l r r r r}", r"\toprule",
        " & ".join(["Broad class / model"] + [name for _, name in STATUSES]) + r" \\",
        r"\midrule", *body, r"\bottomrule", r"\end{tabular}",
        r"\begin{tablenotes}[flushleft]", r"\footnotesize",
        rf"\item {NOTE}",
        r"\end{tablenotes}", r"\end{threeparttable}", r"\end{table}",
    ])
    OUTPUT_FILE.write_text(tex + "\n", encoding="utf-8")
    print(tex)
    print(f"\nSaved file as: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
