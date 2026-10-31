#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 1: number of social group references per sentence, by newspaper
====================================================================
Share of sentences (%) in the manually coded Step 1 sample (gold standard, train +
test, 6,300 sentences) with 0, 1, 2 and 3 or more references, i.e. "HasGroup" spans.
Rows per outlet, per country (French / German) and for all sentences; the 5
sentences without an outlet count only in their country and in "All".

Outputs: tables/step_1_groups_per_sentence.csv (rounded to whole percentages)
Usage:   python step_1_groups_per_sentence.py   (run from anywhere)
"""

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
STEP_1 = ROOT.parent / "data" / "model_performance" / "step_1"
FILES = ["train_with_all_outlets.json", "test_with_all_outlets.json"]
OUT_CSV = ROOT / "step_1_groups_per_sentence.csv"

# "paper" value in the data -> name in the paper, in table order
OUTLETS = {
    "Figaro": "Le Figaro", "Le Monde": "Le Monde", "Le Monde Diplomatique": "Le Monde Diplomatique",
    "Parisien": "Le Parisien", "Libération": "Libération", "Médiapart": "Mediapart",
    "Bild": "Bild", "Spiegel": "Der Spiegel", "Welt": "Die Welt", "Zeit": "Die Zeit",
    "FAZ": "Frankfurter Allgemeine Zeitung", "SZ": "Süddeutsche Zeitung",
}
BINS = ["0", "1", "2", "3 or more"]


def load():
    rows = []
    for name in FILES:
        with open(STEP_1 / name, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    d = json.loads(line.replace("NaN", "null"))
                    n = sum(1 for span in d.get("label") or [] if span[2] == "HasGroup")
                    rows.append({"paper": d["paper"], "language": d["language"], "n": n})
    return pd.DataFrame(rows)


def shares(n):
    pct = n.clip(upper=3).value_counts(normalize=True).reindex(range(4), fill_value=0) * 100
    return dict(zip([f"{b} (%)" for b in BINS], pct.round().astype(int)), N=len(n))


def main():
    df = load()
    assert len(df) == 6300, f"expected 6,300 sentences, got {len(df)}"
    unknown = set(df["paper"].dropna()) - set(OUTLETS)
    assert not unknown, f"unmapped outlets: {unknown}"

    rows = [{"Newspaper": OUTLETS[p], **shares(df.loc[df["paper"] == p, "n"])} for p in OUTLETS]
    rows += [{"Newspaper": lang, **shares(df.loc[df["language"] == lang, "n"])} for lang in ["French", "German"]]
    rows.append({"Newspaper": "All", **shares(df["n"])})

    table = pd.DataFrame(rows)
    table.to_csv(OUT_CSV, index=False, encoding="utf-8")
    print(table.to_string(index=False))
    print(f"[saved] {OUT_CSV}")


if __name__ == "__main__":
    main()
