#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Recall as a function of semantic proximity to the training mentions
====================================================================
Second part of the robustness check (see 01_seen_unseen_recall.py): does
SOCCAT recognise mentions that are absent from its training data but
semantically close to those it contains?

For every test pair of 01's output (best_fold_pairs.csv, best fold of each
broad class), each annotated mention is embedded on its own (no sentence
context: the claim is about the mention's words) with a multilingual
sentence-embedding model (EMBEDDING_MODEL; French and German share one
space). Its proximity is the cosine similarity to the most similar training
mention of the SAME label, training = the four other folds, exactly as in 01.
A pair with several mentions takes its most similar mention.

The pairs of interest are split into N_BINS equal-size bins by proximity;
seen pairs form one extra reference group. Recall per bin and model, with
Wilson 95% CIs; a "pooled" row gives recall over all pairs of interest (all
bins together). Six variants (VARIANTS x category filter):
    pairs of interest: partial overlap + no word overlap (main), partial
                       only, or no word overlap only
    categories:        all, or only labels whose SOCCAT median F1
                       (f1_binary across the 5 CV folds) is >= F1_MEDIAN_MIN;
                       the filter applies to the seen reference too
Bins are recomputed within each variant (equal-size within that subset).

The training folds are read from best_fold_pairs.csv's "fold" column, not
from the current best-fold file, so training and test always come from the
same run of 01 even if the best folds have changed since.

Outputs (unseen_generalization/output/):
    proximity_pairs.csv            one row per test pair: proximity, the test
                                    mention and its nearest training mention
                                    (bin = main variant, all categories)
    recall_by_proximity_bin[_<variant>].csv
                                   recall per bin x model (+ seen reference);
                                   no suffix = main variant, all categories
    status_shares.txt              share of seen / partial / no_overlap pairs,
                                   overall and per proximity quintile
    embeddings/<model>.npz         embedding cache (delete to recompute)

Requires 01_seen_unseen_recall.py to have been run first. The figure is made
by 04_plot_semantic_proximity.py.
Usage: python 03_semantic_proximity.py   (run from anywhere)
"""

import re
import sys
from importlib import import_module
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
base = import_module("01_seen_unseen_recall")

OUTPUT_DIR = base.OUTPUT_DIR
PAIRS_FILE = OUTPUT_DIR / "best_fold_pairs.csv"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
CACHE_FILE = OUTPUT_DIR / "embeddings" / (EMBEDDING_MODEL.split("/")[-1] + ".npz")

N_BINS = 5
MODELS = [("SOCCAT", "soccat_hit"), ("LLM", "llm_hit"), ("Dictionary", "dict_hit")]
STEP_2_DATA_DIR = base.REPO / "data" / "model_performance" / "step_2"
F1_MEDIAN_MIN = 0.70
# variant name -> statuses binned by proximity (seen is always the reference)
VARIANTS = {
    "": ["partial", "no_overlap"],
    "partial": ["partial"],
    "no_overlap": ["no_overlap"],
}
FILTER_SUFFIX = f"f1med{int(F1_MEDIAN_MIN * 100)}"


def clean(mention: str) -> str:
    return re.sub(r"\s+", " ", mention).strip(" \t\r\n\"'«»„“”,.;:()")


def embed(texts: list) -> dict:
    """{text: unit-norm embedding}, cached on disk per model."""
    cache = {}
    if CACHE_FILE.exists():
        data = np.load(CACHE_FILE, allow_pickle=False)
        cache = dict(zip(data["texts"].tolist(), data["vectors"]))
    todo = sorted(set(texts) - set(cache))
    if todo:
        from sentence_transformers import SentenceTransformer
        print(f"  embedding {len(todo):,} mentions with {EMBEDDING_MODEL} ...")
        model = SentenceTransformer(EMBEDDING_MODEL)
        vecs = model.encode(todo, batch_size=64, normalize_embeddings=True, show_progress_bar=True)
        cache.update(zip(todo, vecs))
        CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        keys = sorted(cache)
        np.savez_compressed(CACHE_FILE, texts=np.array(keys), vectors=np.stack([cache[k] for k in keys]))
    return cache


def soccat_f1_median() -> dict:
    """{label: median of SOCCAT's per-fold f1_binary across the 5 CV folds}."""
    out = {}
    for folder, _ in base.CATEGORIES.values():
        folds = pd.concat(pd.read_csv(f) for f in (STEP_2_DATA_DIR / folder).glob("fold_*_per_label.csv"))
        out.update(folds.groupby("hypothesis_label")["f1_binary"].median().to_dict())
    return out


def recall_by_bin(prox: pd.DataFrame, statuses: list, models: list) -> tuple:
    """Bin the pairs with a status in `statuses` into N_BINS equal-size proximity
    bins (seen pairs = reference). Returns (bin per pair, recall table)."""
    of_interest = prox["status"].isin(statuses)
    bins = pd.Series(pd.NA, index=prox.index, dtype="object")
    bins[prox["status"] == "seen"] = "seen"
    bins[of_interest] = pd.qcut(prox.loc[of_interest, "proximity"], N_BINS,
                                labels=[f"Q{i + 1}" for i in range(N_BINS)]).astype(str)
    rows = []
    for b in [f"Q{i + 1}" for i in range(N_BINS)] + ["pooled", "seen"]:
        # "pooled" = all pairs of interest, i.e. every Q bin together
        sub = prox[of_interest] if b == "pooled" else prox[bins == b]
        for model, col in models:
            k, n = int(sub[col].sum()), len(sub)
            lo, hi = base.wilson_ci(k, n)
            rows.append({"bin": b, "model": model, "n": n, "hits": k, "recall": k / n,
                         "ci_low": lo, "ci_high": hi,
                         "prox_min": sub["proximity"].min(), "prox_max": sub["proximity"].max(),
                         "prox_median": sub["proximity"].median(),
                         "share_no_overlap": (sub["status"] == "no_overlap").mean()})
    return bins, pd.DataFrame(rows)


def status_shares_text(prox: pd.DataFrame, title: str) -> list:
    """Text tables: share of each seen status in the test pairs, and per proximity
    quintile (a) over not-seen pairs as in the main figure, (b) over all pairs."""
    def table(df, group_col, statuses):
        counts = pd.crosstab(df[group_col], df["status"]).reindex(columns=statuses, fill_value=0)
        counts.loc["Total"] = counts.sum()
        shares = counts.div(counts.sum(axis=1), axis=0)
        out = pd.DataFrame({"N": counts.sum(axis=1)})
        for s in statuses:
            out[s] = [f"{n:5d} ({p * 100:5.1f}%)" for n, p in zip(counts[s], shares[s])]
        out.index.name = "quintile"
        return out.to_string().splitlines()

    statuses = ["seen", "partial", "no_overlap"]
    other = prox[~prox["status"].isin(statuses)]
    counts = prox["status"].value_counts().reindex(statuses, fill_value=0)
    lines = [f"=== {title} ===", "",
             f"Test pairs with a seen status: {counts.sum():,}"
             + (f" (excluded: {len(other)} pairs with status {sorted(other['status'].unique())})" if len(other) else "")]
    lines += [f"  {s:<8} {n:5d} ({n / counts.sum() * 100:5.1f}%)" for s, n in counts.items()]

    labels = [f"Q{i + 1}" for i in range(N_BINS)]
    not_seen = prox[prox["status"].isin(["partial", "no_overlap"])].copy()
    not_seen["q"] = pd.qcut(not_seen["proximity"], N_BINS, labels=labels)
    lines += ["", "(a) Quintiles of similarity over NOT-SEEN pairs (as in semantic_proximity.pdf):"]
    lines += ["  " + l for l in table(not_seen, "q", ["partial", "no_overlap"])]

    known = prox[prox["status"].isin(statuses)].copy()
    known["q"] = pd.qcut(known["proximity"], N_BINS, labels=labels)
    ranges = known.groupby("q", observed=True)["proximity"].agg(["min", "max"])
    lines += ["", "(b) Quintiles of similarity over ALL pairs (seen included):"]
    lines += ["  " + l for l in table(known, "q", statuses)]
    lines += ["  similarity ranges: " + ", ".join(f"{q} {r['min']:.2f}-{r['max']:.2f}"
                                                   for q, r in ranges.iterrows()), ""]
    return lines


def training_mentions(mentions: pd.DataFrame, best_folds: dict) -> pd.DataFrame:
    """Annotated mentions of gold-positive training pairs, per broad class (same as 01)."""
    rows = []
    for cat, (folder, _) in base.CATEGORIES.items():
        preds = base.load_predictions(folder)
        gold = preds[(preds["nli_label"] == 0) & (preds["fold"] != best_folds[cat])]
        rows.append(mentions.merge(gold[["sentence_id", "hypothesis_label"]],
                                   left_on=["sentence_id", "label"],
                                   right_on=["sentence_id", "hypothesis_label"]).assign(broad_class=cat))
    return pd.concat(rows, ignore_index=True)


def main():
    pairs = pd.read_csv(PAIRS_FILE)
    models = [(m, c) for m, c in MODELS if c in pairs.columns]
    print("Loading mentions...")
    gt = pd.read_csv(base.ANNOTATIONS_FILE)
    mentions = base.load_mentions(gt)
    mentions["text"] = mentions["mention"].map(clean)
    mentions = mentions[mentions["text"] != ""]
    # folds the test pairs were taken from (see module docstring)
    pair_folds = pairs.groupby("broad_class")["fold"].first().to_dict()
    train = training_mentions(mentions, pair_folds)

    test = mentions.merge(pairs[["broad_class", "sentence_id", "label"]], on=["sentence_id", "label"])
    vectors = embed(pd.concat([train["text"], test["text"]]).tolist())

    # nearest same-label training mention for every test mention
    nearest = []
    for (cat, label), t in test.groupby(["broad_class", "label"]):
        tr = train[(train["broad_class"] == cat) & (train["label"] == label)]
        if tr.empty:
            continue
        tr_texts = tr["text"].drop_duplicates().tolist()
        sims = np.stack([vectors[x] for x in t["text"]]) @ np.stack([vectors[x] for x in tr_texts]).T
        best = sims.argmax(axis=1)
        nearest.append(t.assign(proximity=sims.max(axis=1),
                                nearest_train_mention=[tr_texts[i] for i in best]))
    nearest = pd.concat(nearest, ignore_index=True)

    # a pair takes its most similar mention
    top = (nearest.sort_values("proximity", ascending=False)
           .drop_duplicates(["sentence_id", "label"])
           [["sentence_id", "label", "text", "nearest_train_mention", "proximity"]]
           .rename(columns={"text": "test_mention"}))
    prox = pairs.merge(top, on=["sentence_id", "label"], how="inner")
    print(f"  {len(prox):,} of {len(pairs):,} test pairs have a proximity "
          "(the rest have no mention with content, see 01)")

    # equal-size bins over the pairs of interest; seen pairs as reference group
    f1_median = soccat_f1_median()
    strong = prox["label"].map(f1_median) >= F1_MEDIAN_MIN
    print(f"  {prox.loc[strong, 'label'].nunique()} of {prox['label'].nunique()} labels have "
          f"SOCCAT median F1 >= {F1_MEDIAN_MIN}")
    for filtered in (False, True):
        subset = prox[strong] if filtered else prox
        for variant, statuses in VARIANTS.items():
            bins, recall = recall_by_bin(subset, statuses, models)
            name = "_".join(p for p in (variant, FILTER_SUFFIX if filtered else "") if p)
            recall.to_csv(OUTPUT_DIR / f"recall_by_proximity_bin{'_' + name if name else ''}.csv", index=False)
            if not name:
                prox.assign(bin=bins).to_csv(OUTPUT_DIR / "proximity_pairs.csv", index=False)
            print(f"\nRecall by proximity bin [{name or 'main'}]:")
            print(recall.pivot_table(index="bin", columns="model", values="recall", sort=False).round(2).to_string())
            print(recall.drop_duplicates("bin")[["bin", "n", "prox_min", "prox_max", "share_no_overlap"]]
                  .round(2).to_string(index=False))
    lines = []
    for filtered in (False, True):
        lines += status_shares_text(prox[strong] if filtered else prox,
                                    f"SOCCAT median F1 >= {F1_MEDIAN_MIN} categories only" if filtered
                                    else "All categories")
    (OUTPUT_DIR / "status_shares.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n" + "\n".join(lines))
    print(f"\nDone. Outputs written to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
