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

Pairs that are not "seen" (partial overlap + unseen) are split into N_BINS
equal-size bins by proximity; seen pairs form one extra reference group.
Recall per bin and model, with Wilson 95% CIs.

Outputs (unseen_generalization/output/):
    proximity_pairs.csv            one row per test pair: proximity, the test
                                    mention and its nearest training mention
    recall_by_proximity_bin.csv    recall per bin x model (+ seen reference)
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
    train = training_mentions(mentions, base.load_best_folds())

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

    # equal-size bins over the non-seen pairs; seen pairs as reference group
    non_seen = prox["status"].isin(["partial", "unseen"])
    prox["bin"] = "seen"
    prox.loc[non_seen, "bin"] = pd.qcut(prox.loc[non_seen, "proximity"], N_BINS,
                                        labels=[f"Q{i + 1}" for i in range(N_BINS)]).astype(str)
    prox.loc[~non_seen & (prox["status"] != "seen"), "bin"] = pd.NA
    prox.to_csv(OUTPUT_DIR / "proximity_pairs.csv", index=False)

    rows = []
    for b in [f"Q{i + 1}" for i in range(N_BINS)] + ["seen"]:
        sub = prox[prox["bin"] == b]
        for model, col in models:
            k, n = int(sub[col].sum()), len(sub)
            lo, hi = base.wilson_ci(k, n)
            rows.append({"bin": b, "model": model, "n": n, "hits": k, "recall": k / n,
                         "ci_low": lo, "ci_high": hi,
                         "prox_min": sub["proximity"].min(), "prox_max": sub["proximity"].max(),
                         "prox_median": sub["proximity"].median(),
                         "share_unseen": (sub["status"] == "unseen").mean()})
    recall = pd.DataFrame(rows)
    recall.to_csv(OUTPUT_DIR / "recall_by_proximity_bin.csv", index=False)

    print("\nRecall by proximity bin:")
    print(recall.pivot_table(index="bin", columns="model", values="recall", sort=False).round(2).to_string())
    print(recall.drop_duplicates("bin")[["bin", "n", "prox_min", "prox_max", "share_unseen"]]
          .round(2).to_string(index=False))
    print(f"\nDone. Outputs written to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
