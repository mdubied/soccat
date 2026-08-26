#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build per-label word dictionaries from the training split
============================================================
For each of the 57 taxonomy labels, mines a ranked list of words that are
distinctively associated with sentences carrying that label, using the
log-odds-ratio with informative Dirichlet prior (Monroe, Colaresi & Quinn,
2008, "Fightin' Words: Lexical Feature Selection and Evaluation for
Identifying the Content of Political Conflict"). This is a standard method
for finding words that distinguish one corpus from a background corpus,
robust to rare words (unlike a raw frequency ratio).

For label L:
    positive corpus  = train sentences whose labels include L
    background corpus = all other train sentences
    prior             = word frequencies over the full train corpus

Tokenization: lowercase unigrams (unicode word characters), stopwords
removed (pooled French + German list from the stopwords-iso project, since
the corpus is bilingual). No stemming/lemmatization and no multi-word
phrases in this v1 — kept deliberately simple for a baseline comparison,
not tuned for best recall.

Output: dictionary/dictionaries/<category>__<label>.txt, one word per line,
plus dictionary/dictionaries/summary.csv with word counts and top words.
"""

import json
import math
import re
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "dictionaries"
RESOURCES_DIR = ROOT / "resources"
CATEGORIES_FILE = ROOT.parent / "src" / "step_2" / "categories.json"

TOP_K = 20          # max words kept per label
MIN_POS_FREQ = 3    # a word must appear at least this many times in the positive corpus

WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)

# Pooled French + German stopwords, since sentences are French or German and
# we build one combined dictionary (a French sentence will simply never
# match German stopwords/vocabulary anyway). Source: stopwords-iso project
# (https://github.com/stopwords-iso), MIT licensed, checked in verbatim as
# dictionary/resources/stopwords-{fr,de}.json for reproducibility.
def _load_stopwords(lang: str) -> set:
    with open(RESOURCES_DIR / f"stopwords-{lang}.json", encoding="utf-8") as f:
        return set(json.load(f))


STOPWORDS = _load_stopwords("fr") | _load_stopwords("de")


def tokenize(text: str) -> list:
    return [w for w in WORD_RE.findall(str(text).lower())
            if len(w) >= 2 and w not in STOPWORDS]


def log_odds_dirichlet(pos_counts: Counter, bg_counts: Counter, prior_counts: Counter) -> dict:
    """Monroe et al. (2008) log-odds-ratio with informative Dirichlet prior.

    The prior corpus is the full train corpus (pos + bg combined), used
    unscaled -- the standard formulation (e.g. as in the "Fightin' Words"
    reference implementation): alpha_w = count of w in the full corpus,
    alpha0 = total token count of the full corpus.

    Returns {word: z_score} for every word seen in pos_counts or bg_counts.
    """
    n_pos = sum(pos_counts.values())
    n_bg = sum(bg_counts.values())
    alpha0 = sum(prior_counts.values())

    vocab = set(pos_counts) | set(bg_counts)
    scores = {}
    for w in vocab:
        y_pos = pos_counts.get(w, 0)
        y_bg = bg_counts.get(w, 0)
        a_w = prior_counts.get(w, 0) + 1e-6  # avoid zero prior for unseen words

        log_odds_pos = math.log(y_pos + a_w) - math.log(n_pos + alpha0 - y_pos - a_w)
        log_odds_bg = math.log(y_bg + a_w) - math.log(n_bg + alpha0 - y_bg - a_w)
        delta = log_odds_pos - log_odds_bg

        variance = 1.0 / (y_pos + a_w) + 1.0 / (y_bg + a_w)
        scores[w] = delta / math.sqrt(variance)
    return scores


def main():
    train = pd.read_csv(DATA_DIR / "train.csv", keep_default_na=False)
    train["label_set"] = train["labels"].apply(lambda s: set(s.split(";")) if s else set())
    train["tokens"] = train["text"].apply(tokenize)

    with open(CATEGORIES_FILE, encoding="utf-8") as f:
        categories = json.load(f)

    # Word frequencies over the whole train corpus -> used as the Dirichlet prior
    prior_counts = Counter()
    for toks in train["tokens"]:
        prior_counts.update(toks)

    OUT_DIR.mkdir(exist_ok=True)
    summary_rows = []

    for cat in categories:
        for label in cat["labels"]:
            is_pos = train["label_set"].apply(lambda s: label in s)
            pos_counts = Counter()
            for toks in train.loc[is_pos, "tokens"]:
                pos_counts.update(toks)
            bg_counts = Counter()
            for toks in train.loc[~is_pos, "tokens"]:
                bg_counts.update(toks)

            scores = log_odds_dirichlet(pos_counts, bg_counts, prior_counts)
            # keep words that are (a) positively associated, (b) frequent enough
            # in the positive corpus to not be a single-sentence fluke
            candidates = [w for w, z in scores.items()
                          if z > 0 and pos_counts.get(w, 0) >= MIN_POS_FREQ]
            ranked = sorted(candidates, key=lambda w: -scores[w])[:TOP_K]

            fname = f"{cat['name']}__{label}".replace("/", "-")
            fname = re.sub(r'[<>:"\\|?*]', "_", fname) + ".txt"
            with open(OUT_DIR / fname, "w", encoding="utf-8") as f:
                f.write("\n".join(ranked))

            summary_rows.append({
                "category": cat["name"],
                "label": label,
                "n_pos_train": int(is_pos.sum()),
                "n_words": len(ranked),
                "top_words": ", ".join(ranked[:10]),
            })
            print(f"  {cat['name']:28s} {label:60s} n_pos={int(is_pos.sum()):4d}  "
                  f"n_words={len(ranked):3d}  top={ranked[:5]}")

    pd.DataFrame(summary_rows).to_csv(OUT_DIR / "summary.csv", index=False)
    print(f"\nDone. Dictionaries written to: {OUT_DIR}")


if __name__ == "__main__":
    main()
