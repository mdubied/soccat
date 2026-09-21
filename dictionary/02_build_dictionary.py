#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build per-label word dictionaries from the training split
============================================================
For each of the 57 taxonomy labels, mines a ranked list of words that are
distinctively associated with sentences carrying that label, using the
log-odds-ratio with informative Dirichlet prior (Monroe, Colaresi & Quinn,
2008, "Fightin' Words: Lexical Feature Selection and Evaluation for
Identifying the Content of Political Conflict", Political Analysis 16(4),
372-403). Equation numbers referenced in comments below are from that paper.

For label L:
    positive corpus (i) = train sentences whose labels include L
    background corpus (j) = all other train sentences
    prior                 = word frequencies over the full train corpus,
                             scaled by alpha0 (eq. 23) -- see _choose_alpha0()

Tokenization (see tokenizer.py, shared with 03_evaluate_dictionary.py):
lowercase unigrams, stopwords removed (stopwords-iso, pooled French +
German). No stemming/lemmatization and no multi-word phrases in this v1 --
kept deliberately simple for a baseline comparison, not tuned for best
recall (stemming was tried and reverted, see tokenizer.py docstring).

Alpha0 robustness sweep: eq. (23)'s alpha0 (see _choose_alpha0()) is a free
"prior sample size" the paper sets deliberately (p.388). To check how much
the resulting dictionaries depend on it, --alpha0-mode selects one of three
pre-registered points on the shrinkage-strength spectrum (all computed from
train.csv only, never from evaluation output):
    low    = medium / 10
    medium = median positive-corpus token count across the 57 labels
             (mirrors the paper's own calibration to "typical group volume")
    high   = full train-corpus token count (the degenerate/maximal case)
Run all three at once with 04_run_alpha0_sweep.py.

Output: dictionary/dictionaries/<mode>/<category>__<label>.txt, one word per
line, plus summary.csv (word counts/top words) and alpha0.json (the resolved
alpha0 value and how it was computed) in that same <mode> folder.
"""

import argparse
import json
import math
import re
import statistics
from collections import Counter
from pathlib import Path

import pandas as pd

from tokenizer import tokenize

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
DICTIONARIES_DIR = ROOT / "dictionaries"
CATEGORIES_FILE = ROOT.parent / "src" / "step_2" / "categories.json"

TOP_K = 20          # max words kept per label
MIN_POS_FREQ = 3    # a word must appear at least this many times in the positive corpus


def log_odds_dirichlet(pos_counts: Counter, bg_counts: Counter, prior_counts: Counter,
                        alpha0: float) -> dict:
    """Monroe et al. (2008) log-odds-ratio with informative Dirichlet prior.

    alpha0 is the total prior pseudo-count ("prior sample size", eq. 23) --
    see _choose_alpha0() for how it is set. Returns {word: z_score} for
    every word seen in pos_counts or bg_counts.
    """
    n_pos = sum(pos_counts.values())
    n_bg = sum(bg_counts.values())
    n_prior = sum(prior_counts.values())  # n in eq. 23 -- full train-corpus token count

    vocab = set(pos_counts) | set(bg_counts)
    scores = {}
    for w in vocab:
        y_pos = pos_counts.get(w, 0)
        y_bg = bg_counts.get(w, 0)

        # eq. (23): alpha_w = alpha0 * pi_hat_w^MLE = alpha0 * (y_w / n)
        # -- prior pseudo-count for w, proportional to its background rate.
        a_w = alpha0 * (prior_counts.get(w, 0) / n_prior) + 1e-9  # +eps: avoid zero prior

        # eq. (16): delta_w^(i-j), pairwise log-odds-ratio between the positive
        # corpus (i) and the background corpus (j), sharing the same prior a_w/alpha0.
        log_odds_pos = math.log(y_pos + a_w) - math.log(n_pos + alpha0 - y_pos - a_w)
        log_odds_bg = math.log(y_bg + a_w) - math.log(n_bg + alpha0 - y_bg - a_w)
        delta = log_odds_pos - log_odds_bg

        # eq. (20): approximate variance of delta (drops the n>>y terms present in
        # the full eq. 19 -- paper notes this is negligible for moderate-size corpora).
        variance = 1.0 / (y_pos + a_w) + 1.0 / (y_bg + a_w)

        # eq. (22): z-score (the paper's zeta, zeta-hat) = delta / sqrt(variance).
        scores[w] = delta / math.sqrt(variance)
    return scores


def resolve_alpha0(mode: str, label_pos_counts: list, n_prior: int) -> tuple:
    """Resolve --alpha0-mode to a numeric alpha0 (eq. 23's free "prior sample
    size") plus a human-readable description of how it was computed, without
    looking at any evaluation output.

    Monroe et al. (p.388) set alpha0 to "imply a 'prior sample' of 500 words per
    party every day, roughly the average number of words per day used per party
    on each topic in the data set" -- i.e. a value on the same scale as a
    *typical* group's own data volume, not an arbitrary constant and not the
    full corpus size. 'medium' mirrors that logic directly; 'low' and 'high'
    are a pre-registered x10/full-corpus spread around it for a robustness
    check on how much the dictionaries depend on this choice.
    """
    n_pos_tokens = [sum(pc.values()) for pc in label_pos_counts]
    medium = statistics.median(n_pos_tokens)

    if mode == "medium":
        return medium, "median positive-corpus token count across the 57 labels (train.csv only)"
    if mode == "low":
        return medium / 10, "medium / 10"
    if mode == "high":
        return float(n_prior), "full train-corpus token count (n in eq. 23; the degenerate/maximal case)"
    raise ValueError(f"unknown alpha0-mode: {mode}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha0-mode", choices=["low", "medium", "high"], default="medium",
                         help="which pre-registered alpha0 (eq. 23) to use -- see module docstring")
    args = parser.parse_args()

    train = pd.read_csv(DATA_DIR / "train.csv", keep_default_na=False)
    train["label_set"] = train["labels"].apply(lambda s: set(s.split(";")) if s else set())
    train["tokens"] = train.apply(lambda r: tokenize(r["text"], r["country"]), axis=1)

    with open(CATEGORIES_FILE, encoding="utf-8") as f:
        categories = json.load(f)

    # Word frequencies over the whole train corpus -> used as the Dirichlet prior (eq. 23)
    prior_counts = Counter()
    for toks in train["tokens"]:
        prior_counts.update(toks)

    # First pass: compute pos/bg token counts per label once (reused for both alpha0
    # selection and the actual scoring below).
    label_data = []
    for cat in categories:
        for label in cat["labels"]:
            is_pos = train["label_set"].apply(lambda s: label in s)
            pos_counts = Counter()
            for toks in train.loc[is_pos, "tokens"]:
                pos_counts.update(toks)
            bg_counts = Counter()
            for toks in train.loc[~is_pos, "tokens"]:
                bg_counts.update(toks)
            label_data.append({
                "category": cat["name"], "label": label,
                "n_pos_sentences": int(is_pos.sum()),
                "pos_counts": pos_counts, "bg_counts": bg_counts,
            })

    n_prior = sum(prior_counts.values())
    alpha0, alpha0_description = resolve_alpha0(
        args.alpha0_mode, [d["pos_counts"] for d in label_data], n_prior)
    print(f"alpha0-mode = {args.alpha0_mode}  ->  alpha0 (eq. 23 prior sample size) = "
          f"{alpha0:.0f}  ({alpha0_description})\n")

    out_dir = DICTIONARIES_DIR / args.alpha0_mode
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "alpha0.json", "w", encoding="utf-8") as f:
        json.dump({
            "mode": args.alpha0_mode,
            "alpha0": alpha0,
            "description": alpha0_description,
        }, f, indent=2)
    summary_rows = []

    for d in label_data:
        scores = log_odds_dirichlet(d["pos_counts"], d["bg_counts"], prior_counts, alpha0)
        # keep words that are (a) positively associated, (b) frequent enough
        # in the positive corpus to not be a single-sentence fluke
        candidates = [w for w, z in scores.items()
                      if z > 0 and d["pos_counts"].get(w, 0) >= MIN_POS_FREQ]
        ranked = sorted(candidates, key=lambda w: -scores[w])[:TOP_K]

        fname = f"{d['category']}__{d['label']}".replace("/", "-")
        fname = re.sub(r'[<>:"\\|?*]', "_", fname) + ".txt"
        with open(out_dir / fname, "w", encoding="utf-8") as f:
            f.write("\n".join(ranked))

        summary_rows.append({
            "category": d["category"], "label": d["label"],
            "n_pos_train": d["n_pos_sentences"],
            "n_words": len(ranked),
            "top_words": ", ".join(ranked[:10]),
        })
        print(f"  {d['category']:28s} {d['label']:60s} n_pos={d['n_pos_sentences']:4d}  "
              f"n_words={len(ranked):3d}  top={ranked[:5]}")

    pd.DataFrame(summary_rows).to_csv(out_dir / "summary.csv", index=False)
    print(f"\nDone. Dictionaries written to: {out_dir}")


if __name__ == "__main__":
    main()
