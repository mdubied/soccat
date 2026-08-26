#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shared tokenizer for the dictionary baseline (build + evaluate)
====================================================================
Used identically by 02_build_dictionary.py (train) and
03_evaluate_dictionary.py (test) -- they must tokenize the same way.

Pipeline: lowercase unigrams -> stopword removal (stopwords-iso, pooled
French+German). No stemming/lemmatization: tried as a robustness variant
(nltk's Porter/Snowball stemmer, per-sentence via the `country` column --
mirrors Monroe, Colaresi & Quinn 2008, footnote 2, whose own running
example stems words) and reverted. It merged inflected-form duplicates as
intended, but on the full alpha0 sweep it traded precision for recall and
made per-label F1 worse, not better (e.g. high-alpha0 mode: 0.343 -> 0.276),
while also making the output dictionaries much harder to read (stems like
"commerc", "industri" instead of whole words) -- not worth it for a
baseline meant to stay simple and legible.
"""

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESOURCES_DIR = ROOT / "resources"

WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)


def _load_stopwords(lang: str) -> set:
    with open(RESOURCES_DIR / f"stopwords-{lang}.json", encoding="utf-8") as f:
        return set(json.load(f))


STOPWORDS = _load_stopwords("fr") | _load_stopwords("de")


def tokenize(text: str, country: str = None) -> list:
    return [w for w in WORD_RE.findall(str(text).lower())
            if len(w) >= 2 and w not in STOPWORDS]
