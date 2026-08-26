#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shared tokenizer for the dictionary baseline (build + evaluate)
====================================================================
Used identically by 02_build_dictionary.py (train) and
03_evaluate_dictionary.py (test) -- they must tokenize the same way.

Pipeline: lowercase unigrams -> stopword removal (stopwords-iso, pooled
French+German). No stemming/lemmatization.
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
