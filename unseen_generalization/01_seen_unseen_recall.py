#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Recall of SOCCAT vs the dictionary baseline on seen vs unseen mentions
=======================================================================
Robustness check for the claim that SOCCAT generalizes to mentions absent
from its training data. For every broad class, uses its best fold (the one
whose model is published, see tables/step_2_best_fold_used.txt): the other
four folds are the training data, the best fold is the test set.

Unit of analysis: a gold-positive (sentence, label) pair of the best fold.
Gold labels are the ones SOCCAT was trained/evaluated on (nli_label == 0 in
data/model_performance/step_2/<folder>/fold_<k>_human_vs_model.csv), not a
re-derivation from the annotation file -- the two differ slightly (label
mapping, sentence subset), and SOCCAT must be scored on its own gold.

Mentions: the annotated character spans in
data/manual_annotations/step_2/annotations_ground_truth.csv. Each mention is
reduced to its set of content-word stems: dictionary tokenizer (lowercase,
stopwords removed, see dictionary/tokenizer.py) + Snowball stemmer (German
or French, by the sentence's country). Stopwords are removed so that a
function word ("les enfants" -> "les") can never make a mention "seen".

Seen status of a test mention, compared only with training mentions of the
SAME label in the other four folds:
    seen     its stem set is identical to a training mention's stem set
    partial  shares at least one stem with a training mention, or is a
             compound of one: one stem contains the other, the shorter
             being >= COMPOUND_MIN_LEN letters ('altertumswissenschaftl'
             vs seen 'wissenschaftl', 'hotelgast' vs seen 'gast'). 4 rather
             than 5 letters: it also catches short heads (gast, chef, bauer)
             at the cost of a few accidental matches, which can only make
             the unseen set stricter, never more favorable to SOCCAT.
    unseen   none of the above
A pair takes the most-seen status of its mentions (seen > partial > unseen),
so a pair is "unseen" only if ALL its mentions are -- deliberately strict.
Pairs whose mentions are all stopwords (e.g. pronouns) are "no_content";
gold pairs with no matching annotated span are "no_span". Both are kept in
the "all" row but excluded from the seen/partial/unseen split.

Metric: recall only. Seen status is a property of gold mentions, so false
positives cannot be assigned to seen/unseen and precision/F1 are undefined
per status. Overall precision/F1 are in tables/step_2_best_fold_metrics_*.

Dictionary baseline: rebuilt per broad class from the SAME training
sentences and gold labels, with the best-performing setting of the
dictionary/ sweep (alpha0-mode=high, z-threshold=1.96; see
dictionary/output/alpha0_sweep_comparison.csv). Scoring function, tokenizer
and constants are imported from dictionary/ -- nothing there is modified
or overwritten.

LLM reference (optional): the best step-2 LLM run (LLM_RUN: claude-sonnet-5,
high effort, long prompt). An LLM has no training data, so seen/unseen does
not apply to it the same way -- it shows how hard each status is in itself.
A pair counts as an LLM hit if the gold label is among the LLM's predicted
labels for the sentence. Predictions are only READ, from the existing
random-sample run (llm/classification/output/step_2/LLM_RUN, never modified)
plus a targeted run for the test sentences it doesn't cover
(output/llm/LLM_RUN). This script writes those sentence ids to
output/llm/ids_to_classify.txt; the LLM rows are added once every test
sentence has a prediction. To classify them (from llm/classification/):
    python classify_step2.py --prompt long --model claude-sonnet-5 --effort high \\
        --ids-file ../../unseen_generalization/output/llm/ids_to_classify.txt \\
        --output-root ../../unseen_generalization/output/llm
then rerun this script and 02_make_latex_table.py.

Outputs (unseen_generalization/output/):
    best_fold_pairs.csv          one row per test pair: mentions, status,
                                  SOCCAT/dictionary(/LLM) hit
    recall_by_seen_status.csv    recall + Wilson 95% CI per broad class x
                                  status x model (plus a pooled "Total")
    dictionaries_best_fold.json  the rebuilt dictionaries, for inspection
    llm/ids_to_classify.txt      test sentences still missing an LLM prediction

Usage: python 01_seen_unseen_recall.py   (run from anywhere)
"""

import json
import math
import re
import sys
from collections import Counter
from importlib import import_module
from pathlib import Path

import pandas as pd
from nltk.stem.snowball import SnowballStemmer

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
sys.path.insert(0, str(REPO / "dictionary"))
from tokenizer import tokenize  # noqa: E402

build = import_module("02_build_dictionary")
prep = import_module("01_prepare_data")

ANNOTATIONS_FILE = REPO / "data" / "manual_annotations" / "step_2" / "annotations_ground_truth.csv"
STEP_2_DATA_DIR = REPO / "data" / "model_performance" / "step_2"
BEST_FOLD_FILE = REPO / "tables" / "step_2_best_fold_used.txt"
OUTPUT_DIR = ROOT / "output"

LLM_RUN = "claude-sonnet-5-high__long"
LLM_RUN_DIRS = [REPO / "llm" / "classification" / "output" / "step_2" / LLM_RUN,  # read-only
                OUTPUT_DIR / "llm" / LLM_RUN]
LLM_IDS_FILE = OUTPUT_DIR / "llm" / "ids_to_classify.txt"

DICT_ALPHA0_MODE = "high"
DICT_Z_THRESHOLD = 1.96

# taxonomy category name -> (results folder, key used in step_2_best_fold_used.txt)
CATEGORIES = {
    "age_and_family_status":     ("age_family_status", "age_family"),
    "identities":                ("identities_minority_majority_status", "identity"),
    "labor_market_position":     ("labor_market_position", "labor_market_w_entrepreneurs"),
    "profession":                ("profession", "profession"),
    "real_estate_ownership":     ("real_estate_ownership", "real_estate"),
    "social_deviance":           ("social_deviance", "social_deviance"),
    "social_roles_and_behavior": ("social_roles_behavior", "social_roles"),
    "socio_economic_position":   ("socio_economic_position", "socio_economic"),
}
DISPLAY_NAMES = {
    "age_and_family_status": "Age and family status",
    "identities": "Identities and minority/majority status",
    "labor_market_position": "Labor market position",
    "profession": "Profession",
    "real_estate_ownership": "Real estate ownership",
    "social_deviance": "Social deviance",
    "social_roles_and_behavior": "Social roles and behavior",
    "socio_economic_position": "Socio-economic position",
}

# Raw span labels missing from dictionary/01_prepare_data.py's LABEL_MAP but
# counted as these labels in SOCCAT's training data (else their gold pairs
# have no mention). Only used to attach spans to gold pairs, never to add gold.
EXTRA_SPAN_LABELS = {
    "enterprises": "entrepreneurs",
    "religious groups": "multiple (or other) religious or minority groups",
    "minorities": "multiple (or other) religious or minority groups",
}

STEMMERS = {"Germany": SnowballStemmer("german"), "France": SnowballStemmer("french")}
STATUS_ORDER = ["seen", "partial", "unseen"]
COMPOUND_MIN_LEN = 4  # min length of the contained stem for a compound match


# ── Annotations ───────────────────────────────────────────────────────────────
def parse_spans(raw) -> list:
    """[(start, end, raw_label), ...] from the ground_truth column.

    Not ast.literal_eval: a few rows are malformed (unterminated quotes,
    labels containing brackets like 'entrepreneurs in [specific] sector'), so
    split on the '[start, end, ' markers instead.
    """
    if not isinstance(raw, str):
        return []
    marks = list(re.finditer(r"\[(\d+),\s*(\d+),\s*['\"]", raw))
    spans = []
    for i, m in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(raw)
        label = re.sub(r"['\"]?\]\s*,?\s*\]?\s*[,;]?\s*$", "", raw[m.end():end].rstrip(" ,["))
        spans.append((int(m[1]), int(m[2]), label.strip("' ")))
    return spans


def snap_span(text: str, start: int, end: int) -> tuple:
    """Fix spans shifted by a few characters, e.g. 'd[er mächtigsten ... Ukraine.]'.

    A span whose START falls inside a word is a shift artifact -> extend both
    ends to word boundaries. A span that only ENDS inside a word is kept as
    is: annotators deliberately marked the group part of compounds
    ('[Kinder]baracke', '[Flüchtlings]politik').
    """
    if 0 < start < len(text) and text[start - 1].isalpha() and text[start].isalpha():
        while start > 0 and text[start - 1].isalpha():
            start -= 1
        while end < len(text) and text[end - 1].isalpha() and text[end].isalpha():
            end += 1
        return start, end, True
    return start, end, False


def stems(text: str, country: str) -> frozenset:
    stemmer = STEMMERS.get(country, STEMMERS["Germany"])
    return frozenset(stemmer.stem(w) for w in tokenize(text, country))


def load_mentions(gt: pd.DataFrame) -> pd.DataFrame:
    rows, n_snapped = [], 0
    for r in gt.itertuples():
        for start, end, raw_label in parse_spans(r.ground_truth):
            label = prep.LABEL_MAP.get(raw_label.lower()) or EXTRA_SPAN_LABELS.get(raw_label.lower())
            if label is None:
                continue
            start, end, snapped = snap_span(r.text, start, end)
            n_snapped += snapped
            mention = r.text[start:end]
            rows.append({"sentence_id": r.id, "label": label, "raw_label": raw_label,
                         "mention": mention, "stems": stems(mention, r.country)})
    print(f"  {len(rows):,} taxonomy mentions ({n_snapped} shifted spans snapped to word boundaries)")
    return pd.DataFrame(rows)


# ── SOCCAT predictions / folds ────────────────────────────────────────────────
def load_best_folds() -> dict:
    key_to_cat = {key: cat for cat, (_, key) in CATEGORIES.items()}
    best = {}
    for line in BEST_FOLD_FILE.read_text(encoding="utf-8").splitlines():
        m = re.search(r"\((\w+)\): fold (\d+)", line)
        if m:
            best[key_to_cat[m[1]]] = int(m[2])
    assert set(best) == set(CATEGORIES), f"best folds missing for {set(CATEGORIES) - set(best)}"
    return best


def load_predictions(folder: str) -> pd.DataFrame:
    files = sorted((STEP_2_DATA_DIR / folder).glob("fold_*_human_vs_model.csv"))
    assert files, f"no fold_*_human_vs_model.csv in {STEP_2_DATA_DIR / folder}"
    return pd.concat(
        [pd.read_csv(f, usecols=["sentence_id", "hypothesis_label", "nli_label", "pred_label", "prob_entail"])
         .assign(fold=int(re.search(r"fold_(\d+)_", f.name)[1])) for f in files],
        ignore_index=True,
    )


# ── Dictionary baseline (same method as dictionary/02_build_dictionary.py) ────
def build_dictionaries(train_tokens: pd.Series, train_gold: pd.DataFrame, labels: list) -> dict:
    """{label: set(words)} from training sentences only.

    train_tokens: sentence_id -> token list; train_gold: gold-positive
    (sentence_id, label) pairs. Positive corpus = train sentences with the
    label, background = all other train sentences, prior = full train corpus.
    """
    prior_counts = Counter()
    for toks in train_tokens:
        prior_counts.update(toks)

    label_data = []
    for label in labels:
        pos_ids = set(train_gold.loc[train_gold["hypothesis_label"] == label, "sentence_id"])
        is_pos = train_tokens.index.isin(pos_ids)
        pos_counts, bg_counts = Counter(), Counter()
        for toks in train_tokens[is_pos]:
            pos_counts.update(toks)
        for toks in train_tokens[~is_pos]:
            bg_counts.update(toks)
        label_data.append((label, pos_counts, bg_counts))

    alpha0, _ = build.resolve_alpha0(DICT_ALPHA0_MODE, [pc for _, pc, _ in label_data],
                                     sum(prior_counts.values()))
    dicts = {}
    for label, pos_counts, bg_counts in label_data:
        scores = build.log_odds_dirichlet(pos_counts, bg_counts, prior_counts, alpha0)
        candidates = [w for w, z in scores.items()
                      if z > DICT_Z_THRESHOLD and pos_counts.get(w, 0) >= build.MIN_POS_FREQ]
        # tie-break on the word: log_odds_dirichlet iterates a set, so equal
        # scores at the TOP_K cutoff would otherwise vary between runs
        dicts[label] = sorted(candidates, key=lambda w: (-scores[w], w))[:build.TOP_K]
    return dicts


# ── Seen status ───────────────────────────────────────────────────────────────
def mention_status(mention_stems: frozenset, seen_sets: set, seen_stems: set) -> str:
    if not mention_stems:
        return "no_content"
    if mention_stems in seen_sets:
        return "seen"
    if mention_stems & seen_stems:
        return "partial"
    if any(is_compound_match(m, s) for m in mention_stems for s in seen_stems):
        return "partial"
    return "unseen"


def is_compound_match(a: str, b: str) -> bool:
    short, long_ = sorted((a, b), key=len)
    return len(short) >= COMPOUND_MIN_LEN and short in long_


def pair_status(statuses: list) -> str:
    if not statuses:
        return "no_span"
    for s in STATUS_ORDER:
        if s in statuses:
            return s
    return "no_content"


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple:
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


def recall_rows(pairs: pd.DataFrame, broad_class: str) -> list:
    rows = []
    models = [("SOCCAT", "soccat_hit"), ("LLM", "llm_hit"), ("Dictionary", "dict_hit")]
    subsets = [("all", pairs)] + [(s, pairs[pairs["status"] == s]) for s in STATUS_ORDER]
    for status, sub in subsets:
        for model, col in [(m, c) for m, c in models if c in pairs.columns]:
            n, k = len(sub), int(sub[col].sum())
            lo, hi = wilson_ci(k, n)
            rows.append({"broad_class": broad_class, "status": status, "model": model, "n": n,
                         "hits": k, "recall": k / n if n else float("nan"),
                         "ci_low": lo, "ci_high": hi})
    return rows


# ── LLM reference ─────────────────────────────────────────────────────────────
def load_llm_predictions() -> dict:
    """{sentence_id: set of predicted specific labels (lowercased)} over all LLM_RUN_DIRS.

    Rows that failed (pred_categories == 'ERROR') are left out so they get
    listed again in ids_to_classify.txt. Both runs must share model, effort
    and prompt.
    """
    preds, config = {}, None
    for run_dir in LLM_RUN_DIRS:
        path = run_dir / "predictions.csv"
        if not path.exists():
            continue
        meta = json.loads((run_dir / "run_meta.json").read_text(encoding="utf-8"))
        run_config = (meta["model"], meta["effort"], meta["prompt"])
        assert config in (None, run_config), f"LLM runs differ: {config} vs {run_config} ({run_dir})"
        config = run_config
        df = pd.read_csv(path, usecols=["id", "pred_categories"])
        for r in df[df["pred_categories"] != "ERROR"].itertuples():
            preds[int(r.id)] = {label.lower() for _, label in json.loads(r.pred_categories)}
    return preds


def add_llm_hits(pairs: pd.DataFrame) -> pd.DataFrame:
    """Add an llm_hit column if every test sentence has an LLM prediction; otherwise
    write the missing ids to LLM_IDS_FILE and leave pairs unchanged."""
    preds = load_llm_predictions()
    missing = sorted(set(pairs["sentence_id"]) - set(preds))
    LLM_IDS_FILE.parent.mkdir(parents=True, exist_ok=True)
    LLM_IDS_FILE.write_text("".join(f"{i}\n" for i in missing), encoding="utf-8")
    if missing:
        print(f"\nLLM rows skipped: {len(missing)} of {pairs['sentence_id'].nunique()} test sentences "
              f"have no {LLM_RUN} prediction yet (ids in {LLM_IDS_FILE}; see docstring to classify them).")
        return pairs
    print(f"\nLLM predictions ({LLM_RUN}) found for all {pairs['sentence_id'].nunique()} test sentences.")
    return pairs.assign(llm_hit=[int(label.lower() in preds[sid])
                                 for sid, label in zip(pairs["sentence_id"], pairs["label"])])


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    print("Loading annotations...")
    gt = pd.read_csv(ANNOTATIONS_FILE)
    mentions = load_mentions(gt)
    texts = gt.set_index("id")[["text", "country"]]
    best_folds = load_best_folds()

    all_pairs, all_dicts = [], {}
    for cat, (folder, _) in CATEGORIES.items():
        fold = best_folds[cat]
        preds = load_predictions(folder)
        gold = preds[preds["nli_label"] == 0]
        labels = sorted(preds["hypothesis_label"].unique())
        train_ids = set(preds.loc[preds["fold"] != fold, "sentence_id"])
        test_ids = set(preds.loc[preds["fold"] == fold, "sentence_id"])
        assert not train_ids & test_ids, f"{cat}: sentence in both train and test folds"

        # training mentions: annotated spans of gold-positive training pairs
        train_gold = gold[gold["sentence_id"].isin(train_ids)]
        train_m = mentions.merge(train_gold[["sentence_id", "hypothesis_label"]],
                                 left_on=["sentence_id", "label"],
                                 right_on=["sentence_id", "hypothesis_label"])
        seen_sets = train_m.groupby("label")["stems"].apply(lambda s: {x for x in s if x}).to_dict()
        seen_stems = train_m.groupby("label")["stems"].apply(lambda s: set().union(*s)).to_dict()

        # dictionary from the same training sentences
        train_tokens = pd.Series({sid: tokenize(texts.at[sid, "text"], texts.at[sid, "country"])
                                  for sid in sorted(train_ids)})
        dicts = build_dictionaries(train_tokens, train_gold, labels)
        all_dicts[cat] = dicts

        # test pairs
        test = gold[gold["fold"] == fold]
        m_by_pair = mentions.groupby(["sentence_id", "label"])
        pair_rows = []
        for r in test.itertuples():
            key = (r.sentence_id, r.hypothesis_label)
            ms = m_by_pair.get_group(key) if key in m_by_pair.groups else mentions.iloc[0:0]
            statuses = [mention_status(s, seen_sets.get(r.hypothesis_label, set()),
                                       seen_stems.get(r.hypothesis_label, set())) for s in ms["stems"]]
            sent_tokens = set(tokenize(texts.at[r.sentence_id, "text"], texts.at[r.sentence_id, "country"]))
            pair_rows.append({
                "broad_class": cat, "fold": fold, "sentence_id": r.sentence_id,
                "label": r.hypothesis_label, "country": texts.at[r.sentence_id, "country"],
                "mentions": " | ".join(ms["mention"]),
                "mention_statuses": " | ".join(statuses),
                "status": pair_status(statuses),
                "prob_entail": r.prob_entail,
                "soccat_hit": int(r.pred_label == 0),
                "dict_hit": int(bool(sent_tokens & set(dicts[r.hypothesis_label]))),
            })
        pairs = pd.DataFrame(pair_rows)
        all_pairs.append(pairs)

        counts = pairs["status"].value_counts().to_dict()
        print(f"  {cat:28s} fold {fold}  pairs={len(pairs):4d}  " +
              "  ".join(f"{s}={counts.get(s, 0)}" for s in STATUS_ORDER + ["no_content", "no_span"]))

    pairs = add_llm_hits(pd.concat(all_pairs, ignore_index=True))
    all_recall = []
    for cat, sub in pairs.groupby("broad_class", sort=False):
        all_recall += recall_rows(sub, cat)
    all_recall += recall_rows(pairs, "Total")
    recall = pd.DataFrame(all_recall)

    pairs.to_csv(OUTPUT_DIR / "best_fold_pairs.csv", index=False)
    recall.to_csv(OUTPUT_DIR / "recall_by_seen_status.csv", index=False)
    with open(OUTPUT_DIR / "dictionaries_best_fold.json", "w", encoding="utf-8") as f:
        json.dump({"alpha0_mode": DICT_ALPHA0_MODE, "z_threshold": DICT_Z_THRESHOLD,
                   "best_folds": best_folds, "dictionaries": all_dicts}, f, ensure_ascii=False, indent=2)

    wide = recall.pivot_table(index="broad_class", columns=["model", "status"], values="recall")
    print("\nRecall by seen status (best fold):")
    print(wide.round(2).to_string())
    print(f"\nDone. Outputs written to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
