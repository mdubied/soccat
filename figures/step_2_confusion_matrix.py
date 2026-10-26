"""
step_2_confusion_matrix.py

Description:
Confusion matrices of the Step 2 NLI models (SOCCAT) vs the ground truth, one
per broad class, built exactly like the ground truth vs independent coder
matrices of tables/icr_step2_krippendorff.ipynb (shared code:
confusion_matrix_utils.py): rows = ground-truth labels, columns = SOCCAT's
labels, restricted to the specific labels of the class plus "Other broad
class" (a specific label of another class), "Others" (the residual label) and
"None" (no label). Cells show counts; colours show the share of the
ground-truth row.

Predictions: each broad class has its own model, cross-validated on 5 folds
(fold membership differs between classes). The 5 folds' held-out predictions
are pooled, so every sentence has one held-out prediction from each of the 8
class models, i.e. for all 57 specific labels; SOCCAT's label set for a
sentence is the union of the labels these models predict (pred_label == 0,
i.e. prob_entail > 0.5, as in fold_*_per_label.csv). A single best fold
would not allow this: a sentence in one class's best fold is usually not in
another class's best fold, so confusions across broad classes could not be
observed.

Ground truth: the 57 specific labels from the same files (nli_label == 0),
i.e. the gold the models were trained and evaluated on, plus the residual
label "others" from annotations_ground_truth.csv's specific_group_new column
(the curated labels the NLI pairs were built from), since no model predicts
"others". Its "Others" column is therefore empty by construction. The coder
matrices parse the raw spans instead, which maps a few legacy labels folded
into "others" in specific_group_new (e.g. "insured persons", "engineers") to
a specific group, so about 27 sentences have slightly different ground truth.

Sentences: the 7,234 sentences of the model files (non-empty ground truth),
one unit per sentence.

Data:
- data/model_performance/step_2/<class folder>/fold_*_human_vs_model.csv
- data/manual_annotations/step_2/annotations_ground_truth.csv (residual "others", specific_group_new)

Outputs (in step_2/confusion_matrix/):
- by_broad_class/confusion_<class>.csv / .pdf, by_broad_class/figure_widths.txt
- soccat_outcomes.csv / .txt: what SOCCAT predicted for each ground-truth label
  (same label, other label of the class, other broad class, nothing), as
  tables/icr/icr_step2_coder_outcomes.* for the coders

Usage (from this directory):
python step_2_confusion_matrix.py
"""
import sys
from pathlib import Path

import pandas as pd

FIGURES_DIR = Path(__file__).resolve().parent
REPO_ROOT = FIGURES_DIR.parent
sys.path.insert(0, str(REPO_ROOT / "src" / "step_2"))
from annotation_labels import CODEBOOK, GROUND_TRUTH_FILE, normalise_label  # noqa: E402

import confusion_matrix_utils as cmu  # noqa: E402

STEP_2_DATA_DIR = REPO_ROOT / "data" / "model_performance" / "step_2"
CLASS_FOLDERS = [
    "age_family_status", "identities_minority_majority_status", "labor_market_position", "profession",
    "real_estate_ownership", "social_deviance", "social_roles_behavior", "socio_economic_position",
]
OUTPUT_DIR = FIGURES_DIR / "step_2" / "confusion_matrix"
CM_DIR = OUTPUT_DIR / "by_broad_class"


def load_model_units() -> list:
    """[(ground-truth labels, SOCCAT labels)] per sentence, from the pooled held-out
    predictions of all 8 class models."""
    cols = ["sentence_id", "hypothesis_label", "nli_label", "pred_label"]
    pairs = pd.concat([
        pd.read_csv(f, usecols=cols)
        for folder in CLASS_FOLDERS
        for f in sorted((STEP_2_DATA_DIR / folder).glob("fold_*_human_vs_model.csv"))
    ])
    pairs["label"] = pairs["hypothesis_label"].map(normalise_label)
    n_labels = pairs["label"].nunique()
    assert n_labels == 57, f"expected the 57 specific labels, got {n_labels}"
    assert not pairs.duplicated(["sentence_id", "label"]).any(), "a (sentence, label) pair is predicted twice"
    per_sentence = pairs.groupby("sentence_id")["label"].size()
    assert (per_sentence == n_labels).all(), "some sentences lack a held-out prediction for some labels"

    gold = pairs[pairs["nli_label"] == 0].groupby("sentence_id")["label"].agg(frozenset)
    pred = pairs[pairs["pred_label"] == 0].groupby("sentence_id")["label"].agg(frozenset)

    # "others" from the curated specific_group_new column (the one the NLI pairs were built
    # from), not from the raw spans: legacy span labels such as "insured persons" or
    # "engineers" were folded into "others" there, but the ICR parser maps them to a group
    ann = pd.read_csv(GROUND_TRUTH_FILE, usecols=["id", "specific_group_new"]).set_index("id")["specific_group_new"]
    sentence_ids = per_sentence.index
    assert set(sentence_ids) <= set(ann.index), "model sentences missing from the annotations"
    has_others = {sid for sid in sentence_ids
                  if cmu.RESIDUAL in {p.strip().lower() for p in str(ann[sid]).split(";")}}

    empty = frozenset()
    units = [(gold.get(sid, empty) | ({cmu.RESIDUAL} if sid in has_others else empty), pred.get(sid, empty))
             for sid in sentence_ids]
    print(f"{len(units):,} sentences | with a specific gold label: {len(gold):,} | "
          f"with 'others' in the annotations: {len(has_others):,} | with a SOCCAT prediction: {len(pred):,}")
    return units


def main():
    units = load_model_units()
    cmu.write_class_confusions(units, CODEBOOK, CM_DIR, xlabel="SOCCAT")
    print(cmu.write_figure_widths(CM_DIR))

    outcomes = cmu.outcome_table(units, CODEBOOK)
    outcomes.to_csv(OUTPUT_DIR / "soccat_outcomes.csv")
    summary = cmu.outcome_summary(outcomes, who="SOCCAT", units=units, codebook=CODEBOOK)
    (OUTPUT_DIR / "soccat_outcomes.txt").write_text(summary + "\n", encoding="utf-8")
    print("\n" + summary)


if __name__ == "__main__":
    main()
