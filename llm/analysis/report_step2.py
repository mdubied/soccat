"""
Compute performance metrics and cost estimates across all Step 2 LLM
classification runs, and write a single combined txt report.

Scans llm/classification/output/step_2/ for every {model}__{prompt} run
directory produced by classify_step2.py, and reports on whichever ones it
finds -- it is not an error for some model/prompt combinations to be
missing (e.g. you haven't run --model claude-opus-5 yet). Kept separate
from classify_step2.py so the report can be regenerated (or runs compared)
without spending any more Claude credits.

Step 2 has no single label: each specific taxonomy label is scored as its
own binary detection problem (present vs. absent in that sentence), then:
  - broad-category scores are the weighted average of their specific-label
    scores, weighted by each label's number of positive (true) cases;
  - an overall score is the same weighted average across every specific
    label in the taxonomy.
This mirrors how SOCCAT (our step 2 pipeline, src/step_2/step_2_cv_pipeline.py)
scores each specific label as an independent NLI entailment problem, and the
report compares against SOCCAT's own performance throughout (see
load_soccat_baseline for exactly which score that is).

Usage:
    python report_step2.py
"""

import csv
import json
import re
import statistics
import sys
from datetime import datetime
from pathlib import Path

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent

sys.path.insert(0, str(REPO_ROOT / "llm" / "classification"))
from step2_taxonomy import LABEL_MAP, TAXONOMY, parse_true_categories  # noqa: E402

OUTPUT_ROOT = REPO_ROOT / "llm" / "classification" / "output" / "step_2"
STEP_2_DATA_DIR = REPO_ROOT / "data" / "model_performance" / "step_2"
GROUND_TRUTH_PATH = REPO_ROOT / "data" / "manual_annotations" / "step_2" / "annotations_ground_truth.csv"

# Display names for broad categories in the report. TAXONOMY keys are left as
# is: they are the labels stored in the runs' predictions.csv (and given to
# the LLM), so renaming them there would break matching with existing runs.
BROAD_DISPLAY_NAMES = {
    "Identities and minority/majority status": "Gender, sexuality, and sociocultural characteristics",
}


def display_broad(broad: str) -> str:
    return BROAD_DISPLAY_NAMES.get(broad, broad)

# broad-class file stem -> its own folder directly under STEP_2_DATA_DIR,
# containing fold_*_per_label.csv (one file per fold, no "fold" column).
# Matches figures/step_2_boxplot.py's FOLDER_NAME_MAP. All 8 broad classes
# are migrated to this layout; the old model_performance/{stem}_per_fold.csv
# files are stale (pre-retrain) and will eventually be removed.
FOLDER_NAME_MAP = {
    "age_family": "age_family_status",
    "identity": "identities_minority_majority_status",
    "labor_market_w_entrepreneurs": "labor_market_position",
    "profession": "profession",
    "real_estate": "real_estate_ownership",
    "social_deviance": "social_deviance",
    "social_roles": "social_roles_behavior",
    "socio_economic": "socio_economic_position",
}

# fold-selection metric for "which fold's model is the deployed one" --
# n_pos_entail-weighted macro F1, matching figures/step_2_boxplot.py and
# tables/step_2_best_fold_table.py (chosen there for robustness to the
# heavy class imbalance in the current test sets; see conversation history).
# The score actually reported/compared against LLM runs stays f1_binary.
BEST_FOLD_METRIC = "f1_macro"

# SOCCAT columns/rows in the LaTeX tables: the released model (each broad
# category's best fold) and the mean / median across its 5 CV folds
SOCCAT_VARIANTS = ["best", "mean", "median"]
SOCCAT_VARIANT_NAMES = {"best": "Best fold", "mean": "Mean", "median": "Median"}
SOCCAT_VARIANT_SHORT = {"best": "best", "mean": "mean", "median": "med"}  # narrow table columns

# Per-label table layout, same as dictionary/05_make_latex_table.py
LABEL_LINE_WRAP_THRESHOLD = 24  # chars; longer labels are split onto 2 balanced lines
LABEL_LINE_GAP = "-3pt"         # tighten \shortstack's default inter-line gap
BROAD_CLASS_LINE_GAP = "3pt"    # gap between a broad-class name and its first label
GROUP_GAP_HEIGHT = "5pt"        # height of the blank spacer row before each broad class
ARRAY_STRETCH = "0.9"           # \arraystretch factor, < 1 tightens row spacing
MANUAL_LINE_BREAKS = {          # labels/headings still too wide on 2 auto-split lines
    "Terrorists, rebels, revolutionaries and/or movements of armed resistance":
        ["Terrorists, rebels,", "revolutionaries and/or", "movements of armed resistance"],
    "Gender, sexuality, and sociocultural characteristics":
        ["Gender, sexuality, and", "sociocultural characteristics"],
}


def label_lines(text: str) -> list:
    """Capitalised, LaTeX-escaped label split onto lines (manual break, or 2
    balanced lines above LABEL_LINE_WRAP_THRESHOLD characters)."""
    text = text[:1].upper() + text[1:]
    if text in MANUAL_LINE_BREAKS:
        return [escape_latex(p) for p in MANUAL_LINE_BREAKS[text]]
    text = escape_latex(text)
    spaces = [i for i, c in enumerate(text) if c == " "]
    if len(text) <= LABEL_LINE_WRAP_THRESHOLD or not spaces:
        return [text]
    split_at = min(spaces, key=lambda i: abs(i - len(text) / 2))
    return [text[:split_at].strip(), text[split_at:].strip()]


def broad_heading_lines(name: str) -> list:
    return [escape_latex(p) for p in MANUAL_LINE_BREAKS.get(name, [name])]
SOCCAT_METRIC_COLS = {"precision": "precision_binary", "recall": "recall_binary",
                      "f1": "f1_binary", "f1_macro": "f1_macro"}

# {broad_display_name: filename stem for {stem}_per_fold.csv}. Matches
# figures/step_2_boxplot.py's BROAD_CLASS_LIST (note "identity" is singular in the
# filename despite our taxonomy's display name). "Others" has no trained SOCCAT
# model -- it's a human-annotation catch-all, never part of the NLI taxonomy.
BROAD_CLASS_FILE_STEM = {
    "Socio-economic position": "socio_economic",
    "Labor market position": "labor_market_w_entrepreneurs",
    "Age and family status": "age_family",
    "Identities and minority/majority status": "identity",
    "Profession": "profession",
    "Social roles and behavior": "social_roles",
    "Social deviance": "social_deviance",
    "Real estate ownership": "real_estate",
}


def find_runs(root: Path) -> list:
    """Return sorted (model, prompt, run_dir) for every complete run under root.
    Silently skips subdirectories that don't look like a finished run (e.g. a
    run that was started but never produced output) rather than erroring."""
    if not root.exists():
        return []
    runs = []
    for run_dir in sorted(root.iterdir()):
        if not run_dir.is_dir() or "__" not in run_dir.name:
            continue
        required = ["predictions.csv", "usage_totals.json", "run_meta.json"]
        if not all((run_dir / f).exists() for f in required):
            continue
        model, _, prompt = run_dir.name.partition("__")
        runs.append((model, prompt, run_dir))
    return runs


def display_model(model: str) -> str:
    """Drop the "claude-" prefix for display -- every model we run is a Claude
    model, so it's redundant noise in report tables/headers."""
    return model[len("claude-"):] if model.startswith("claude-") else model


def format_duration(ms: float) -> str:
    seconds = ms / 1000
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, secs = divmod(seconds, 60)
    return f"{int(minutes)}m {secs:.0f}s"


def _load_broad_class_rows(stem: str) -> list:
    """Per-fold, per-label rows (each dict gets a "fold" key) for one broad
    class, read from its own folder directly under STEP_2_DATA_DIR (one
    fold_*_per_label.csv per fold, no "fold" column in the file itself).
    Mirrors figures/step_2_boxplot.py's load_broad_class_df."""
    folder = FOLDER_NAME_MAP.get(stem, stem)
    fold_files = sorted(
        (STEP_2_DATA_DIR / folder).glob("fold_*_per_label.csv"),
        key=lambda p: int(re.search(r"fold_(\d+)_per_label", p.name).group(1)),
    )
    rows = []
    for f in fold_files:
        fold_num = re.search(r"fold_(\d+)_per_label", f.name).group(1)
        with f.open("r", encoding="utf-8-sig", newline="") as fh:
            for row in csv.DictReader(fh):
                row["fold"] = fold_num
                rows.append(row)
    return rows


def load_soccat_baseline() -> dict:
    """The actual SOCCAT pipeline's performance, for comparison. SOCCAT trains one
    NLI model per broad category (covering every specific label within it) via
    5-fold CV (src/step_2/step_2_cv_pipeline.py); the single best-performing fold
    becomes that category's deployed model, so its specific labels' reported scores
    all come from that one fold -- not a per-label best fold, and not a cross-fold
    average. This mirrors exactly how figures/step_2_boxplot.py and
    tables/step_2_best_fold_table.py select and annotate one "best fold" per broad
    category: n_pos_entail-weighted BEST_FOLD_METRIC (macro F1, for robustness to
    the heavy class imbalance in the current test sets), while the score actually
    reported/compared against LLM runs stays f1_binary.

    Reads fold_*_per_label.csv via _load_broad_class_rows -- raw per-fold,
    per-label data, NOT the pre-averaged *_mean_ci.csv files, which report the
    cross-fold mean rather than the deployed best-fold score. hypothesis_label
    spellings in these files
    sometimes differ from our taxonomy (case/wording drift) -- normalise them
    through the same LABEL_MAP used to build SOCCAT's own training pairs
    (src/step_2/convert_annotations.py), so labels match ours exactly.

    Returns {"broad_f1": {broad: f1}, "broad_weight": {broad: total n_pos at best fold},
    "specific_f1": {(broad, specific): f1} (best fold), "specific_stats": {variant:
    {(broad, specific): {n_pos, precision, recall, f1, f1_macro}}}} with variant in
    SOCCAT_VARIANTS: "best" = the broad category's best fold; "mean"/"median" =
    across its 5 folds, per label (n_pos = mean positive count across folds).
    """
    broad_f1, broad_weight, specific_f1 = {}, {}, {}
    specific_stats = {v: {} for v in SOCCAT_VARIANTS}

    for broad, stem in BROAD_CLASS_FILE_STEM.items():
        rows = _load_broad_class_rows(stem)
        if not rows:
            continue

        by_fold = {}
        for row in rows:
            specific = LABEL_MAP.get(row["hypothesis_label"].strip().lower())
            if specific is None:
                continue  # label dropped from the final taxonomy (e.g. "enterprises")
            by_fold.setdefault(row["fold"], []).append((specific, row))

        def weighted(fold_rows, metric):
            total_w = sum(float(r["n_pos_entail"]) for _, r in fold_rows)
            if not total_w:
                return 0.0, 0.0
            weighted_val = sum(float(r[metric]) * float(r["n_pos_entail"]) for _, r in fold_rows) / total_w
            return weighted_val, total_w

        best_fold, best_sel_score = None, -1.0
        for fold, fold_rows in by_fold.items():
            sel_score, _ = weighted(fold_rows, BEST_FOLD_METRIC)
            if sel_score > best_sel_score:
                best_fold, best_sel_score = fold, sel_score

        if best_fold is None:
            continue
        best_score, best_weight = weighted(by_fold[best_fold], "f1_binary")
        broad_f1[broad] = best_score
        broad_weight[broad] = best_weight
        for specific, row in by_fold[best_fold]:
            specific_f1[(broad, specific)] = float(row["f1_binary"])
            specific_stats["best"][(broad, specific)] = {
                **{k: float(row[col]) for k, col in SOCCAT_METRIC_COLS.items()},
                "n_pos": float(row["n_pos_entail"])}
        per_label = {}
        for fold_rows in by_fold.values():
            for specific, row in fold_rows:
                per_label.setdefault(specific, []).append(row)
        for specific, label_rows in per_label.items():
            for variant, agg in (("mean", statistics.mean), ("median", statistics.median)):
                stats = {k: agg(float(r[col]) for r in label_rows) for k, col in SOCCAT_METRIC_COLS.items()}
                stats["n_pos"] = statistics.mean(float(r["n_pos_entail"]) for r in label_rows)
                specific_stats[variant][(broad, specific)] = stats

    return {"broad_f1": broad_f1, "broad_weight": broad_weight, "specific_f1": specific_f1,
            "specific_stats": specific_stats}


def compute_soccat_overall_f1(baseline: dict) -> float | None:
    """Weighted avg of each broad category's best-CV-fold F1, weighted by
    positive-case count -- SOCCAT's single headline number, comparable to a run's
    metrics["overall"]["f1"]. Note SOCCAT has no comparable overall accuracy/
    precision/recall (load_soccat_baseline only carries F1 per broad category)."""
    broad_f1, broad_weight = baseline["broad_f1"], baseline["broad_weight"]
    if not broad_f1:
        return None
    total_w = sum(broad_weight.values())
    if not total_w:
        return None
    return sum(broad_f1[b] * broad_weight[b] for b in broad_f1) / total_w


def compute_soccat_overall(baseline: dict, variant: str = "best") -> dict | None:
    """SOCCAT's overall precision/recall/F1/macro F1 for one SOCCAT_VARIANTS entry:
    per-label scores weighted by positive-case count -- the same aggregation as a
    run's metrics["overall"] (_weighted_avg). The "best" F1 equals
    compute_soccat_overall_f1."""
    stats = list(baseline.get("specific_stats", {}).get(variant, {}).values())
    total_w = sum(s["n_pos"] for s in stats)
    if not total_w:
        return None
    return {m: sum(s[m] * s["n_pos"] for s in stats) / total_w
            for m in ("precision", "recall", "f1", "f1_macro")}


def format_soccat_summary(baseline: dict) -> str:
    overall_f1 = compute_soccat_overall_f1(baseline)
    if overall_f1 is None:
        return "  (not available)"
    return (
        f"  F1={overall_f1:.3f}  (weighted avg across {len(baseline['broad_f1'])} broad categories' "
        f"best CV fold, weighted by positive-case count)"
    )


def _label_stats(key: tuple, true_sets: list, pred_sets: list, n: int) -> dict:
    y_true = [1 if key in ts else 0 for ts in true_sets]
    y_pred = [1 if key in ps else 0 for ps in pred_sets]
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    # macro F1 = mean of the positive- and negative-class F1 (as SOCCAT's f1_macro)
    f1_neg = 2 * tn / (2 * tn + fp + fn) if (2 * tn + fp + fn) else 0.0
    return {"n_pos": tp + fn, "accuracy": (tp + tn) / n, "precision": precision, "recall": recall,
            "f1": f1, "f1_macro": (f1 + f1_neg) / 2}


def _weighted_avg(stats_list: list):
    """Weighted average of accuracy/precision/recall/F1 across labels, weighted by
    each label's positive-case count (n_pos) -- i.e. sklearn's "weighted" averaging
    (support-weighted), applied across specific labels instead of across sentences."""
    total_w = sum(s["n_pos"] for s in stats_list)
    if total_w == 0:
        return None
    out = {"n_pos": total_w}
    for m in ("accuracy", "precision", "recall", "f1", "f1_macro"):
        out[m] = sum(s[m] * s["n_pos"] for s in stats_list) / total_w
    return out


def load_ground_truth() -> dict:
    """{sentence id: list of (broad, specific) pairs} from the current ground-truth
    file, parsed exactly as classify_step2.py does."""
    with GROUND_TRUTH_PATH.open("r", encoding="utf-8-sig", newline="") as f:
        return {row["id"]: parse_true_categories(row.get("specific_group_new"))
                for row in csv.DictReader(f)}


def apply_ground_truth(rows: list, ground_truth: dict) -> None:
    """Overwrite the true_* columns stored in predictions.csv at classification
    time with labels from the current ground-truth file, so runs are always
    scored against the same annotations as SOCCAT, whenever they were made."""
    for r in rows:
        true_categories = ground_truth[r["id"]]
        r["true_categories"] = json.dumps(true_categories, ensure_ascii=False)
        r["true_has_social_category"] = int(bool(true_categories))


def compute_metrics(rows: list) -> dict:
    """rows: predictions dicts with true_categories/pred_categories JSON columns
    (list of [broad, specific] pairs) and true/pred_has_social_category. Excludes
    ERROR rows."""
    valid = [r for r in rows if r["pred_has_social_category"] != "ERROR"]
    n = len(valid)
    if not n:
        return {}

    true_sets = [{tuple(x) for x in json.loads(r["true_categories"])} for r in valid]
    pred_sets = [{tuple(x) for x in json.loads(r["pred_categories"])} for r in valid]

    det_true = [int(r["true_has_social_category"]) for r in valid]
    det_pred = [int(r["pred_has_social_category"]) for r in valid]
    detection = {
        "n": n,
        "accuracy": accuracy_score(det_true, det_pred),
        "precision": precision_score(det_true, det_pred, zero_division=0),
        "recall": recall_score(det_true, det_pred, zero_division=0),
        "f1": f1_score(det_true, det_pred, zero_division=0),
    }

    label_metrics = {}
    broad_metrics = {}
    for broad, labels in TAXONOMY.items():
        stats_for_broad = []
        for specific in labels:
            stats = _label_stats((broad, specific), true_sets, pred_sets, n)
            label_metrics[(broad, specific)] = stats
            stats_for_broad.append(stats)
        broad_metrics[broad] = _weighted_avg(stats_for_broad)

    overall = _weighted_avg(list(label_metrics.values()))

    return {"n": n, "detection": detection, "label_metrics": label_metrics,
            "broad_metrics": broad_metrics, "overall": overall}


def _format_metric_line(label: str, stats, baseline_f1=None, show_baseline: bool = False) -> str:
    if stats is None:
        line = f"{label:<47} (no positive cases in sample)"
    else:
        line = (
            f"{label:<47} N_pos={stats['n_pos']:>4}  Accuracy={stats['accuracy']:.3f}  "
            f"Precision={stats['precision']:.3f}  Recall={stats['recall']:.3f}  F1={stats['f1']:.3f}"
        )
    if show_baseline:
        baseline_str = f"{baseline_f1:.3f}" if baseline_f1 is not None else "n/a"
        line += f"  |  SOCCAT F1={baseline_str}"
    return line


def build_run_section(model: str, prompt: str, meta: dict, rows: list, usage: dict, baseline: dict) -> list:
    n_sampled = len(rows)
    n_errors = sum(1 for r in rows if r["pred_has_social_category"] == "ERROR")
    metrics = compute_metrics(rows)

    lines = [f"Model: {display_model(model)}   Prompt: {prompt}", "-" * 60]
    lines.append(f"Prompt file:     {meta.get('prompt_path', '?')}")
    lines.append(f"Sentences:       {n_sampled} sampled / {meta.get('n_available', '?')} available (seed={meta.get('seed', '?')})")
    lines.append(f"Batch size:      {meta.get('batch_size', '?')}")
    lines.append(f"Errors:          {n_errors}")
    lines.append("")

    if metrics:
        det = metrics["detection"]
        lines.append(
            f"Detection (has_social_category vs. ground truth): Accuracy={det['accuracy']:.3f}  "
            f"Precision={det['precision']:.3f}  Recall={det['recall']:.3f}  F1={det['f1']:.3f}  (N={det['n']})"
        )
        lines.append("")
        lines.append("Category performance (each specific label scored as its own binary detection problem;")
        lines.append("broad-category and overall scores are the weighted average of specific-label scores,")
        lines.append("weighted by each label's positive-case count)")
        lines.append(_format_metric_line("OVERALL (all specific labels)", metrics["overall"]))
        lines.append("")
        for broad, labels in TAXONOMY.items():
            lines.append(_format_metric_line(display_broad(broad), metrics["broad_metrics"][broad]))
            broad_baseline_f1 = baseline["broad_f1"].get(broad)
            if broad_baseline_f1 is not None:
                lines.append(f"  SOCCAT baseline: F1={broad_baseline_f1:.3f}  (best CV fold, weighted across its labels)")
            for specific in labels:
                baseline_f1 = baseline["specific_f1"].get((broad, specific))
                lines.append(_format_metric_line(
                    f"  {specific}", metrics["label_metrics"][(broad, specific)],
                    baseline_f1, show_baseline=True,
                ))
            lines.append("")
    else:
        lines.append("(no successfully classified rows)")
        lines.append("")

    cost = usage.get("cost_usd", 0.0)
    duration_ms = usage.get("duration_ms", 0)
    lines.append(f"Claude CLI calls:     {usage.get('n_calls', 0)}")
    lines.append(f"Total cost (USD):     ${cost:.4f}")
    if n_sampled:
        cost_per_sentence = cost / n_sampled
        n_available = meta.get("n_available")
        lines.append(f"Cost per sentence:    ${cost_per_sentence:.5f}")
        if n_available:
            lines.append(
                f"Extrapolated cost for full corpus (N={n_available}): "
                f"${cost_per_sentence * n_available:.2f}"
            )
        lines.append(f"Total time:           {format_duration(duration_ms)}")
        rate_per_1000 = duration_ms / n_sampled * 1000
        lines.append(f"Rate:                 {format_duration(rate_per_1000)} per 1,000 sentences")
        if n_available:
            lines.append(
                f"Extrapolated time for full corpus (N={n_available}): "
                f"{format_duration(duration_ms / n_sampled * n_available)}"
            )
    lines.append("")
    return lines, metrics, cost, n_sampled, duration_ms


def build_summary_table(summary_rows: list) -> list:
    """Ranked best-overall-F1-first; runs with no successfully classified rows sort last."""
    def sort_key(row):
        metrics = row[2]
        return metrics["overall"]["f1"] if metrics and metrics["overall"] else -1
    ranked = sorted(summary_rows, key=sort_key, reverse=True)

    header = (
        f"{'Model':<18} {'Prompt':<8} {'N':>5} {'DetectAcc':>9} {'OverallF1':>10} "
        f"{'Cost($)':>9} {'Time':>8} {'Rate/1k':>9}"
    )
    lines = [header, "-" * len(header)]
    for model, prompt, metrics, cost, n_sampled, duration_ms in ranked:
        if metrics:
            det_acc = f"{metrics['detection']['accuracy']:.3f}"
            overall_f1 = f"{metrics['overall']['f1']:.3f}" if metrics["overall"] else "n/a"
        else:
            det_acc, overall_f1 = "n/a", "n/a"
        time_str = format_duration(duration_ms)
        rate_str = format_duration(duration_ms / n_sampled * 1000) if n_sampled else "n/a"
        lines.append(
            f"{display_model(model):<18} {prompt:<8} {n_sampled:>5} {det_acc:>9} {overall_f1:>10} "
            f"{cost:>9.4f} {time_str:>8} {rate_str:>9}"
        )
    return lines


def build_broad_comparison_table(summary_rows: list, baseline: dict) -> list:
    """Rows = broad categories, columns = SOCCAT baseline then each run's F1. For the
    model columns each cell is metrics["broad_metrics"][broad]["f1"] -- the weighted
    average of that broad category's specific-label F1s, weighted by positive-case
    count. The SOCCAT column is that same broad category's actual deployed-model
    score (its best CV fold, itself already weighted the same way -- see
    load_soccat_baseline)."""
    ranked = sorted(
        summary_rows,
        key=lambda r: r[2]["overall"]["f1"] if r[2] and r[2]["overall"] else -1,
        reverse=True,
    )
    run_labels = [f"{display_model(model)}/{prompt}" for model, prompt, *_ in ranked]

    cat_col_width = max(len("Broad category"), max((len(display_broad(b)) for b in TAXONOMY), default=0))
    col_widths = [max(len(rl), 9) for rl in run_labels]

    header = [f"{'Broad category':<{cat_col_width}}", f"{'SOCCAT':>9}"]
    header += [f"{rl:>{w}}" for rl, w in zip(run_labels, col_widths)]
    header_line = " ".join(header)
    lines = [header_line, "-" * len(header_line)]

    for broad in TAXONOMY:
        baseline_f1 = baseline["broad_f1"].get(broad)
        baseline_str = f"{baseline_f1:.3f}" if baseline_f1 is not None else "n/a"

        row = [f"{display_broad(broad):<{cat_col_width}}", f"{baseline_str:>9}"]
        for (model, prompt, metrics, cost, n_sampled, duration_ms), w in zip(ranked, col_widths):
            bm = metrics["broad_metrics"][broad] if metrics else None
            row.append(f"{bm['f1']:.3f}".rjust(w) if bm else "n/a".rjust(w))
        lines.append(" ".join(row))
    return lines


def build_category_comparison_table(summary_rows: list, baseline: dict) -> list:
    """Rows = every taxonomy category (broad section headers + their specific labels),
    columns = N_pos_llm (positive-case count in the LLM sample, max across runs --
    they should all match if every run used the same seed/sample, but we don't
    assume it), then SOCCAT baseline F1, then each run's F1, ranked the same as the
    summary table (best overall F1 first) so column order matches it.

    Note: SOCCAT's F1 is computed on its own CV held-out test fold, not on this
    LLM sample -- the two columns are not scored on identical sentences, just
    independent samples of the same annotated corpus (see discussion in report
    history / conversation)."""
    ranked = sorted(
        summary_rows,
        key=lambda r: r[2]["overall"]["f1"] if r[2] and r[2]["overall"] else -1,
        reverse=True,
    )
    run_labels = [f"{display_model(model)}/{prompt}" for model, prompt, *_ in ranked]

    cat_col_width = max(
        len("Category"),
        max((len(f"  {s}") for labels in TAXONOMY.values() for s in labels), default=0),
        max((len(display_broad(b)) for b in TAXONOMY), default=0),
    )
    col_widths = [max(len(rl), 9) for rl in run_labels]

    header = [f"{'Category':<{cat_col_width}}", f"{'N_pos_llm':>9}", f"{'SOCCAT':>9}"]
    header += [f"{rl:>{w}}" for rl, w in zip(run_labels, col_widths)]
    header_line = " ".join(header)
    lines = [header_line, "-" * len(header_line)]

    for broad, labels in TAXONOMY.items():
        lines.append(f"{display_broad(broad):<{cat_col_width}}")
        for specific in labels:
            n_pos_values = [
                metrics["label_metrics"][(broad, specific)]["n_pos"]
                for _, _, metrics, *_ in ranked
                if metrics and (broad, specific) in metrics["label_metrics"]
            ]
            n_pos_str = str(max(n_pos_values)) if n_pos_values else "n/a"

            baseline_f1 = baseline["specific_f1"].get((broad, specific))
            baseline_str = f"{baseline_f1:.3f}" if baseline_f1 is not None else "n/a"
            row = [f"{'  ' + specific:<{cat_col_width}}", f"{n_pos_str:>9}", f"{baseline_str:>9}"]
            for (model, prompt, metrics, cost, n_sampled, duration_ms), w in zip(ranked, col_widths):
                if metrics and (broad, specific) in metrics["label_metrics"]:
                    f1 = metrics["label_metrics"][(broad, specific)]["f1"]
                    row.append(f"{f1:.3f}".rjust(w))
                else:
                    row.append("n/a".rjust(w))
            lines.append(" ".join(row))
    return lines


def build_category_outperform_table(summary_rows: list, baseline: dict) -> list:
    """Same shape as build_category_comparison_table, but restricted to specific
    labels where at least one run's F1 beats SOCCAT's F1 for that label (marked
    with a trailing "*" on the qualifying cell). Labels with no SOCCAT baseline
    (e.g. "Others") are excluded -- there's no baseline to beat."""
    ranked = sorted(
        summary_rows,
        key=lambda r: r[2]["overall"]["f1"] if r[2] and r[2]["overall"] else -1,
        reverse=True,
    )
    run_labels = [f"{display_model(model)}/{prompt}" for model, prompt, *_ in ranked]

    qualifying = {}
    for broad, labels in TAXONOMY.items():
        keep = []
        for specific in labels:
            baseline_f1 = baseline["specific_f1"].get((broad, specific))
            if baseline_f1 is None:
                continue
            run_f1s = [
                metrics["label_metrics"][(broad, specific)]["f1"]
                for _, _, metrics, *_ in ranked
                if metrics and (broad, specific) in metrics["label_metrics"]
            ]
            if any(f1 > baseline_f1 for f1 in run_f1s):
                keep.append(specific)
        if keep:
            qualifying[broad] = keep

    if not qualifying:
        return ["(no category where an LLM run beat the SOCCAT baseline)"]

    cat_col_width = max(
        len("Category"),
        max((len(f"  {s}") for labels in qualifying.values() for s in labels), default=0),
        max((len(display_broad(b)) for b in qualifying), default=0),
    )
    col_widths = [max(len(rl), 9) for rl in run_labels]

    header = [f"{'Category':<{cat_col_width}}", f"{'N_pos_llm':>9}", f"{'SOCCAT':>9}"]
    header += [f"{rl:>{w}}" for rl, w in zip(run_labels, col_widths)]
    header_line = " ".join(header)
    lines = [header_line, "-" * len(header_line)]

    for broad, labels in qualifying.items():
        lines.append(f"{display_broad(broad):<{cat_col_width}}")
        for specific in labels:
            n_pos_values = [
                metrics["label_metrics"][(broad, specific)]["n_pos"]
                for _, _, metrics, *_ in ranked
                if metrics and (broad, specific) in metrics["label_metrics"]
            ]
            n_pos_str = str(max(n_pos_values)) if n_pos_values else "n/a"

            baseline_f1 = baseline["specific_f1"][(broad, specific)]
            row = [f"{'  ' + specific:<{cat_col_width}}", f"{n_pos_str:>9}", f"{baseline_f1:.3f}".rjust(9)]
            for (model, prompt, metrics, cost, n_sampled, duration_ms), w in zip(ranked, col_widths):
                if metrics and (broad, specific) in metrics["label_metrics"]:
                    f1 = metrics["label_metrics"][(broad, specific)]["f1"]
                    cell = f"{f1:.3f}" + ("*" if f1 > baseline_f1 else "")
                    row.append(cell.rjust(w))
                else:
                    row.append("n/a".rjust(w))
            lines.append(" ".join(row))
    return lines


def escape_latex(text: str) -> str:
    for ch, repl in (("&", r"\&"), ("%", r"\%"), ("_", r"\_"), ("#", r"\#"), ("$", r"\$")):
        text = text.replace(ch, repl)
    return text


def parse_model_thinking(model: str) -> tuple:
    """"claude-sonnet-5-high" -> ("Sonnet-5", "High"). Falls back to
    (display_model(model), "--") if the trailing "-high"/"-low" isn't there."""
    base = display_model(model)
    for suffix, thinking in ((("-high"), "High"), (("-low"), "Low")):
        if base.endswith(suffix):
            name = base[: -len(suffix)]
            return name[:1].upper() + name[1:], thinking
    return base[:1].upper() + base[1:], "--"


# The specific-label table (build_latex_category_tables) is too tall for one
# page, so it's split across two files at this taxonomy boundary.
BROAD_SPLIT = [
    ["Socio-economic position", "Labor market position", "Age and family status",
     "Identities and minority/majority status"],
    ["Profession", "Social roles and behavior", "Social deviance",
     "Real estate ownership", "Others"],
]


def build_latex_category_tables(summary_rows: list, baseline: dict) -> list:
    """Two LaTeX tables (one per BROAD_SPLIT half): rows = every specific taxonomy
    label grouped under its broad category, columns = SOCCAT then each LLM run's F1
    (a 3-line column header: model / thinking effort / prompt length). A specific
    label's row and any run F1 that beats SOCCAT's mean F1 for that label are set in italics (bold is used for the broad-class names) --
    labels with no SOCCAT baseline (e.g. "Others") never qualify. Column order is
    grouped High-effort-then-Low, short/medium/long within each, not F1-ranked --
    this is a reference table, not a leaderboard.
    Returns [part_1_lines, part_2_lines]."""

    def run_sort_key(row):
        model, prompt = row[0], row[1]
        _, thinking = parse_model_thinking(model)
        thinking_rank = {"High": 0, "Low": 1}.get(thinking, 2)
        prompt_rank = {"short": 0, "medium": 1, "long": 2}.get(prompt, 3)
        return (thinking_rank, prompt_rank)

    ordered_runs = sorted(summary_rows, key=run_sort_key)
    n_runs = len(ordered_runs)
    n_soccat = len(SOCCAT_VARIANTS)

    parts = []
    for part_idx, broads in enumerate(BROAD_SPLIT, start=1):
        lines = [
            r"\begin{table}[htb]",
            r"\centering",
            rf"\renewcommand{{\arraystretch}}{{{ARRAY_STRETCH}}}",
            r"\scriptsize",
            r"\begin{threeparttable}",
            r"\begin{tabular}{l" + " c" * (n_runs + n_soccat) + "}",
            r"\toprule",
        ]
        row1 = ["Model", f"\\multicolumn{{{n_runs}}}{{c}}{{Sonnet-5}}",
                f"\\multicolumn{{{n_soccat}}}{{c}}{{SOCCAT}}"]
        row2 = ["Thinking"] + [parse_model_thinking(model)[1] for model, prompt, *_ in ordered_runs] + [""] * n_soccat
        row3 = (["Prompt"] + [prompt.capitalize() for model, prompt, *_ in ordered_runs]
                + [SOCCAT_VARIANT_SHORT[v] for v in SOCCAT_VARIANTS])
        lines.append(" & ".join(row1) + r" \\")
        lines.append(f"\\cmidrule(lr){{2-{1 + n_runs}}} "
                     f"\\cmidrule(lr){{{2 + n_runs}-{1 + n_runs + n_soccat}}}")
        lines.append(" & ".join(row2) + r" \\")
        lines.append(" & ".join(row3) + r" \\")
        lines.append(r"\midrule")

        # same layout as dictionary/05_make_latex_table.py: a thin spacer row
        # before each broad class, whose name is stacked into its first label's cell
        for broad in broads:
            lines.append(" & ".join([rf"\rule{{0pt}}{{{GROUP_GAP_HEIGHT}}}"] + [""] * (n_runs + n_soccat)) + r" \\")
            for j, specific in enumerate(TAXONOMY[broad]):
                soccat_f1s = [baseline["specific_stats"][v].get((broad, specific), {}).get("f1")
                              for v in SOCCAT_VARIANTS]
                baseline_f1 = soccat_f1s[SOCCAT_VARIANTS.index("mean")]  # reference for bold
                run_f1s = []
                for model, prompt, metrics, cost, n_sampled, duration_ms in ordered_runs:
                    stats = metrics["label_metrics"].get((broad, specific)) if metrics else None
                    run_f1s.append(stats["f1"] if stats else None)

                beats = baseline_f1 is not None and any(
                    f1 is not None and f1 > baseline_f1 for f1 in run_f1s
                )
                label = [rf"\textit{{{line}}}" if beats else line for line in label_lines(specific)]
                line_sep = "\\\\[" + LABEL_LINE_GAP + "]"
                if j == 0:
                    heading = [rf"\textbf{{{line}}}" for line in broad_heading_lines(display_broad(broad))]
                    stacked = line_sep.join(heading) + "\\\\[" + BROAD_CLASS_LINE_GAP + "]" + line_sep.join(label)
                    cell = r"\shortstack[l]{" + stacked + "}"
                elif len(label) > 1:
                    cell = r"\shortstack[l]{" + line_sep.join(label) + "}"
                else:
                    cell = label[0]
                cells = [cell]
                for f1 in run_f1s:
                    if f1 is None:
                        cells.append("--")
                    elif beats and f1 > baseline_f1:
                        cells.append(r"\textit{" + f"{f1:.2f}" + "}")
                    else:
                        cells.append(f"{f1:.2f}")
                cells += [f"{f1:.2f}" if f1 is not None else "--" for f1 in soccat_f1s]
                lines.append(" & ".join(cells) + r" \\")

        lines += [
            r"\bottomrule",
            r"\end{tabular}",
            r"\begin{tablenotes}[flushleft]",
            r"\footnotesize",
            r"\item \textit{Note:} Specific group labels are listed under the name of their broad class. "
            r"Scores are positive-class F1. SOCCAT: best cross-validation fold (best, released model), or "
            r"mean / median (med) across the 5 folds. Italics are used for a label when at least one LLM "
            r"run beats SOCCAT's mean F1, and it shows the beating F1 values.",
            r"\end{tablenotes}",
            r"\end{threeparttable}",
            f"\\caption{{Performance comparison with LLMs for Step 2, detailed F1 score by specific "
            f"group labels. Part {part_idx}/2.}}",
            f"\\label{{tab:llm-step2-{part_idx}}}",
            r"\end{table}",
        ]
        parts.append(lines)
    return parts


def build_latex_summary_table(summary_rows: list, baseline: dict) -> list:
    """Same shape as report_step1's build_latex_table: one row per (model, prompt)
    run plus a pinned SOCCAT row, Model/Thinking/Prompt as separate columns, 2
    decimals, cost/rate normalised per 1k sentences. Uses metrics["overall"]
    (each specific label scored as its own binary problem, weighted-averaged by
    positive-case count) rather than metrics["detection"] -- this is the step 2
    analogue of step 1's single binary-classification F1, and it's what SOCCAT's
    own scores (compute_soccat_overall, same aggregation) are comparable to."""
    ranked = sorted(
        summary_rows,
        key=lambda r: r[2]["overall"]["f1"] if r[2] and r[2]["overall"] else -1,
        reverse=True,
    )

    lines = [
        r"\begin{table}[htb]",
        r"\centering",
        r"\begin{threeparttable}",
        r"\begin{tabular}{lllrrrrrr}",
        r"\toprule",
        r"Model & Thinking & Prompt & Precision & Recall & F1 & Macro F1 & Cost/1k (\$) & Rate/1k \\",
        r"\midrule",
    ]
    for model, prompt, metrics, cost, n_sampled, duration_ms in ranked:
        model_name, thinking = parse_model_thinking(model)
        overall = metrics["overall"] if metrics else None
        f1m = f"{overall['f1_macro']:.2f}" if overall else "--"
        prec = f"{overall['precision']:.2f}" if overall else "--"
        rec = f"{overall['recall']:.2f}" if overall else "--"
        f1 = f"{overall['f1']:.2f}" if overall else "--"
        cost_per_1k = cost / n_sampled * 1000 if n_sampled else 0.0
        rate_str = format_duration(duration_ms / n_sampled * 1000) if n_sampled else "--"
        lines.append(
            f"{model_name} & {thinking} & {prompt.capitalize()} & {prec} & {rec} & {f1} & {f1m} & "
            f"{cost_per_1k:.2f} & {rate_str} \\\\"
        )

    lines.append(r"\midrule")
    for variant in SOCCAT_VARIANTS:
        soccat = compute_soccat_overall(baseline, variant)
        # variant spans the Thinking/Prompt columns (a longer first column overflows the page)
        name = f"SOCCAT & \\multicolumn{{2}}{{l}}{{{SOCCAT_VARIANT_NAMES[variant]}}}"
        if soccat is not None:
            lines.append(
                f"{name} & {soccat['precision']:.2f} & {soccat['recall']:.2f} & "
                f"{soccat['f1']:.2f} & {soccat['f1_macro']:.2f} & -- & -- \\\\"
            )
        else:
            lines.append(f"{name} & -- & -- & -- & -- & -- & -- \\\\")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\begin{tablenotes}[flushleft]",
        r"\footnotesize",
        r"\item \textit{Note:} Precision, recall and F1 refer to the positive class of each specific "
        r"category; macro F1 averages the F1 scores of the positive and negative class. The scores for "
        r"specific categories are weighted by positive-case count to obtain these average performances. "
        r"SOCCAT: best cross-validation fold (released model), or mean / median across the 5 folds.",
        r"\end{tablenotes}",
        r"\end{threeparttable}",
        r"\caption{Overall performance comparison with LLMs for Step 2.}",
        r"\label{tab:llm-step2-summary}",
        r"\end{table}",
    ])
    return lines


def main():
    runs = find_runs(OUTPUT_ROOT)
    baseline = load_soccat_baseline()

    lines = []
    lines.append("Step 2 LLM classification report -- all runs")
    lines.append("=" * 60)
    lines.append(f"Generated:  {datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"Runs found: {len(runs)}  (in {OUTPUT_ROOT.relative_to(REPO_ROOT)})")
    lines.append("")
    lines.append(
        f"SOCCAT baseline for reference "
        f"({STEP_2_DATA_DIR.relative_to(REPO_ROOT)}/*/fold_*_per_label.csv, "
        f"best fold selected by n_pos-weighted macro F1):"
    )
    lines.append(format_soccat_summary(baseline))
    lines.append("")

    report_dir = SCRIPT_DIR / "output"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "step_2_report.txt"

    if not runs:
        lines.append("No completed runs found. Run classify_step2.py first.")
        report_path.write_text("\n".join(lines), encoding="utf-8")
        print(f"Report: {report_path}")
        print(f"No runs found under {OUTPUT_ROOT}")
        return

    summary_rows = []
    detail_lines = []
    ground_truth = load_ground_truth()
    for model, prompt, run_dir in runs:
        rows = list(csv.DictReader((run_dir / "predictions.csv").open("r", encoding="utf-8", newline="")))
        apply_ground_truth(rows, ground_truth)
        usage = json.loads((run_dir / "usage_totals.json").read_text(encoding="utf-8"))
        meta = json.loads((run_dir / "run_meta.json").read_text(encoding="utf-8"))

        section, metrics, cost, n_sampled, duration_ms = build_run_section(model, prompt, meta, rows, usage, baseline)
        detail_lines.extend(section)
        summary_rows.append((model, prompt, metrics, cost, n_sampled, duration_ms))

    lines.append("Summary (all runs)")
    lines.append("=" * 60)
    lines.extend(build_summary_table(summary_rows))
    lines.append("")

    lines.append("Broad category F1 by model (weighted avg of specific-label F1s, weighted by positive-case count)")
    lines.append("=" * 60)
    lines.extend(build_broad_comparison_table(summary_rows, baseline))
    lines.append("")

    lines.append("Category F1 by model (rows = taxonomy categories, columns = SOCCAT baseline then each run)")
    lines.append("=" * 60)
    lines.extend(build_category_comparison_table(summary_rows, baseline))
    lines.append("")

    lines.append("Categories where an LLM run beats SOCCAT (* marks the beating cell)")
    lines.append("=" * 60)
    lines.extend(build_category_outperform_table(summary_rows, baseline))
    lines.append("")

    lines.append("Run details")
    lines.append("=" * 60)
    lines.extend(detail_lines)

    report_path.write_text("\n".join(lines), encoding="utf-8")

    summary_table_path = report_dir / "step_2_summary_table.tex"
    summary_table_path.write_text(
        "\n".join(build_latex_summary_table(summary_rows, baseline)) + "\n", encoding="utf-8"
    )

    table_parts = build_latex_category_tables(summary_rows, baseline)
    table_paths = [summary_table_path]
    for part_idx, part_lines in enumerate(table_parts, start=1):
        table_path = report_dir / f"step_2_table_part{part_idx}.tex"
        table_path.write_text("\n".join(part_lines) + "\n", encoding="utf-8")
        table_paths.append(table_path)

    print(f"Report: {report_path}")
    for table_path in table_paths:
        print(f"LaTeX table: {table_path}")
    print(f"{len(runs)} run(s) included")


if __name__ == "__main__":
    main()
