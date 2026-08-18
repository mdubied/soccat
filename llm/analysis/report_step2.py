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
This mirrors how mDeBERTa's own step 2 pipeline scores each specific label
as an independent NLI entailment problem (src/step_2/step_2_cv_pipeline.py).

Usage:
    python report_step2.py
"""

import csv
import json
import sys
from datetime import datetime
from pathlib import Path

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent

sys.path.insert(0, str(REPO_ROOT / "llm" / "classification"))
from step2_taxonomy import TAXONOMY  # noqa: E402

OUTPUT_ROOT = REPO_ROOT / "llm" / "classification" / "output" / "step_2"
BASELINE_PATH = REPO_ROOT / "data" / "model_performance" / "step_2" / "model_performance" / "broad_cat_mean_ci.csv"

# broad_cat_mean_ci.csv's "model" column (short internal names) -> our taxonomy's
# display names. "business_activity" isn't in our taxonomy (dropped from the final
# category list); "Others" has no mDeBERTa baseline since it was never trained on it.
BASELINE_BROAD_NAME_MAP = {
    "socio_economic": "Socio-economic position",
    "labor_market_w_entrepreneurs": "Labor market position",
    "age_family": "Age and family status",
    "identities": "Identities and minority/majority status",
    "profession": "Profession",
    "social_roles": "Social roles and behavior",
    "social_deviance": "Social deviance",
    "real_estate": "Real estate ownership",
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


def load_baseline_summary() -> str:
    if not BASELINE_PATH.exists():
        return "  (not available)"
    with BASELINE_PATH.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return "  (empty baseline file)"

    # broad_cat_mean_ci.csv has one row per broad category (ALL_LABELS), no single
    # overall row -- report an unweighted mean across categories.
    accs = [float(r["accuracy_mean"]) for r in rows if r.get("accuracy_mean")]
    f1s = [float(r["f1_macro_mean"]) for r in rows if r.get("f1_macro_mean")]
    if not accs:
        return "  (unrecognized baseline format)"
    return (
        f"  Accuracy={sum(accs) / len(accs):.3f}  F1(macro)={sum(f1s) / len(f1s):.3f}  "
        f"(unweighted mean across {len(rows)} broad categories, mDeBERTa cross-validation)"
    )


def load_broad_baseline() -> dict:
    """Per-broad-category mDeBERTa CV baseline, for display next to each broad
    category's LLM score. Keyed by our taxonomy's display names."""
    if not BASELINE_PATH.exists():
        return {}
    with BASELINE_PATH.open("r", encoding="utf-8-sig", newline="") as f:
        rows = {row["model"]: row for row in csv.DictReader(f)}
    out = {}
    for raw_name, display in BASELINE_BROAD_NAME_MAP.items():
        row = rows.get(raw_name)
        if row:
            out[display] = (
                f"Accuracy={float(row['accuracy_mean']):.3f}  "
                f"F1(macro)={float(row['f1_macro_mean']):.3f}  "
                f"(mDeBERTa, {row['folds_count']}-fold CV mean)"
            )
    return out


def format_duration(ms: float) -> str:
    seconds = ms / 1000
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, secs = divmod(seconds, 60)
    return f"{int(minutes)}m {secs:.0f}s"


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
    return {"n_pos": tp + fn, "accuracy": (tp + tn) / n, "precision": precision, "recall": recall, "f1": f1}


def _weighted_avg(stats_list: list):
    """Weighted average of accuracy/precision/recall/F1 across labels, weighted by
    each label's positive-case count (n_pos) -- i.e. sklearn's "weighted" averaging
    (support-weighted), applied across specific labels instead of across sentences."""
    total_w = sum(s["n_pos"] for s in stats_list)
    if total_w == 0:
        return None
    out = {"n_pos": total_w}
    for m in ("accuracy", "precision", "recall", "f1"):
        out[m] = sum(s[m] * s["n_pos"] for s in stats_list) / total_w
    return out


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


def _format_metric_line(label: str, stats) -> str:
    if stats is None:
        return f"{label:<47} (no positive cases in sample)"
    return (
        f"{label:<47} N_pos={stats['n_pos']:>4}  Accuracy={stats['accuracy']:.3f}  "
        f"Precision={stats['precision']:.3f}  Recall={stats['recall']:.3f}  F1={stats['f1']:.3f}"
    )


def build_run_section(model: str, prompt: str, meta: dict, rows: list, usage: dict, baseline_per_broad: dict) -> list:
    n_sampled = len(rows)
    n_errors = sum(1 for r in rows if r["pred_has_social_category"] == "ERROR")
    metrics = compute_metrics(rows)

    lines = [f"Model: {model}   Prompt: {prompt}", "-" * 60]
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
            lines.append(_format_metric_line(broad, metrics["broad_metrics"][broad]))
            if broad in baseline_per_broad:
                lines.append(f"  mDeBERTa baseline: {baseline_per_broad[broad]}")
            for specific in labels:
                lines.append(_format_metric_line(f"  {specific}", metrics["label_metrics"][(broad, specific)]))
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
        f"{'Model':<22} {'Prompt':<8} {'N':>5} {'DetectAcc':>9} {'OverallF1':>10} "
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
            f"{model:<22} {prompt:<8} {n_sampled:>5} {det_acc:>9} {overall_f1:>10} "
            f"{cost:>9.4f} {time_str:>8} {rate_str:>9}"
        )
    return lines


def main():
    runs = find_runs(OUTPUT_ROOT)

    lines = []
    lines.append("Step 2 LLM classification report -- all runs")
    lines.append("=" * 60)
    lines.append(f"Generated:  {datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"Runs found: {len(runs)}  (in {OUTPUT_ROOT.relative_to(REPO_ROOT)})")
    lines.append("")
    lines.append(f"mDeBERTa baseline for reference ({BASELINE_PATH.relative_to(REPO_ROOT)}):")
    lines.append(load_baseline_summary())
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

    baseline_per_broad = load_broad_baseline()

    summary_rows = []
    detail_lines = []
    for model, prompt, run_dir in runs:
        rows = list(csv.DictReader((run_dir / "predictions.csv").open("r", encoding="utf-8", newline="")))
        usage = json.loads((run_dir / "usage_totals.json").read_text(encoding="utf-8"))
        meta = json.loads((run_dir / "run_meta.json").read_text(encoding="utf-8"))

        section, metrics, cost, n_sampled, duration_ms = build_run_section(model, prompt, meta, rows, usage, baseline_per_broad)
        detail_lines.extend(section)
        summary_rows.append((model, prompt, metrics, cost, n_sampled, duration_ms))

    lines.append("Summary (all runs)")
    lines.append("=" * 60)
    lines.extend(build_summary_table(summary_rows))
    lines.append("")

    lines.append("Run details")
    lines.append("=" * 60)
    lines.extend(detail_lines)

    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Report: {report_path}")
    print(f"{len(runs)} run(s) included")


if __name__ == "__main__":
    main()
