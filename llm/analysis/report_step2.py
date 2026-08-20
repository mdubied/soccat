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
import sys
from datetime import datetime
from pathlib import Path

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent

sys.path.insert(0, str(REPO_ROOT / "llm" / "classification"))
from step2_taxonomy import LABEL_MAP, TAXONOMY  # noqa: E402

OUTPUT_ROOT = REPO_ROOT / "llm" / "classification" / "output" / "step_2"
PER_FOLD_DIR = REPO_ROOT / "data" / "model_performance" / "step_2" / "model_performance"

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


def load_soccat_baseline() -> dict:
    """The actual SOCCAT pipeline's performance, for comparison. SOCCAT trains one
    NLI model per broad category (covering every specific label within it) via
    5-fold CV (src/step_2/step_2_cv_pipeline.py); the single best-performing fold
    becomes that category's deployed model, so its specific labels' reported scores
    all come from that one fold -- not a per-label best fold, and not a cross-fold
    average. This mirrors exactly how figures/step_2_boxplot.py selects and
    annotates one "best fold" per broad category (weighted by n_pos_entail, i.e.
    positive-case count, matching --sel_metric f1_binary's own default).

    Reads {broad}_per_fold.csv (raw per-fold, per-label data -- NOT the pre-averaged
    *_mean_ci.csv files, which report the cross-fold mean rather than the deployed
    best-fold score). hypothesis_label spellings in these files sometimes differ
    from our taxonomy (older run, case/wording drift) -- normalise them through the
    same LABEL_MAP used to build SOCCAT's own training pairs
    (src/step_2/convert_annotations.py), so labels match ours exactly.

    Returns {"broad_f1": {broad: f1}, "broad_weight": {broad: total n_pos at best fold},
    "specific_f1": {(broad, specific): f1}}.
    """
    broad_f1, broad_weight, specific_f1 = {}, {}, {}

    for broad, stem in BROAD_CLASS_FILE_STEM.items():
        path = PER_FOLD_DIR / f"{stem}_per_fold.csv"
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))

        by_fold = {}
        for row in rows:
            specific = LABEL_MAP.get(row["hypothesis_label"].strip().lower())
            if specific is None:
                continue  # label dropped from the final taxonomy (e.g. "enterprises")
            by_fold.setdefault(row["fold"], []).append((specific, row))

        def weighted_f1(fold_rows):
            total_w = sum(float(r["n_pos_entail"]) for _, r in fold_rows)
            if not total_w:
                return 0.0, 0.0
            weighted = sum(float(r["f1_binary"]) * float(r["n_pos_entail"]) for _, r in fold_rows) / total_w
            return weighted, total_w

        best_fold, best_score, best_weight = None, -1.0, 0.0
        for fold, fold_rows in by_fold.items():
            score, weight = weighted_f1(fold_rows)
            if score > best_score:
                best_fold, best_score, best_weight = fold, score, weight

        if best_fold is None:
            continue
        broad_f1[broad] = best_score
        broad_weight[broad] = best_weight
        for specific, row in by_fold[best_fold]:
            specific_f1[(broad, specific)] = float(row["f1_binary"])

    return {"broad_f1": broad_f1, "broad_weight": broad_weight, "specific_f1": specific_f1}


def format_soccat_summary(baseline: dict) -> str:
    broad_f1, broad_weight = baseline["broad_f1"], baseline["broad_weight"]
    if not broad_f1:
        return "  (not available)"
    total_w = sum(broad_weight.values())
    if not total_w:
        return "  (not available)"
    overall_f1 = sum(broad_f1[b] * broad_weight[b] for b in broad_f1) / total_w
    return (
        f"  F1={overall_f1:.3f}  (weighted avg across {len(broad_f1)} broad categories' best CV fold, "
        f"weighted by positive-case count)"
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
            lines.append(_format_metric_line(broad, metrics["broad_metrics"][broad]))
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

    cat_col_width = max(len("Broad category"), max((len(b) for b in TAXONOMY), default=0))
    col_widths = [max(len(rl), 9) for rl in run_labels]

    header = [f"{'Broad category':<{cat_col_width}}", f"{'SOCCAT':>9}"]
    header += [f"{rl:>{w}}" for rl, w in zip(run_labels, col_widths)]
    header_line = " ".join(header)
    lines = [header_line, "-" * len(header_line)]

    for broad in TAXONOMY:
        baseline_f1 = baseline["broad_f1"].get(broad)
        baseline_str = f"{baseline_f1:.3f}" if baseline_f1 is not None else "n/a"

        row = [f"{broad:<{cat_col_width}}", f"{baseline_str:>9}"]
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
        max((len(b) for b in TAXONOMY), default=0),
    )
    col_widths = [max(len(rl), 9) for rl in run_labels]

    header = [f"{'Category':<{cat_col_width}}", f"{'N_pos_llm':>9}", f"{'SOCCAT':>9}"]
    header += [f"{rl:>{w}}" for rl, w in zip(run_labels, col_widths)]
    header_line = " ".join(header)
    lines = [header_line, "-" * len(header_line)]

    for broad, labels in TAXONOMY.items():
        lines.append(f"{broad:<{cat_col_width}}")
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
        max((len(b) for b in qualifying), default=0),
    )
    col_widths = [max(len(rl), 9) for rl in run_labels]

    header = [f"{'Category':<{cat_col_width}}", f"{'N_pos_llm':>9}", f"{'SOCCAT':>9}"]
    header += [f"{rl:>{w}}" for rl, w in zip(run_labels, col_widths)]
    header_line = " ".join(header)
    lines = [header_line, "-" * len(header_line)]

    for broad, labels in qualifying.items():
        lines.append(f"{broad:<{cat_col_width}}")
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


def main():
    runs = find_runs(OUTPUT_ROOT)
    baseline = load_soccat_baseline()

    lines = []
    lines.append("Step 2 LLM classification report -- all runs")
    lines.append("=" * 60)
    lines.append(f"Generated:  {datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"Runs found: {len(runs)}  (in {OUTPUT_ROOT.relative_to(REPO_ROOT)})")
    lines.append("")
    lines.append(f"SOCCAT baseline for reference ({PER_FOLD_DIR.relative_to(REPO_ROOT)}/*_per_fold.csv):")
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
    for model, prompt, run_dir in runs:
        rows = list(csv.DictReader((run_dir / "predictions.csv").open("r", encoding="utf-8", newline="")))
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
    print(f"Report: {report_path}")
    print(f"{len(runs)} run(s) included")


if __name__ == "__main__":
    main()
