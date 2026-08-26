"""
Compute performance metrics and cost estimates across all Step 1 LLM
classification runs, and write a single combined txt report.

Scans llm/classification/output/step_1/ for every {model}__{prompt} run
directory produced by classify_step1.py, and reports on whichever ones it
finds -- it is not an error for some model/prompt combinations to be
missing (e.g. you haven't run --model claude-opus-5 yet). Kept separate
from classify_step1.py so the report can be regenerated (or runs compared)
without spending any more Claude credits.

Step 1 is a single binary classification problem (accuracy/precision/
recall/F1 vs. has_group).

Usage:
    python report_step1.py
"""

import csv
import json
from datetime import datetime
from pathlib import Path

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent

OUTPUT_ROOT = REPO_ROOT / "llm" / "classification" / "output" / "step_1"
BASELINE_PATH = REPO_ROOT / "data" / "model_performance" / "step_1" / "performance_all_levels.csv"


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


def load_baseline_metrics() -> dict | None:
    """Return the SOCCAT mDeBERTa baseline's overall row from
    performance_all_levels.csv, or None if unavailable. Note the "F1" column
    there is weighted F1 (see report_step1.py's own f1_weighted computation
    above) -- the baseline pipeline doesn't report a macro F1."""
    if not BASELINE_PATH.exists():
        return None
    with BASELINE_PATH.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("group") == "ALL":
                return {
                    "accuracy": float(row["Accuracy"]),
                    "precision": float(row["Precision"]),
                    "recall": float(row["Recall"]),
                    "f1_weighted": float(row["F1"]),
                    "n": row["N"],
                }
    return None


def format_baseline_summary(baseline: dict | None) -> str:
    if baseline is None:
        return "  (not available)"
    return (
        f"  Accuracy={baseline['accuracy']:.3f}  Precision={baseline['precision']:.3f}  "
        f"Recall={baseline['recall']:.3f}  F1={baseline['f1_weighted']:.3f}  (N={baseline['n']})"
    )


def format_duration(ms: float) -> str:
    seconds = ms / 1000
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, secs = divmod(seconds, 60)
    return f"{int(minutes)}m {secs:.0f}s"


def compute_metrics(rows: list) -> dict:
    """rows: predictions dicts with "true"/"pred" columns. Excludes ERROR rows."""
    y_true = [r["true"] for r in rows if r["pred"] != "ERROR"]
    y_pred = [r["pred"] for r in rows if r["pred"] != "ERROR"]
    if not y_true:
        return {}

    classes = sorted(set(y_true) | set(y_pred))
    metrics = {
        "n": len(y_true),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_weighted": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall_weighted": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }

    if len(classes) == 2:
        pos = classes[-1]  # "1"/"True"-ish label sorts last for the {0,1}/{False,True} cases used here
        neg = classes[0]
        metrics["confusion"] = (
            f"TP={sum(1 for t, p in zip(y_true, y_pred) if t == pos and p == pos)}  "
            f"FP={sum(1 for t, p in zip(y_true, y_pred) if t == neg and p == pos)}  "
            f"TN={sum(1 for t, p in zip(y_true, y_pred) if t == neg and p == neg)}  "
            f"FN={sum(1 for t, p in zip(y_true, y_pred) if t == pos and p == neg)}"
        )
    else:
        counts = {c: sum(1 for t in y_true if t == c) for c in classes}
        metrics["confusion"] = "class counts (true): " + ", ".join(f"{c}={n}" for c, n in counts.items())

    return metrics


def build_run_section(model: str, prompt: str, meta: dict, rows: list, usage: dict) -> list:
    n_sampled = len(rows)
    n_errors = sum(1 for r in rows if r["pred"] == "ERROR")
    metrics = compute_metrics(rows)

    lines = [f"Model: {model}   Prompt: {prompt}", "-" * 60]
    lines.append(f"Prompt file:     {meta.get('prompt_path', '?')}")
    lines.append(f"Sentences:       {n_sampled} sampled / {meta.get('n_available', '?')} available (seed={meta.get('seed', '?')})")
    lines.append(f"Batch size:      {meta.get('batch_size', '?')}")
    lines.append(f"Errors:          {n_errors}")

    if metrics:
        lines.append(f"Accuracy:             {metrics['accuracy']:.3f}")
        lines.append(f"Precision (weighted): {metrics['precision_weighted']:.3f}")
        lines.append(f"Recall (weighted):    {metrics['recall_weighted']:.3f}")
        lines.append(f"F1 (weighted):        {metrics['f1_weighted']:.3f}")
        lines.append(f"F1 (macro):           {metrics['f1_macro']:.3f}")
        lines.append(f"Confusion:            {metrics['confusion']}")
    else:
        lines.append("(no successfully classified rows)")

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
                f"Extrapolated cost for full test set (N={n_available}): "
                f"${cost_per_sentence * n_available:.2f}"
            )
        lines.append(f"Total time:           {format_duration(duration_ms)}")
        rate_per_1000 = duration_ms / n_sampled * 1000
        lines.append(f"Rate:                 {format_duration(rate_per_1000)} per 1,000 sentences")
        if n_available:
            lines.append(
                f"Extrapolated time for full test set (N={n_available}): "
                f"{format_duration(duration_ms / n_sampled * n_available)}"
            )
    lines.append("")
    return lines, metrics, cost, n_sampled, duration_ms


def build_summary_table(summary_rows: list) -> list:
    """Ranked best-F1(weighted)-first; runs with no successfully classified rows sort last."""
    ranked = sorted(summary_rows, key=lambda r: r[2]["f1_weighted"] if r[2] else -1, reverse=True)

    header = (
        f"{'Model':<22} {'Prompt':<8} {'N':>5} {'Accuracy':>9} {'Prec(w)':>8} {'Recall(w)':>10} "
        f"{'F1(w)':>7} {'F1(macro)':>10} {'Cost/1k($)':>11} {'Time':>8} {'Rate/1k':>9}"
    )
    lines = [header, "-" * len(header)]
    for model, prompt, metrics, cost, n_sampled, duration_ms in ranked:
        acc = f"{metrics['accuracy']:.3f}" if metrics else "n/a"
        prec = f"{metrics['precision_weighted']:.3f}" if metrics else "n/a"
        rec = f"{metrics['recall_weighted']:.3f}" if metrics else "n/a"
        f1w = f"{metrics['f1_weighted']:.3f}" if metrics else "n/a"
        f1m = f"{metrics['f1_macro']:.3f}" if metrics else "n/a"
        cost_per_1k = cost / n_sampled * 1000 if n_sampled else 0.0
        time_str = format_duration(duration_ms)
        rate_str = format_duration(duration_ms / n_sampled * 1000) if n_sampled else "n/a"
        lines.append(
            f"{model:<22} {prompt:<8} {n_sampled:>5} {acc:>9} {prec:>8} {rec:>10} {f1w:>7} {f1m:>10} "
            f"{cost_per_1k:>11.4f} {time_str:>8} {rate_str:>9}"
        )
    return lines


def build_latex_table(summary_rows: list, baseline: dict | None) -> list:
    """LaTeX tabular of the summary table: one row per (model, prompt) run,
    ranked best-F1(weighted)-first, with the SOCCAT mDeBERTa baseline pinned
    as the last row. N is dropped (fixed at 1260 for every run); cost and
    rate/1k are kept. All numeric metrics use 2 decimals."""
    ranked = sorted(summary_rows, key=lambda r: r[2]["f1_weighted"] if r[2] else -1, reverse=True)

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\begin{tabular}{llrrrrrr}",
        r"\toprule",
        r"Model & Prompt & Accuracy & Precision & Recall & F1 & Cost/1k (\$) & Rate/1k \\",
        r"\midrule",
    ]
    for model, prompt, metrics, cost, n_sampled, duration_ms in ranked:
        acc = f"{metrics['accuracy']:.2f}" if metrics else "--"
        prec = f"{metrics['precision_weighted']:.2f}" if metrics else "--"
        rec = f"{metrics['recall_weighted']:.2f}" if metrics else "--"
        f1w = f"{metrics['f1_weighted']:.2f}" if metrics else "--"
        cost_per_1k = cost / n_sampled * 1000 if n_sampled else 0.0
        rate_str = format_duration(duration_ms / n_sampled * 1000) if n_sampled else "--"
        lines.append(
            f"{model} & {prompt} & {acc} & {prec} & {rec} & {f1w} & {cost_per_1k:.2f} & {rate_str} \\\\"
        )

    lines.append(r"\midrule")
    if baseline is not None:
        lines.append(
            f"SOCCAT (mDeBERTa) & -- & {baseline['accuracy']:.2f} & {baseline['precision']:.2f} & "
            f"{baseline['recall']:.2f} & {baseline['f1_weighted']:.2f} & -- & -- \\\\"
        )
    else:
        lines.append(r"SOCCAT (mDeBERTa) & -- & -- & -- & -- & -- & -- & -- \\")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Step 1 classification performance across LLM runs and the SOCCAT mDeBERTa baseline.}",
        r"\label{tab:llm-step1}",
        r"\end{table}",
    ])
    return lines


def main():
    runs = find_runs(OUTPUT_ROOT)

    lines = []
    lines.append("Step 1 LLM classification report -- all runs")
    lines.append("=" * 60)
    lines.append(f"Generated:  {datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"Runs found: {len(runs)}  (in {OUTPUT_ROOT.relative_to(REPO_ROOT)})")
    lines.append("")
    baseline = load_baseline_metrics()
    lines.append(f"mDeBERTa baseline for reference ({BASELINE_PATH.relative_to(REPO_ROOT)}):")
    lines.append(format_baseline_summary(baseline))
    lines.append("")

    report_dir = SCRIPT_DIR / "output"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "step_1_report.txt"
    table_path = report_dir / "step_1_table.tex"

    if not runs:
        lines.append("No completed runs found. Run classify_step1.py first.")
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

        section, metrics, cost, n_sampled, duration_ms = build_run_section(model, prompt, meta, rows, usage)
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
    table_path.write_text("\n".join(build_latex_table(summary_rows, baseline)) + "\n", encoding="utf-8")

    print(f"Report: {report_path}")
    print(f"LaTeX table: {table_path}")
    print(f"{len(runs)} run(s) included")


if __name__ == "__main__":
    main()
