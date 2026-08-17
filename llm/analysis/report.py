"""
Compute performance metrics and cost estimates across all LLM classification
runs found for a given pipeline step, and write a single combined txt report.

Scans llm/classification/output/step_{1,2}/ for every {model}__{prompt} run
directory produced by classify_step1.py / classify_step2.py, and reports on
whichever ones it finds -- it is not an error for some model/prompt
combinations to be missing (e.g. you haven't run --model claude-opus-5 yet).
Kept separate from the classification scripts so the report can be
regenerated (or runs compared) without spending any more Claude credits.

Usage:
    python report.py --step 1
    python report.py --step 2
"""

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent

BASELINE_PATHS = {
    1: REPO_ROOT / "data" / "model_performance" / "step_1" / "performance_all_levels.csv",
    2: REPO_ROOT / "data" / "model_performance" / "step_2" / "model_performance" / "broad_cat_mean_ci.csv",
}


def classification_output_root(step: int) -> Path:
    return REPO_ROOT / "llm" / "classification" / "output" / f"step_{step}"


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


def load_baseline_summary(step: int) -> str:
    path = BASELINE_PATHS.get(step)
    if path is None or not path.exists():
        return "  (not available)"

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return "  (empty baseline file)"

    if step == 1:
        for row in rows:
            if row.get("group") == "ALL":
                return (
                    f"  Accuracy={float(row['Accuracy']):.3f}  Precision={float(row['Precision']):.3f}  "
                    f"Recall={float(row['Recall']):.3f}  F1={float(row['F1']):.3f}  (N={row['N']})"
                )
        return "  (overall row not found)"

    # Step 2: broad_cat_mean_ci.csv has one row per broad category (ALL_LABELS),
    # no single overall row -- report an unweighted mean across categories.
    accs = [float(r["accuracy_mean"]) for r in rows if r.get("accuracy_mean")]
    f1s = [float(r["f1_macro_mean"]) for r in rows if r.get("f1_macro_mean")]
    if not accs:
        return "  (unrecognized baseline format)"
    return (
        f"  Accuracy={sum(accs) / len(accs):.3f}  F1(macro)={sum(f1s) / len(f1s):.3f}  "
        f"(unweighted mean across {len(rows)} broad categories, mDeBERTa cross-validation)"
    )


def compute_metrics(rows: list) -> dict:
    """rows: predictions dicts with "true"/"pred" columns. Excludes ERROR rows.
    Works for binary (step 1) or multi-class (step 2) labels."""
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
    lines.append("")
    return lines, metrics, cost, n_sampled


def build_summary_table(summary_rows: list) -> list:
    header = f"{'Model':<22} {'Prompt':<8} {'N':>5} {'Accuracy':>9} {'F1(w)':>7} {'F1(macro)':>10} {'Cost($)':>9}"
    lines = [header, "-" * len(header)]
    for model, prompt, metrics, cost, n_sampled in summary_rows:
        acc = f"{metrics['accuracy']:.3f}" if metrics else "n/a"
        f1w = f"{metrics['f1_weighted']:.3f}" if metrics else "n/a"
        f1m = f"{metrics['f1_macro']:.3f}" if metrics else "n/a"
        lines.append(f"{model:<22} {prompt:<8} {n_sampled:>5} {acc:>9} {f1w:>7} {f1m:>10} {cost:>9.4f}")
    return lines


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--step", type=int, required=True, choices=[1, 2], help="Pipeline step to report on")
    args = parser.parse_args()

    output_root = classification_output_root(args.step)
    runs = find_runs(output_root)

    lines = []
    lines.append(f"Step {args.step} LLM classification report -- all runs")
    lines.append("=" * 60)
    lines.append(f"Generated:  {datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"Runs found: {len(runs)}  (in {output_root.relative_to(REPO_ROOT)})")
    lines.append("")
    lines.append(f"mDeBERTa baseline for reference ({BASELINE_PATHS[args.step].relative_to(REPO_ROOT)}):")
    lines.append(load_baseline_summary(args.step))
    lines.append("")

    report_dir = SCRIPT_DIR / "output"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"step_{args.step}_report.txt"

    if not runs:
        lines.append(f"No completed runs found. Run classify_step{args.step}.py first.")
        report_path.write_text("\n".join(lines), encoding="utf-8")
        print(f"Report: {report_path}")
        print(f"No runs found under {output_root}")
        return

    summary_rows = []
    detail_lines = []
    for model, prompt, run_dir in runs:
        rows = list(csv.DictReader((run_dir / "predictions.csv").open("r", encoding="utf-8", newline="")))
        usage = json.loads((run_dir / "usage_totals.json").read_text(encoding="utf-8"))
        meta = json.loads((run_dir / "run_meta.json").read_text(encoding="utf-8"))

        section, metrics, cost, n_sampled = build_run_section(model, prompt, meta, rows, usage)
        detail_lines.extend(section)
        summary_rows.append((model, prompt, metrics, cost, n_sampled))

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
