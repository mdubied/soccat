"""
Compute performance metrics and cost estimates across all Step 1 LLM
classification runs, and write a single combined txt report.

Scans llm/classification/output/step_1/ for every {model}__{prompt} run
directory produced by classify_step1.py, and reports on whichever ones it
finds -- it is not an error for some model/prompt combinations to be
missing (e.g. you haven't run --model claude-opus-5 yet). Kept separate
from classify_step1.py so the report can be regenerated (or runs compared)
without spending any more Claude credits.

Step 1 is a single binary classification problem (vs. has_group). Precision,
recall and F1 are positive-class scores (positive = sentence mentions a
social group), plus macro F1 -- the same metrics as SOCCAT's step 1 table
(data/model_performance/step_1/performance_all_levels.csv, *_binary and
F1_macro columns). Support-weighted scores are not used: with 28% positives
they are dominated by the negative class (weighted recall == accuracy).

Gold labels are read from the test file, not from predictions.csv, and
predictions are matched to test sentences by (id, text): the test file reuses
2 ids for 7 different sentences, and classify_step1.py keys its output by id,
so runs made before that was fixed (key now (id, text)) hold one prediction
per duplicate id written several times. Those copies are dropped, i.e. each
run is scored on its distinct sentences (1,255 of 1,260 for such runs, until
the 5 overwritten sentences are classified by resuming the run).

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
TEST_PATH = REPO_ROOT / "data" / "model_performance" / "step_1" / "test_with_all_outlets.json"
POS = "1"  # positive class: sentence mentions a social group


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
    performance_all_levels.csv, or None if unavailable. Positive-class
    (*_binary) and macro scores, as computed for the LLM runs below."""
    if not BASELINE_PATH.exists():
        return None
    with BASELINE_PATH.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("group") == "ALL":
                return {
                    "accuracy": float(row["Accuracy"]),
                    "precision": float(row["Precision_binary"]),
                    "recall": float(row["Recall_binary"]),
                    "f1": float(row["F1_binary"]),
                    "f1_macro": float(row["F1_macro"]),
                    "n": row["N"],
                }
    return None


def format_baseline_summary(baseline: dict | None) -> str:
    if baseline is None:
        return "  (not available)"
    return (
        f"  Accuracy={baseline['accuracy']:.3f}  Precision={baseline['precision']:.3f}  "
        f"Recall={baseline['recall']:.3f}  F1={baseline['f1']:.3f}  "
        f"F1(macro)={baseline['f1_macro']:.3f}  (N={baseline['n']}; precision/recall/F1 = positive class)"
    )


def format_duration(ms: float) -> str:
    seconds = ms / 1000
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, secs = divmod(seconds, 60)
    return f"{int(minutes)}m {secs:.0f}s"


def load_gold() -> dict:
    """{(id, text): "0"/"1"} from the step 1 test file (has_group)."""
    with TEST_PATH.open("r", encoding="utf-8") as f:
        return {(str(r["id"]), r["text"]): str(int(r["has_group"]))
                for r in (json.loads(line) for line in f if line.strip())}


def distinct_rows(rows: list, gold: dict) -> list:
    """One row per distinct test sentence (duplicate-id copies dropped, see the
    module docstring), with "true" taken from the test file."""
    out, seen = [], set()
    for r in rows:
        key = (r["id"], r["text"])
        if key in seen:
            continue
        seen.add(key)
        out.append({**r, "true": gold[key]})
    return out


def compute_metrics(rows: list) -> dict:
    """rows: predictions dicts with "true"/"pred" columns. Excludes ERROR rows."""
    y_true = [r["true"] for r in rows if r["pred"] != "ERROR"]
    y_pred = [r["pred"] for r in rows if r["pred"] != "ERROR"]
    if not y_true:
        return {}

    pairs = list(zip(y_true, y_pred))
    tp = sum(t == POS and p == POS for t, p in pairs)
    fp = sum(t != POS and p == POS for t, p in pairs)
    fn = sum(t == POS and p != POS for t, p in pairs)
    tn = len(pairs) - tp - fp - fn
    return {
        "n": len(y_true),
        "n_pos": tp + fn,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, pos_label=POS, zero_division=0),
        "recall": recall_score(y_true, y_pred, pos_label=POS, zero_division=0),
        "f1": f1_score(y_true, y_pred, pos_label=POS, zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "confusion": f"TP={tp}  FP={fp}  TN={tn}  FN={fn}",
    }


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
        lines.append(f"Scored sentences:     {metrics['n']} distinct ({metrics['n_pos']} positive)")
        lines.append(f"Accuracy:             {metrics['accuracy']:.3f}")
        lines.append(f"Precision (pos.):     {metrics['precision']:.3f}")
        lines.append(f"Recall (pos.):        {metrics['recall']:.3f}")
        lines.append(f"F1 (pos.):            {metrics['f1']:.3f}")
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
    """Ranked best-F1(positive class)-first; runs with no successfully classified rows sort last."""
    ranked = sorted(summary_rows, key=lambda r: r[2]["f1"] if r[2] else -1, reverse=True)

    header = (
        f"{'Model':<22} {'Prompt':<8} {'N':>5} {'Accuracy':>9} {'Prec':>8} {'Recall':>10} "
        f"{'F1':>7} {'F1(macro)':>10} {'Cost/1k($)':>11} {'Time':>8} {'Rate/1k':>9}"
    )
    lines = [header, "-" * len(header)]
    for model, prompt, metrics, cost, n_sampled, duration_ms in ranked:
        acc = f"{metrics['accuracy']:.3f}" if metrics else "n/a"
        prec = f"{metrics['precision']:.3f}" if metrics else "n/a"
        rec = f"{metrics['recall']:.3f}" if metrics else "n/a"
        f1w = f"{metrics['f1']:.3f}" if metrics else "n/a"
        f1m = f"{metrics['f1_macro']:.3f}" if metrics else "n/a"
        cost_per_1k = cost / n_sampled * 1000 if n_sampled else 0.0
        time_str = format_duration(duration_ms)
        rate_str = format_duration(duration_ms / n_sampled * 1000) if n_sampled else "n/a"
        lines.append(
            f"{model:<22} {prompt:<8} {(metrics['n'] if metrics else 0):>5} {acc:>9} {prec:>8} {rec:>10} {f1w:>7} {f1m:>10} "
            f"{cost_per_1k:>11.4f} {time_str:>8} {rate_str:>9}"
        )
    return lines


def parse_model_thinking(model: str) -> tuple:
    """"claude-sonnet-5-high" -> ("Sonnet-5", "High"). Split out as its own column
    (rather than left in a single "Model" string) to match step 2's per-category
    tables, which need Model/Thinking/Prompt as separate header lines."""
    base = model[len("claude-"):] if model.startswith("claude-") else model
    for suffix, thinking in (("-high", "High"), ("-low", "Low")):
        if base.endswith(suffix):
            name = base[: -len(suffix)]
            return name[:1].upper() + name[1:], thinking
    return base[:1].upper() + base[1:], "--"


def build_latex_table(summary_rows: list, baseline: dict | None) -> list:
    """LaTeX tabular of the summary table: one row per (model, prompt) run,
    ranked best-F1(positive class)-first, with the SOCCAT mDeBERTa baseline
    pinned as the last row. Positive-class precision/recall/F1 and macro F1;
    cost and rate/1k are kept. All numeric metrics use 2 decimals."""
    ranked = sorted(summary_rows, key=lambda r: r[2]["f1"] if r[2] else -1, reverse=True)
    n_llm = sorted({m["n"] for _, _, m, *_ in summary_rows if m})

    lines = [
        r"\begin{table}[htb]",
        r"\centering",
        r"\begin{threeparttable}",
        r"\begin{tabular}{lllrrrrrrr}",
        r"\toprule",
        r"Model & Thinking & Prompt & Accuracy & Precision & Recall & F1 & Macro F1 & Cost/1k (\$) & Rate/1k \\",
        r"\midrule",
    ]
    for model, prompt, metrics, cost, n_sampled, duration_ms in ranked:
        model_name, thinking = parse_model_thinking(model)
        acc = f"{metrics['accuracy']:.2f}" if metrics else "--"
        prec = f"{metrics['precision']:.2f}" if metrics else "--"
        rec = f"{metrics['recall']:.2f}" if metrics else "--"
        f1w = f"{metrics['f1']:.2f}" if metrics else "--"
        f1m = f"{metrics['f1_macro']:.2f}" if metrics else "--"
        cost_per_1k = cost / n_sampled * 1000 if n_sampled else 0.0
        rate_str = format_duration(duration_ms / n_sampled * 1000) if n_sampled else "--"
        lines.append(
            f"{model_name} & {thinking} & {prompt.capitalize()} & {acc} & {prec} & {rec} & {f1w} & {f1m} & "
            f"{cost_per_1k:.2f} & {rate_str} \\\\"
        )

    lines.append(r"\midrule")
    if baseline is not None:
        lines.append(
            f"SOCCAT & -- & -- & {baseline['accuracy']:.2f} & {baseline['precision']:.2f} & "
            f"{baseline['recall']:.2f} & {baseline['f1']:.2f} & {baseline['f1_macro']:.2f} & -- & -- \\\\"
        )
    else:
        lines.append(r"SOCCAT & -- & -- & -- & -- & -- & -- & -- & -- & -- \\")

    n_soccat = int(baseline["n"]) if baseline else None
    if n_soccat is not None and n_llm == [n_soccat]:
        n_note = f"Held-out test set, N = {n_soccat:,} sentences."
    else:
        n_llm_str = ", ".join(f"{n:,}" for n in n_llm) or "--"
        n_note = (f"Held-out test set: SOCCAT N = {n_soccat:,}; LLM runs N = {n_llm_str} sentences "
                  "(the others lack an LLM prediction)." if n_soccat is not None else "")
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\begin{tablenotes}[flushleft]",
        r"\footnotesize",
        r"\item \textit{Note:} Precision, recall and F1 refer to the positive class (sentence mentions "
        r"a social group); macro F1 averages the F1 scores of both classes. " + n_note,
        r"\end{tablenotes}",
        r"\end{threeparttable}",
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
    gold = load_gold()
    for model, prompt, run_dir in runs:
        rows = list(csv.DictReader((run_dir / "predictions.csv").open("r", encoding="utf-8", newline="")))
        rows = distinct_rows(rows, gold)
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
