"""
Classify sentences with an LLM (via a local Claude Code session, not the API)
for Step 1 of the pipeline: does this sentence mention a social group?

The test set is the same one used to report the mDeBERTa baseline in
data/model_performance/step_1/performance_all_levels.csv (N=1260, ground
truth column "has_group"), so results are directly comparable to
src/step_1/SOCCAT_mDeBERTa_replication.py.

Sentences are sent to `claude -p` in batches (one CLI call per batch) to
limit fixed per-call overhead. Results are written incrementally to a CSV, so
a run can be interrupted and resumed: rows whose id already appears in the
output file are skipped. If a batch's response is malformed, that batch is
retried one sentence at a time so a single bad response doesn't cost the
whole batch.

This script only classifies and records raw predictions + cost. Performance
metrics and the human-readable report are computed separately by
llm/analysis/report_step1.py, so reports can be regenerated (or runs compared)
without reclassifying anything.

Usage:
    # Full validation set (1260 sentences, default) with the short prompt:
    python classify_step1.py --prompt short --model claude-sonnet-5

    # Smoke test on 50 random sentences:
    python classify_step1.py --prompt short --model claude-sonnet-5 --limit 50

    # Larger sample, bigger batches:
    python classify_step1.py --prompt medium --model claude-sonnet-5 --limit 300 --batch-size 25

    # Low-effort (less thinking) run -- output folder becomes claude-sonnet-5-low__short:
    python classify_step1.py --prompt short --model claude-sonnet-5 --effort low --limit 50

    # Then, to see performance + cost across all runs:
    python ../analysis/report_step1.py
"""

import argparse
import csv
import json
import random
import tempfile
import time
from pathlib import Path

from tqdm import tqdm

from claude_runner import UsageTotals, call_claude, resolve_claude_executable, write_system_prompt

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent

DEFAULT_TEST_PATH = REPO_ROOT / "data" / "model_performance" / "step_1" / "test_with_all_outlets.json"
PROMPTS_DIR = SCRIPT_DIR / "prompts" / "step_1"
OUTPUT_ROOT = SCRIPT_DIR / "output" / "step_1"

BATCH_SCHEMA = json.dumps({
    "type": "object",
    "properties": {
        "results": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "has_social_category": {"type": "boolean"},
                },
                "required": ["id", "has_social_category"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["results"],
    "additionalProperties": False,
})


def load_test_set(path: Path) -> list:
    """Read the JSONL test set into a list of dicts."""
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_prompt(prompt_version: str) -> tuple:
    path = PROMPTS_DIR / f"step_1_{prompt_version}.md"
    if not path.exists():
        available = sorted(p.stem for p in PROMPTS_DIR.glob("step_1_*.md"))
        raise SystemExit(f"error: prompt file not found: {path}\navailable: {available}")
    return path, path.read_text(encoding="utf-8")


def select_rows(all_rows: list, limit: int, seed: int) -> list:
    """Deterministically shuffle once per seed, then take the first `limit` rows.

    This keeps samples nested across runs: --limit 200 with the same --seed
    is a superset of --limit 50, so predictions.csv can just be resumed and
    extended rather than reclassified from scratch.
    """
    order = list(range(len(all_rows)))
    random.Random(seed).shuffle(order)
    if limit is not None:
        order = order[:limit]
    return [all_rows[i] for i in order]


def load_existing_predictions(output_path: Path) -> dict:
    if not output_path.exists():
        return {}
    with output_path.open("r", encoding="utf-8", newline="") as f:
        return {row["id"]: row for row in csv.DictReader(f)}


def build_batch_user_message(items: list) -> str:
    """items: list of (id, text) pairs, one or more sentences (a failed batch is
    retried through here too, one sentence at a time, to keep a single code path)."""
    payload = json.dumps(
        [{"id": str(i), "text": text} for i, (_, text) in enumerate(items)], ensure_ascii=False
    )
    return (
        "Apply the classification instructions above independently to each sentence in the "
        "JSON array below.\n\n"
        f"<sentences>{payload}</sentences>\n\n"
        'Output only JSON: {"results": [<one object per input sentence, same count, any '
        'order>]}, where each result is {"id": "<copied exactly from the input>", '
        '"has_social_category": true or false}'
    )


def model_label(model: str, effort: str) -> str:
    """Fold effort into the model portion of the output path, since low vs. high
    thinking on the same model can produce meaningfully different predictions."""
    return f"{model}-{effort}" if effort else model


def classify_batch(rows: list, system_prompt_file: str, model: str, effort: str, timeout: int, claude_path: str, totals: UsageTotals) -> dict:
    """rows: list of row dicts (each with "id" and "text"). Returns {id: bool}."""
    items = [(r["id"], r["text"]) for r in rows]
    local_to_real = {str(i): real_id for i, (real_id, _) in enumerate(items)}
    message = build_batch_user_message(items)
    response = call_claude(message, system_prompt_file, BATCH_SCHEMA, model, timeout, claude_path, effort)
    totals.add(response)

    results = response.structured_output["results"]
    by_local_id = {str(r["id"]): r["has_social_category"] for r in results}
    if len(results) != len(by_local_id):
        raise RuntimeError("batch response contains duplicate ids")
    missing = [i for i in local_to_real if i not in by_local_id]
    if missing:
        raise RuntimeError(f"batch response missing ids: {missing}")
    return {local_to_real[i]: pred for i, pred in by_local_id.items()}


def write_run_meta(path: Path, args, prompt_path: Path, n_available: int) -> None:
    meta = {
        "model": args.model,
        "effort": args.effort,
        "prompt": args.prompt,
        "prompt_path": str(prompt_path.relative_to(REPO_ROOT)),
        "test_path": str(args.test_path.relative_to(REPO_ROOT)),
        "seed": args.seed,
        "batch_size": args.batch_size,
        "n_available": n_available,
    }
    path.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prompt", required=True, choices=["short", "medium", "long"], help="Prompt version")
    parser.add_argument("--model", required=True, help="Model, e.g. claude-sonnet-5, claude-haiku-4-5")
    parser.add_argument("--effort", default=None, choices=["low", "medium", "high", "xhigh", "max"],
                         help="Thinking/effort level (default: the claude CLI's own default). Recorded as part of the model name in the output path.")
    parser.add_argument("--test-path", type=Path, default=DEFAULT_TEST_PATH, help="JSONL test set")
    parser.add_argument("--limit", type=int, default=None, help="Number of sentences to sample (default: full test set)")
    parser.add_argument("--all", action="store_true", help="Classify the full test set (default even without this flag; kept for parity with --limit)")
    parser.add_argument("--seed", type=int, default=42, help="Sampling seed (default: 42)")
    parser.add_argument("--batch-size", type=int, default=20, help="Sentences per claude invocation")
    parser.add_argument("--timeout", type=int, default=180, help="Per-call subprocess timeout in seconds")
    parser.add_argument("--sleep", type=float, default=0.0, help="Seconds to sleep between calls")
    args = parser.parse_args()

    prompt_path, system_prompt = load_prompt(args.prompt)
    claude_path = resolve_claude_executable()

    all_rows = load_test_set(args.test_path)
    for r in all_rows:
        r["id"] = str(r["id"])

    limit = None if args.all else args.limit
    sampled_rows = select_rows(all_rows, limit, args.seed)

    run_dir = OUTPUT_ROOT / f"{model_label(args.model, args.effort)}__{args.prompt}"
    run_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = run_dir / "predictions.csv"
    usage_path = run_dir / "usage_totals.json"
    meta_path = run_dir / "run_meta.json"

    write_run_meta(meta_path, args, prompt_path, len(all_rows))

    existing = load_existing_predictions(predictions_path)
    todo = [r for r in sampled_rows if r["id"] not in existing]

    output_fields = ["id", "text", "paper", "language", "true", "pred", "correct"]

    def persist():
        with predictions_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=output_fields)
            writer.writeheader()
            for row in sampled_rows:
                out_row = existing.get(row["id"])
                if out_row is not None:
                    writer.writerow(out_row)

    if not todo:
        print(f"Nothing to do: all {len(sampled_rows)} sampled rows already in {predictions_path}")
        return

    print(f"Classifying {len(todo)} sentences (prompt={args.prompt}, model={model_label(args.model, args.effort)})")

    totals = UsageTotals.load(usage_path)

    with tempfile.TemporaryDirectory() as tmpdir:
        system_prompt_file = write_system_prompt(Path(tmpdir), system_prompt)

        def record(row, pred):
            true = row["has_group"]
            existing[row["id"]] = {
                "id": row["id"],
                "text": row["text"],
                "paper": row.get("paper", ""),
                "language": row.get("language", ""),
                "true": true,
                "pred": int(bool(pred)),
                "correct": int(bool(pred)) == int(true),
            }

        def record_error(row, error):
            tqdm.write(f"row {row['id']}: FAILED ({error})")
            existing[row["id"]] = {
                "id": row["id"], "text": row["text"], "paper": row.get("paper", ""),
                "language": row.get("language", ""), "true": row["has_group"],
                "pred": "ERROR", "correct": "",
            }

        batches = [todo[i : i + args.batch_size] for i in range(0, len(todo), args.batch_size)]
        for batch in tqdm(batches, desc="classifying", unit="batch"):
            try:
                by_id = classify_batch(batch, system_prompt_file, args.model, args.effort, args.timeout, claude_path, totals)
                for row in batch:
                    record(row, by_id[row["id"]])
            except Exception as e:
                tqdm.write(f"batch of {len(batch)} FAILED ({e}) - retrying one sentence at a time")
                for row in batch:
                    try:
                        by_id = classify_batch([row], system_prompt_file, args.model, args.effort, args.timeout, claude_path, totals)
                        record(row, by_id[row["id"]])
                    except Exception as e2:
                        record_error(row, e2)
            persist()
            totals.save(usage_path)
            if args.sleep:
                time.sleep(args.sleep)

    n_errors = sum(1 for r in existing.values() if r["pred"] == "ERROR")
    print(f"\nPredictions: {predictions_path}  ({len(sampled_rows)} rows, {n_errors} errors)")
    print(f"Cost so far: ${totals.cost_usd:.4f} ({totals.n_calls} calls)")
    print("Run report with: python ../analysis/report_step1.py")


if __name__ == "__main__":
    main()
