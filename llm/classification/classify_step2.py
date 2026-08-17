"""
Classify sentences with an LLM (via a local Claude Code session, not the API)
for Step 2 of the pipeline: given a sentence that mentions a social group,
which categories (broad class + specific label) does it express?

Unlike Step 1, Step 2 has no single fixed held-out test file in this repo:
src/step_2/step_2_cv_pipeline.py reformulates each sentence into per-category
NLI pairs and cross-validates with 5 folds *independently per broad
category* (fold membership differs category by category), then keeps
whichever fold scored best per category as that category's final model --
so there's no persisted, unified test split to replicate exactly.

Instead this script samples directly from the full annotated corpus,
data/manual_annotations/step_2/annotations_ground_truth.csv (9,639
sentences already confirmed to mention a social group, with human-coded
broad_category/specific_group_new ground truth). By default it samples a
fold-sized slice (~1/5 of the corpus, matching the scale of one CV fold)
rather than the full corpus, since that is roughly the amount of held-out
data mDeBERTa's reported per-category metrics are each based on. Ground
truth labels are normalised to canonical taxonomy labels the same way
src/step_2/convert_annotations.py normalises them for mDeBERTa's NLI pairs
(see step2_taxonomy.py) so LLM and mDeBERTa numbers stay comparable.

Sentences are sent to `claude -p` in batches (one CLI call per batch) to
limit fixed per-call overhead. Results are written incrementally to a CSV, so
a run can be interrupted and resumed: rows whose id already appears in the
output file are skipped. If a batch's response is malformed, that batch is
retried one sentence at a time so a single bad response doesn't cost the
whole batch.

This script only classifies and records raw predictions + cost. Aggregate
performance metrics and the human-readable report are computed separately
by llm/analysis/report_step2.py, so reports can be regenerated (or runs
compared) without reclassifying anything.

Usage:
    # Fold-sized sample (~1/5 of the corpus) with the short prompt:
    python classify_step2.py --prompt short --model claude-sonnet-5

    # Full corpus:
    python classify_step2.py --prompt short --model claude-sonnet-5 --all

    # Explicit sample size, bigger batches:
    python classify_step2.py --prompt medium --model claude-sonnet-5 --limit 500 --batch-size 25
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
from step2_taxonomy import ALL_BROAD_CATEGORIES, ALL_SPECIFIC_LABELS, parse_true_categories

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent

DEFAULT_DATA_PATH = REPO_ROOT / "data" / "manual_annotations" / "step_2" / "annotations_ground_truth.csv"
PROMPTS_DIR = SCRIPT_DIR / "prompts" / "step_2"
OUTPUT_ROOT = SCRIPT_DIR / "output" / "step_2"

CATEGORY_SCHEMA = {
    "type": "object",
    "properties": {
        "broad_category": {"type": "string", "enum": ALL_BROAD_CATEGORIES},
        "specific_category": {"type": "string", "enum": ALL_SPECIFIC_LABELS},
    },
    "required": ["broad_category", "specific_category"],
    "additionalProperties": False,
}

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
                    "categories": {"type": "array", "items": CATEGORY_SCHEMA},
                },
                "required": ["id", "has_social_category", "categories"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["results"],
    "additionalProperties": False,
})


def load_data(path: Path) -> list:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def load_prompt(prompt_version: str) -> tuple:
    path = PROMPTS_DIR / f"step_2_{prompt_version}.md"
    if not path.exists():
        available = sorted(p.stem for p in PROMPTS_DIR.glob("step_2_*.md"))
        raise SystemExit(f"error: prompt file not found: {path}\navailable: {available}")
    return path, path.read_text(encoding="utf-8")


def select_rows(all_rows: list, limit: int, seed: int) -> list:
    """Deterministically shuffle once per seed, then take the first `limit` rows.

    This keeps samples nested across runs: a bigger --limit with the same
    --seed is a superset of a smaller one, so predictions.csv can just be
    resumed and extended rather than reclassified from scratch.
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
        '"has_social_category": true or false, "categories": [{"broad_category": "...", '
        '"specific_category": "..."}]} (categories is an empty array if none apply)'
    )


def classify_batch(rows: list, system_prompt_file: str, model: str, timeout: int, claude_path: str, totals: UsageTotals) -> dict:
    """rows: list of row dicts (each with "id" and "text"). Returns {id: result_dict}."""
    items = [(r["id"], r["text"]) for r in rows]
    local_to_real = {str(i): real_id for i, (real_id, _) in enumerate(items)}
    message = build_batch_user_message(items)
    response = call_claude(message, system_prompt_file, BATCH_SCHEMA, model, timeout, claude_path)
    totals.add(response)

    results = response.structured_output["results"]
    by_local_id = {str(r["id"]): r for r in results}
    if len(results) != len(by_local_id):
        raise RuntimeError("batch response contains duplicate ids")
    missing = [i for i in local_to_real if i not in by_local_id]
    if missing:
        raise RuntimeError(f"batch response missing ids: {missing}")
    return {local_to_real[i]: r for i, r in by_local_id.items()}


def write_run_meta(path: Path, args, prompt_path: Path, n_available: int) -> None:
    meta = {
        "model": args.model,
        "prompt": args.prompt,
        "prompt_path": str(prompt_path.relative_to(REPO_ROOT)),
        "data_path": str(args.data_path.relative_to(REPO_ROOT)),
        "seed": args.seed,
        "batch_size": args.batch_size,
        "n_available": n_available,
    }
    path.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prompt", required=True, choices=["short", "medium", "long"], help="Prompt version")
    parser.add_argument("--model", required=True, help="Model, e.g. claude-sonnet-5, claude-haiku-4-5")
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH, help="Annotated corpus CSV")
    parser.add_argument("--limit", type=int, default=None,
                         help="Number of sentences to sample (default: ~1/5 of the corpus, i.e. one CV-fold-sized sample)")
    parser.add_argument("--all", action="store_true", help="Classify the full corpus, ignoring --limit")
    parser.add_argument("--seed", type=int, default=42, help="Sampling seed (default: 42)")
    parser.add_argument("--batch-size", type=int, default=20, help="Sentences per claude invocation")
    parser.add_argument("--timeout", type=int, default=180, help="Per-call subprocess timeout in seconds")
    parser.add_argument("--sleep", type=float, default=0.0, help="Seconds to sleep between calls")
    args = parser.parse_args()

    prompt_path, system_prompt = load_prompt(args.prompt)
    claude_path = resolve_claude_executable()

    all_rows = load_data(args.data_path)
    for r in all_rows:
        r["id"] = str(r["id"])

    if args.all:
        limit = None
    elif args.limit is not None:
        limit = args.limit
    else:
        limit = len(all_rows) // 5  # fold-sized default: mirrors the ~1/5 held out per CV fold

    sampled_rows = select_rows(all_rows, limit, args.seed)

    run_dir = OUTPUT_ROOT / f"{args.model}__{args.prompt}"
    run_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = run_dir / "predictions.csv"
    usage_path = run_dir / "usage_totals.json"
    meta_path = run_dir / "run_meta.json"

    write_run_meta(meta_path, args, prompt_path, len(all_rows))

    existing = load_existing_predictions(predictions_path)
    todo = [r for r in sampled_rows if r["id"] not in existing]

    output_fields = [
        "id", "text", "outlet", "country",
        "true_has_social_category", "true_categories",
        "pred_has_social_category", "pred_categories",
        "has_social_category_correct", "categories_exact_match",
    ]

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

    print(f"Classifying {len(todo)} sentences (prompt={args.prompt}, model={args.model})")

    totals = UsageTotals.load(usage_path)

    with tempfile.TemporaryDirectory() as tmpdir:
        system_prompt_file = write_system_prompt(Path(tmpdir), system_prompt)

        def record(row, result):
            true_categories = parse_true_categories(row.get("specific_group_new"))
            true_has = bool(true_categories)
            pred_categories = sorted(
                {(c["broad_category"], c["specific_category"]) for c in result["categories"]}
            )
            pred_has = bool(result["has_social_category"])
            existing[row["id"]] = {
                "id": row["id"],
                "text": row["text"],
                "outlet": row.get("outlet", ""),
                "country": row.get("country", ""),
                "true_has_social_category": int(true_has),
                "true_categories": json.dumps(true_categories, ensure_ascii=False),
                "pred_has_social_category": int(pred_has),
                "pred_categories": json.dumps(pred_categories, ensure_ascii=False),
                "has_social_category_correct": int(true_has == pred_has),
                "categories_exact_match": int(set(true_categories) == set(pred_categories)),
            }

        def record_error(row, error):
            tqdm.write(f"row {row['id']}: FAILED ({error})")
            true_categories = parse_true_categories(row.get("specific_group_new"))
            existing[row["id"]] = {
                "id": row["id"], "text": row["text"], "outlet": row.get("outlet", ""),
                "country": row.get("country", ""),
                "true_has_social_category": int(bool(true_categories)),
                "true_categories": json.dumps(true_categories, ensure_ascii=False),
                "pred_has_social_category": "ERROR", "pred_categories": "ERROR",
                "has_social_category_correct": "", "categories_exact_match": "",
            }

        batches = [todo[i : i + args.batch_size] for i in range(0, len(todo), args.batch_size)]
        for batch in tqdm(batches, desc="classifying", unit="batch"):
            try:
                by_id = classify_batch(batch, system_prompt_file, args.model, args.timeout, claude_path, totals)
                for row in batch:
                    record(row, by_id[row["id"]])
            except Exception as e:
                tqdm.write(f"batch of {len(batch)} FAILED ({e}) - retrying one sentence at a time")
                for row in batch:
                    try:
                        by_id = classify_batch([row], system_prompt_file, args.model, args.timeout, claude_path, totals)
                        record(row, by_id[row["id"]])
                    except Exception as e2:
                        record_error(row, e2)
            persist()
            totals.save(usage_path)
            if args.sleep:
                time.sleep(args.sleep)

    n_errors = sum(1 for r in existing.values() if r["pred_has_social_category"] == "ERROR")
    print(f"\nPredictions: {predictions_path}  ({len(sampled_rows)} rows, {n_errors} errors)")
    print(f"Cost so far: ${totals.cost_usd:.4f} ({totals.n_calls} calls)")


if __name__ == "__main__":
    main()
