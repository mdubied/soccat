#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Zero-shot NLI: SOCCAT's base model without fine-tuning, same data and folds as the single NLI model
=================================================================================================
Shows what fine-tuning on the annotations adds: the starting model of SOCCAT
(MoritzLaurer/mDeBERTa-v3-base-mnli-xnli), not trained on any annotation, scores
every (sentence, hypothesis) pair with SOCCAT's own hypotheses ("This sentence
refers to ... as a social group, specifically ..."). These pairs are read from
input/ (the five fold files of a single NLI model trained with these hypotheses,
kept here as this script's input; only the pair columns are used), not from
../single_nli/, whose hypotheses name the specific label only.

P(entailment) follows the usual zero-shot multi-label convention: softmax over
the model's entailment and contradiction logits only (neutral ignored); a pair is
predicted positive if > 0.5, as for SOCCAT. Nothing is trained, so the folds only
organise the outputs: each fold's test sentences are scored and written with the
same files as the fine-tuned models (../common.py: same 7,234 sentences, gold
labels and fold split as the single NLI model), so the results can be read by the
single_nli/ scripts.

Runs on a GPU if one is available (fp16 inference with --fp16). Finished folds
are skipped. Pairs are sorted by length before batching (faster); the outputs
keep the original order.

Usage (from this folder):
    python predict_cv.py                  # all folds, then finalize
    python predict_cv.py --folds 0        # one fold
    python predict_cv.py --finalize       # cross-fold files only
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import common  # noqa: E402

MODEL_NAME = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"
MAX_LENGTH, BATCH = 256, 128


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=HERE / "data", help="Output folder (default: ./data)")
    ap.add_argument("--pairs_dir", type=Path, default=HERE / "input",
                    help="Folder with the fold_*_human_vs_model.csv giving the pairs (data, hypotheses "
                         "and folds; default: ./input, SOCCAT's hypotheses naming the broad class)")
    ap.add_argument("--folds", type=int, nargs="+", default=list(range(common.K_FOLDS)))
    ap.add_argument("--finalize", action="store_true", help="Only write the cross-fold files")
    ap.add_argument("--model", default=MODEL_NAME, help="Hugging Face id or local folder of the NLI model")
    ap.add_argument("--max_length", type=int, default=MAX_LENGTH)
    ap.add_argument("--batch", type=int, default=BATCH)
    ap.add_argument("--fp16", action="store_true", help="fp16 inference on GPU (faster, negligible effect on scores)")
    return ap.parse_args()


def entail_contra_indices(config):
    lab = {v.lower(): int(k) for k, v in config.id2label.items()}
    ent = next((i for name, i in lab.items() if name.startswith("entail")), None)
    con = next((i for name, i in lab.items() if name.startswith("contra")), None)
    if ent is None or con is None:
        raise ValueError(f"cannot find entailment/contradiction in id2label: {config.id2label}")
    return ent, con


def score_pairs(premises, hypotheses, model, tokenizer, args, device):
    import torch
    ent, con = entail_contra_indices(model.config)
    order = np.argsort([len(p) + len(h) for p, h in zip(premises, hypotheses)])
    out = np.empty(len(order))
    n_batches, t0 = -(-len(order) // args.batch), time.time()
    with torch.inference_mode():
        for start in range(0, len(order), args.batch):
            idx = order[start:start + args.batch]
            enc = tokenizer([premises[i] for i in idx], [hypotheses[i] for i in idx], truncation=True,
                            max_length=args.max_length, padding=True, return_tensors="pt").to(device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=args.fp16 and device.type == "cuda"):
                logits = model(**enc).logits.float()
            out[idx] = torch.softmax(logits[:, [ent, con]], dim=-1)[:, 0].cpu().numpy()
            b = start // args.batch + 1
            if b % 50 == 0 or b == n_batches:
                done, el = start + len(idx), time.time() - t0
                common.log(f"  {done:,}/{len(order):,} pairs ({done / len(order):.0%}) | batch {b}/{n_batches} | "
                           f"elapsed {common.fmt_duration(el)} | ETA {common.fmt_duration(el / done * (len(order) - done))}")
    return out


def main():
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    pairs, reference = common.load_pairs(args.pairs_dir)
    fold_map = common.get_fold_map(pairs, reference, args.out)
    (args.out / "run_config.json").write_text(json.dumps(
        {k: str(v) for k, v in vars(args).items()}, indent=2), encoding="utf-8")

    todo = []
    if not args.finalize:
        for k in args.folds:
            if common.fold_done(args.out, k):
                common.log(f"Fold {k + 1}/{common.K_FOLDS}: already done, skipping")
            else:
                todo.append(k)
    if todo:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        common.log(f"Device: {torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU (slow!)'} | folds to run: {[k + 1 for k in todo]}")
        common.log(f"Loading model {args.model}")
        tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
        model = AutoModelForSequenceClassification.from_pretrained(args.model).to(device).eval()
        common.log("Model loaded")
        for fold in todo:
            t_fold = time.time()
            common.log(f"========== Fold {fold + 1}/{common.K_FOLDS} ==========")
            test_ids = set(fold_map.loc[fold_map.fold == fold, "sentence_id"])
            test_pairs = pairs[pairs["sentence_id"].isin(test_ids)].reset_index(drop=True)
            common.log(f"Scoring {len(test_pairs):,} pairs ({len(test_ids):,} sentences)")
            prob = score_pairs(test_pairs["premise"].astype(str).tolist(), test_pairs["hypothesis"].astype(str).tolist(),
                               model, tokenizer, args, device)
            common.write_fold_outputs(args.out, fold, test_pairs, prob,
                                      extra={"model": args.model, "formulation": "zero-shot NLI"})
            common.log(f"Fold {fold + 1}/{common.K_FOLDS} done in {common.fmt_duration(time.time() - t_fold)}")

    if all(common.fold_done(args.out, k) for k in range(common.K_FOLDS)):
        common.finalize(args.out, pairs, fold_map)
    else:
        common.log("Not all folds are done yet; run with --finalize once they are.")


if __name__ == "__main__":
    main()
