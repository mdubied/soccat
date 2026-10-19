#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fine-tuned multi-label classifier: 5-fold CV, same data and folds as the single NLI model
=========================================================================================
Baseline for SOCCAT's NLI formulation: the same starting model
(MoritzLaurer/mDeBERTa-v3-base-mnli-xnli) fine-tuned as a standard multi-label
classifier -- one input per sentence (the sentence alone, no hypothesis) and a new
classification head with one sigmoid output per label (57 labels) -- instead of
one entailment decision per (sentence, hypothesis) pair.

Everything else follows examples/CV_combined_all_dimensions_v4.ipynb (the single
NLI model), see ../common.py: same 7,234 sentences, gold labels and 5-fold split
(checked against the single NLI model's), same hyper-parameters by default, the
same model selection within a fold (best epoch by pair-level accuracy on the
fold's test sentences, as in the NLI notebooks), and the same output files, so
the results can be read by the single_nli/ scripts. The sigmoid output of a label
plays the role of P(entailment): prob_entail = P(label present), predicted
present if > 0.5, written to one row per (sentence, label) pair.

Epochs: the NLI models see each sentence once per hypothesis (57 pairs) and
epoch, the classifier once per epoch, i.e. about 57 times fewer updates; hence
the higher default number of epochs (--epochs), still with best-epoch selection.

Runs on a GPU if one is available (fold by fold; --folds 2 runs fold 2 only, e.g.
from a Slurm job array, see run_train.sbatch). Finished folds are skipped. Once
all folds are done, --finalize writes the cross-fold files.

Usage (from this folder):
    python train_cv.py                    # all folds, then finalize
    python train_cv.py --folds 0          # one fold
    python train_cv.py --finalize         # cross-fold files only
"""

import argparse
import gc
import inspect
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import common  # noqa: E402

MODEL_NAME = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"
MAX_LENGTH, LR = 256, 2e-5
EPOCHS = 10
BATCH_TRAIN, BATCH_EVAL = 16, 32


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=HERE / "data", help="Output folder (default: ./data)")
    ap.add_argument("--pairs_dir", type=Path, default=common.PAIRS_DIR,
                    help="Folder with the single NLI model's fold_*_human_vs_model.csv (data and folds)")
    ap.add_argument("--ckpt_dir", type=Path, default=None,
                    help="Checkpoints during training, deleted after each fold (default: <out>/ckpt; "
                         "a node-local folder such as $TMPDIR is faster on a cluster)")
    ap.add_argument("--folds", type=int, nargs="+", default=list(range(common.K_FOLDS)))
    ap.add_argument("--finalize", action="store_true", help="Only write the cross-fold files")
    ap.add_argument("--model", default=MODEL_NAME, help="Hugging Face id or local folder of the starting model")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--lr", type=float, default=LR)
    ap.add_argument("--max_length", type=int, default=MAX_LENGTH)
    ap.add_argument("--b_train", type=int, default=BATCH_TRAIN)
    ap.add_argument("--b_eval", type=int, default=BATCH_EVAL)
    ap.add_argument("--seed", type=int, default=common.SEED)
    ap.add_argument("--bf16", action="store_true", help="bf16 mixed precision (Ampere GPUs or newer); "
                                                        "off by default, as in the NLI notebooks")
    return ap.parse_args()


def sentence_table(pairs: pd.DataFrame, labels):
    """One row per sentence: text and the 0/1 target of each label (1 = present)."""
    sent = pairs.drop_duplicates("sentence_id").set_index("sentence_id")["premise"].sort_index()
    y = (pairs.assign(present=(pairs["nli_label"] == 0).astype(np.float32))
         .pivot(index="sentence_id", columns="hypothesis_label", values="present")
         .loc[sent.index, labels])
    assert not y.isna().any().any(), "missing (sentence, label) pairs"
    return sent, y


def train_fold(fold, args, pairs, labels, fold_map):
    import torch
    from transformers import (AutoConfig, AutoModelForSequenceClassification, AutoTokenizer,
                              DataCollatorWithPadding, Trainer, TrainingArguments, set_seed)

    set_seed(args.seed)
    print(f"\n========== Fold {fold + 1}/{common.K_FOLDS} | device: "
          f"{torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU (slow!)'} ==========")
    sent, y = sentence_table(pairs, labels)
    test_ids = set(fold_map.loc[fold_map.fold == fold, "sentence_id"])
    is_test = sent.index.isin(test_ids)
    print(f"train sentences: {(~is_test).sum():,} | test sentences: {is_test.sum():,}")

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)

    class SentenceDataset(torch.utils.data.Dataset):
        def __init__(self, texts, targets):
            self.texts, self.targets = list(texts), targets
        def __len__(self):
            return len(self.texts)
        def __getitem__(self, i):
            enc = tokenizer(self.texts[i], truncation=True, max_length=args.max_length)
            enc["labels"] = self.targets[i].tolist()   # float targets: BCE loss
            return enc

    ds_train = SentenceDataset(sent[~is_test].values, y[~is_test].to_numpy())
    ds_test = SentenceDataset(sent[is_test].values, y[is_test].to_numpy())

    config = AutoConfig.from_pretrained(
        args.model, num_labels=len(labels), problem_type="multi_label_classification",
        id2label=dict(enumerate(labels)), label2id={l: i for i, l in enumerate(labels)})
    model = AutoModelForSequenceClassification.from_pretrained(args.model, config=config, ignore_mismatched_sizes=True)

    def to_nli_coding(targets):  # 1 = present -> nli_label 0 (entailment)
        return np.where(np.asarray(targets).ravel() > 0.5, 0, 1)

    def compute_metrics_fn(p):
        logits = p.predictions[0] if isinstance(p.predictions, (tuple, list)) else p.predictions
        probs = 1 / (1 + np.exp(-logits))
        return common.compute_all_metrics(to_nli_coding(p.label_ids), probs.ravel())

    ckpt_dir = (args.ckpt_dir or args.out / "ckpt") / f"fold_{fold}"
    targs = TrainingArguments(
        output_dir=str(ckpt_dir), learning_rate=args.lr,
        per_device_train_batch_size=args.b_train, per_device_eval_batch_size=args.b_eval,
        num_train_epochs=args.epochs, eval_strategy="epoch", save_strategy="epoch", save_total_limit=1,
        load_best_model_at_end=True, metric_for_best_model="eval_accuracy",
        seed=args.seed, report_to=[], logging_steps=50, bf16=args.bf16, fp16=False)
    tok_kw = "processing_class" if "processing_class" in inspect.signature(Trainer.__init__).parameters else "tokenizer"
    trainer = Trainer(model=model, args=targs, train_dataset=ds_train, eval_dataset=ds_test,
                      data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
                      compute_metrics=compute_metrics_fn, **{tok_kw: tokenizer})
    trainer.train()

    logs = pd.DataFrame(trainer.state.log_history)
    logs = logs[logs["epoch"].notna()]

    def last_per_epoch(col, name):  # robust to a loss never being logged (fewer steps than logging_steps)
        if col not in logs:
            return pd.DataFrame(columns=["epoch", name])
        return logs[logs[col].notna()].groupby("epoch", as_index=False)[col].last().rename(columns={col: name})
    tr, ev = last_per_epoch("loss", "training_loss"), last_per_epoch("eval_loss", "validation_loss")
    pd.merge(tr, ev, on="epoch", how="outer").assign(fold=fold).to_csv(
        common.cv_dir(args.out) / f"fold_{fold}_epoch_losses.csv", index=False)

    pred = trainer.predict(ds_test)
    logits = pred.predictions[0] if isinstance(pred.predictions, (tuple, list)) else pred.predictions
    probs = pd.DataFrame(1 / (1 + np.exp(-logits)), index=sent.index[is_test], columns=labels)

    # one row per (sentence, label) pair, in the order of the test pairs
    test_pairs = pairs[pairs["sentence_id"].isin(test_ids)].reset_index(drop=True)
    prob_entail = probs.stack().rename("p").reindex(
        pd.MultiIndex.from_frame(test_pairs[["sentence_id", "hypothesis_label"]])).to_numpy()
    assert not np.isnan(prob_entail).any()
    best_epoch = None
    if trainer.state.best_model_checkpoint:
        best = [h for h in trainer.state.log_history
                if h.get("eval_accuracy") == trainer.state.best_metric and "epoch" in h]
        best_epoch = best[0]["epoch"] if best else None
    common.write_fold_outputs(args.out, fold, test_pairs, prob_entail,
                              extra={"best_epoch": best_epoch, "model": args.model, "formulation": "multi-label classifier"})

    del trainer, model
    shutil.rmtree(ckpt_dir, ignore_errors=True)
    try:
        ckpt_dir.parent.rmdir()   # only if empty (other folds may still be running)
    except OSError:
        pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    pairs, reference = common.load_pairs(args.pairs_dir)
    labels = common.labels_sorted(pairs)
    fold_map = common.get_fold_map(pairs, reference, args.out)
    (args.out / "run_config.json").write_text(json.dumps(
        {k: str(v) for k, v in vars(args).items()}, indent=2), encoding="utf-8")

    if not args.finalize:
        for fold in args.folds:
            if common.fold_done(args.out, fold):
                print(f"Fold {fold}: already done, skipping")
                continue
            train_fold(fold, args, pairs, labels, fold_map)
    if all(common.fold_done(args.out, k) for k in range(common.K_FOLDS)):
        common.finalize(args.out, pairs, fold_map)
    else:
        print("Not all folds are done yet; run with --finalize once they are.")


if __name__ == "__main__":
    main()
