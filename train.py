"""
finetune/train.py
=================
Fine-tunes google/flan-t5-base on the Ames Housing commentary dataset.

Key fixes vs previous version:
  - fp16 disabled by default (caused loss=0, grad_norm=nan on T5 in Colab)
  - early stopping disabled by default (was stopping at 50% due to flat metric)
  - gradient clipping added (max_grad_norm=1.0) to stabilise training
  - metric_for_best_model switched to "eval_loss" — always available, no NaN
  - eval frequency reduced to once per epoch (simpler, more stable)
  - lr lowered to 1e-4 (5e-4 was too aggressive for T5 seq2seq)

Usage
-----
  # Full-database fine-tuning (recommended):
  python finetune/prepare_dataset_full.py
  python finetune/train.py --train finetune/train_full.json \\
                            --val   finetune/val_full.json

  # Small dataset (analysis results only):
  python finetune/train.py

  # Custom params:
  python finetune/train.py --epochs 5 --lr 5e-5 --batch 8

Output
------
  flan_t5_ames/final/   ← ready to use in main.py and qa_engine.py
"""
from __future__ import annotations

import argparse
import inspect
import json
import logging
import sys
from pathlib import Path

import numpy as np

try:
    from datasets import Dataset, DatasetDict
    from transformers import (
        AutoModelForSeq2SeqLM,
        AutoTokenizer,
        DataCollatorForSeq2Seq,
        EarlyStoppingCallback,
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
        set_seed,
    )
except ImportError as exc:
    raise ImportError(
        "Install fine-tuning dependencies: pip install -r requirements.txt"
    ) from exc

try:
    import evaluate
    _ROUGE = evaluate.load("rouge")
    HAS_EVALUATE = True
except Exception:
    _ROUGE = None
    HAS_EVALUATE = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

ROOT           = Path(__file__).resolve().parents[1]
FT_DIR         = Path(__file__).resolve().parent
TRAIN_JSON     = FT_DIR / "train.json"
VAL_JSON       = FT_DIR / "val.json"
DEFAULT_OUTPUT = ROOT / "flan_t5_ames"

MODEL_ID       = "google/flan-t5-base"
MAX_INPUT_LEN  = 256
MAX_TARGET_LEN = 150


# ══════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fine-tune FLAN-T5-base on Ames Housing prompts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python finetune/train.py\n"
            "  python finetune/train.py --train finetune/train_full.json \\\n"
            "                           --val   finetune/val_full.json\n"
            "  python finetune/train.py --epochs 5 --lr 5e-5\n"
        ),
    )
    p.add_argument("--model",   default=MODEL_ID)
    p.add_argument("--epochs",  type=int,   default=3,
                   help="Number of training epochs (default: 3)")
    p.add_argument("--batch",   type=int,   default=8,
                   help="Per-device batch size (default: 8)")
    p.add_argument("--lr",      type=float, default=1e-4,
                   help="Learning rate (default: 1e-4). Use 5e-5 for more stable training.")
    p.add_argument("--warmup",  type=int,   default=50,
                   help="Warmup steps (default: 50)")
    p.add_argument("--seed",    type=int,   default=42)
    p.add_argument("--output",  default=str(DEFAULT_OUTPUT))
    p.add_argument(
        "--fp16", action="store_true",
        help=(
            "Enable fp16 mixed precision. "
            "WARNING: causes loss=0/grad_norm=nan on some T5+Colab setups. "
            "Only use if you verified it works on your GPU."
        ),
    )
    p.add_argument(
        "--early-stopping-patience", type=int, default=0,
        help=(
            "Stop early if metric doesn't improve for N evaluations. "
            "0 = disabled (default). Enable with e.g. --early-stopping-patience 3."
        ),
    )
    p.add_argument("--train", default=None, metavar="FILE",
                   help="Training JSON (default: finetune/train.json). "
                        "Use finetune/train_full.json for full-DB training.")
    p.add_argument("--val",   default=None, metavar="FILE",
                   help="Validation JSON (default: finetune/val.json). "
                        "Use finetune/val_full.json for full-DB training.")
    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════
# Data
# ══════════════════════════════════════════════════════════════════════

def load_json(path: Path) -> list[dict]:
    if not path.exists():
        logger.error("Missing %s. Run python finetune/prepare_dataset_full.py first.", path)
        sys.exit(1)
    data = json.loads(path.read_text(encoding="utf-8"))
    if not data:
        logger.error("Dataset is empty: %s", path)
        sys.exit(1)
    return data


def build_dataset(train: list[dict], val: list[dict]) -> DatasetDict:
    return DatasetDict({
        "train":      Dataset.from_list(train),
        "validation": Dataset.from_list(val),
    })


# ══════════════════════════════════════════════════════════════════════
# Tokenisation
# ══════════════════════════════════════════════════════════════════════

def make_tokenize_fn(tokenizer):
    def tokenize(batch: dict) -> dict:
        inputs = tokenizer(
            batch["prompt"],
            max_length=MAX_INPUT_LEN,
            padding="max_length",
            truncation=True,
        )
        labels = tokenizer(
            text_target=batch["target"],
            max_length=MAX_TARGET_LEN,
            padding="max_length",
            truncation=True,
        )
        # Replace pad token in labels with -100 so loss ignores padding
        inputs["labels"] = [
            [(t if t != tokenizer.pad_token_id else -100) for t in seq]
            for seq in labels["input_ids"]
        ]
        return inputs
    return tokenize


# ══════════════════════════════════════════════════════════════════════
# Metrics
# ══════════════════════════════════════════════════════════════════════

def make_compute_metrics_fn(tokenizer):
    def compute_metrics(eval_preds):
        predictions, labels = eval_preds
        if isinstance(predictions, tuple):
            predictions = predictions[0]

        predictions = np.clip(np.asarray(predictions), 0, tokenizer.vocab_size - 1)
        decoded_preds  = tokenizer.batch_decode(predictions, skip_special_tokens=True)

        labels = np.where(
            np.asarray(labels) != -100,
            np.asarray(labels),
            tokenizer.pad_token_id,
        )
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        decoded_preds  = [p.strip() for p in decoded_preds]
        decoded_labels = [l.strip() for l in decoded_labels]

        if HAS_EVALUATE and _ROUGE is not None:
            result = _ROUGE.compute(
                predictions=decoded_preds,
                references=decoded_labels,
                use_stemmer=True,
            )
            return {
                "rouge1": round(result["rouge1"], 4),
                "rougeL": round(result["rougeL"], 4),
            }

        # Fallback: token overlap
        scores = [
            len(set(p.lower().split()) & set(r.lower().split())) / max(1, len(set(r.lower().split())))
            for p, r in zip(decoded_preds, decoded_labels)
        ]
        return {"token_overlap": round(float(np.mean(scores)), 4)}

    return compute_metrics


# ══════════════════════════════════════════════════════════════════════
# Training arguments
# ══════════════════════════════════════════════════════════════════════

def build_training_args(
    args: argparse.Namespace,
    output_dir: Path,
) -> Seq2SeqTrainingArguments:
    """
    Evaluate once per epoch.
    metric_for_best_model = eval_loss (always finite, unlike ROUGE/token_overlap
    which can be NaN when generation fails with fp16).
    """
    kwargs = dict(
        output_dir=str(output_dir),

        # Schedule
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        learning_rate=args.lr,
        warmup_steps=args.warmup,
        weight_decay=0.01,

        # Gradient clipping — prevents nan gradients from exploding
        max_grad_norm=1.0,

        # Mixed precision — OFF by default, user must explicitly request
        fp16=args.fp16,

        # Eval & save once per epoch — simpler and more stable
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,

        # eval_loss is always available and finite — safe metric for best model
        metric_for_best_model="eval_loss",
        greater_is_better=False,

        # Generation
        predict_with_generate=True,
        generation_max_length=MAX_TARGET_LEN,

        # Logging
        logging_steps=20,
        report_to="none",
        seed=args.seed,
    )

    # Handle eval_strategy vs evaluation_strategy depending on transformers version
    sig = inspect.signature(Seq2SeqTrainingArguments.__init__)
    if "eval_strategy" in sig.parameters:
        kwargs["eval_strategy"] = "epoch"
    else:
        kwargs["evaluation_strategy"] = "epoch"

    return Seq2SeqTrainingArguments(**kwargs)


# ══════════════════════════════════════════════════════════════════════
# Trainer factory
# ══════════════════════════════════════════════════════════════════════

def build_trainer(
    model,
    tokenizer,
    training_args,
    tokenised,
    data_collator,
    compute_metrics,
    callbacks,
) -> Seq2SeqTrainer:
    kwargs = dict(
        model=model,
        args=training_args,
        train_dataset=tokenised["train"],
        eval_dataset=tokenised["validation"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
    )
    sig = inspect.signature(Seq2SeqTrainer.__init__)
    if "processing_class" in sig.parameters:
        kwargs["processing_class"] = tokenizer
    else:
        kwargs["tokenizer"] = tokenizer
    return Seq2SeqTrainer(**kwargs)


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════

def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output)
    final_dir  = output_dir / "final"
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Data ──────────────────────────────────────────────────────────
    train_path = Path(args.train) if args.train else TRAIN_JSON
    val_path   = Path(args.val)   if args.val   else VAL_JSON
    logger.info("Train : %s", train_path)
    logger.info("Val   : %s", val_path)

    train_data = load_json(train_path)
    val_data   = load_json(val_path)
    dataset    = build_dataset(train_data, val_data)

    logger.info(
        "Examples — train: %d | val: %d",
        len(train_data), len(val_data),
    )

    # ── Model ─────────────────────────────────────────────────────────
    logger.info("Loading model: %s", args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model     = AutoModelForSeq2SeqLM.from_pretrained(args.model)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Trainable parameters: %.0fM", n_params / 1e6)

    # Warn explicitly if fp16 requested — known to cause nan on some setups
    if args.fp16:
        logger.warning(
            "fp16=True requested. If you see loss=0 or grad_norm=nan, "
            "re-run WITHOUT --fp16 (CPU/FP32 training is slower but stable)."
        )

    # ── Tokenise ──────────────────────────────────────────────────────
    logger.info("Tokenising …")
    tokenised = dataset.map(
        make_tokenize_fn(tokenizer),
        batched=True,
        remove_columns=["prompt", "target"],
        desc="Tokenising",
    )

    # ── Trainer ───────────────────────────────────────────────────────
    training_args = build_training_args(args, output_dir)
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=-100,
        pad_to_multiple_of=8 if args.fp16 else None,
    )

    callbacks = []
    if args.early_stopping_patience > 0:
        logger.info("Early stopping: patience=%d", args.early_stopping_patience)
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=args.early_stopping_patience
            )
        )

    trainer = build_trainer(
        model=model,
        tokenizer=tokenizer,
        training_args=training_args,
        tokenised=tokenised,
        data_collator=data_collator,
        compute_metrics=make_compute_metrics_fn(tokenizer),
        callbacks=callbacks,
    )

    # ── Train ─────────────────────────────────────────────────────────
    logger.info("=" * 56)
    logger.info("Starting fine-tuning")
    logger.info("  Epochs     : %d", args.epochs)
    logger.info("  Batch size : %d", args.batch)
    logger.info("  LR         : %s", args.lr)
    logger.info("  fp16       : %s", args.fp16)
    logger.info("  Output     : %s", output_dir)
    logger.info("=" * 56)

    train_result = trainer.train()

    # ── Save ──────────────────────────────────────────────────────────
    logger.info("Saving final model → %s", final_dir)
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)

    # Final eval
    eval_metrics = trainer.evaluate(
        eval_dataset=tokenised["validation"],
        metric_key_prefix="eval",
    )
    trainer.log_metrics("eval", eval_metrics)
    trainer.save_metrics("eval", eval_metrics)

    logger.info("=" * 56)
    logger.info("Fine-tuning complete.")
    logger.info("  train_loss : %.4f", train_result.metrics.get("train_loss", float("nan")))
    logger.info("  eval_loss  : %.4f", eval_metrics.get("eval_loss", float("nan")))
    if HAS_EVALUATE:
        logger.info("  ROUGE-L    : %.4f", eval_metrics.get("eval_rougeL", float("nan")))
    logger.info("  Checkpoint : %s", final_dir)
    logger.info("=" * 56)
    logger.info("Next step: python main.py")


if __name__ == "__main__":
    main()
