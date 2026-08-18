"""
main.py
=======
Integrated Etap 3 pipeline — Ames Housing LLM commentary generator.

Workflow
--------
1. Load or rebuild the analytical table from data/raw/train.csv
2. Run Q1-Q3 analyses (analysis.py)
3. Build prompts for FLAN-T5 (prompt_builder.py)
4. Generate analytical comments using the fine-tuned model (llm_model.py)
5. Save all outputs to outputs/

Model priority
--------------
  1. --checkpoint <path>          explicitly given checkpoint
  2. flan_t5_ames/final/          auto-detected fine-tuned model (default)
  3. google/flan-t5-base          fallback when no fine-tuned model is found

Usage
-----
  # Standard run — uses fine-tuned model from flan_t5_ames/final/ (after training)
  python main.py

  # Explicit checkpoint
  python main.py --checkpoint flan_t5_ames/final

  # Force base model (skip fine-tuned checkpoint)
  python main.py --base-model

  # Skip LLM entirely — only build and save prompts
  python main.py --no-llm

  # GPU inference
  python main.py --device cuda

Fine-tuning workflow (run once before main.py)
----------------------------------------------
  python finetune/prepare_dataset.py   # generate train.json + val.json
  python finetune/train.py             # fine-tune → saves to flan_t5_ames/final/
  python main.py                       # pipeline now uses the fine-tuned model
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from analysis import run_all
from dataset import (
    build_report_dataframe,
    ensure_openml_parquet_fallback,
    load_raw,
    save_csv,
    save_sqlite,
)
from prompt_builder import build_all_prompts

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────
BASE       = Path(__file__).resolve().parent
DATA       = BASE / "data"
RAW        = DATA / "raw"
REPORT_CSV = DATA / "ames_report.csv"
DB_PATH    = DATA / "ames.db"
OUTPUT_DIR = BASE / "outputs"

# Default fine-tuned checkpoint produced by finetune/train.py
DEFAULT_CHECKPOINT = BASE / "flan_t5_ames" / "final"
BASE_MODEL_ID      = "google/flan-t5-base"


# ══════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Ames Housing — Etap 3 pipeline with fine-tuned FLAN-T5",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--checkpoint",
        default=None,
        metavar="PATH",
        help=(
            "Path to a fine-tuned model directory. "
            f"Defaults to '{DEFAULT_CHECKPOINT}' when it exists, "
            f"otherwise falls back to '{BASE_MODEL_ID}'."
        ),
    )
    p.add_argument(
        "--base-model",
        action="store_true",
        help=f"Force use of '{BASE_MODEL_ID}' even if a fine-tuned checkpoint exists.",
    )
    p.add_argument(
        "--no-llm",
        action="store_true",
        help="Skip LLM generation entirely — only run analyses and save prompts.",
    )
    p.add_argument(
        "--device",
        default=None,
        choices=["cpu", "cuda"],
        help="Device for LLM inference (default: auto-detect).",
    )
    p.add_argument(
        "--output-dir",
        default=str(OUTPUT_DIR),
        metavar="DIR",
        help="Directory for all output files (default: outputs/).",
    )
    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════
# Checkpoint resolution
# ══════════════════════════════════════════════════════════════════════

def resolve_checkpoint(args: argparse.Namespace) -> str | None:
    """
    Determine which model to load, in priority order:

      1. --base-model flag  → always return None  (AmesFlanT5 uses base model)
      2. --checkpoint PATH  → use that path
      3. flan_t5_ames/final exists  → use the fine-tuned model (default)
      4. nothing found  → return None  (AmesFlanT5 falls back to base model)
    """
    if args.base_model:
        logger.info("--base-model flag set: using '%s'", BASE_MODEL_ID)
        return None

    if args.checkpoint is not None:
        cp = Path(args.checkpoint)
        if not cp.exists():
            logger.error(
                "Checkpoint directory not found: %s\n"
                "Run  python finetune/train.py  first, or omit --checkpoint.",
                cp,
            )
            sys.exit(1)
        logger.info("Using explicitly provided checkpoint: %s", cp)
        return str(cp)

    # Auto-detect default fine-tuned checkpoint
    if DEFAULT_CHECKPOINT.exists() and any(DEFAULT_CHECKPOINT.iterdir()):
        logger.info(
            "Fine-tuned checkpoint found at '%s' — using it for inference.",
            DEFAULT_CHECKPOINT,
        )
        return str(DEFAULT_CHECKPOINT)

    # No fine-tuned model available
    logger.warning(
        "No fine-tuned checkpoint found at '%s'.\n"
        "  → Falling back to base model '%s'.\n"
        "  → To use the fine-tuned model, run:\n"
        "       python finetune/prepare_dataset.py\n"
        "       python finetune/train.py",
        DEFAULT_CHECKPOINT,
        BASE_MODEL_ID,
    )
    return None


# ══════════════════════════════════════════════════════════════════════
# Data loading
# ══════════════════════════════════════════════════════════════════════

def build_or_load_report() -> pd.DataFrame:
    """
    Rebuild the analytical table when raw data is available;
    otherwise load the pre-built ames_report.csv so the project
    remains runnable without re-downloading anything.
    """
    raw_train = RAW / "train.csv"
    if raw_train.exists() and raw_train.stat().st_size > 0:
        logger.info("Building report from %s …", raw_train)
        raw = load_raw(raw_train)
        report_df = build_report_dataframe(raw)
        save_csv(report_df, REPORT_CSV)
        save_sqlite(report_df, DB_PATH)
        return report_df

    if REPORT_CSV.exists() and REPORT_CSV.stat().st_size > 0:
        logger.info("Loading pre-built report from %s …", REPORT_CSV)
        return pd.read_csv(REPORT_CSV)

    logger.info("Downloading data from OpenML …")
    train_path = ensure_openml_parquet_fallback(RAW)
    raw = load_raw(train_path)
    report_df = build_report_dataframe(raw)
    save_csv(report_df, REPORT_CSV)
    save_sqlite(report_df, DB_PATH)
    return report_df


# ══════════════════════════════════════════════════════════════════════
# Output helpers
# ══════════════════════════════════════════════════════════════════════

def _json_safe(obj: Any) -> Any:
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj


def save_outputs(
    results: dict,
    prompts: dict[str, str],
    comments: dict[str, str] | None,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    results["q1_parking_table"].to_csv(output_dir / "q1_parking_table.csv", index=False)
    results["q1_rail_table"].to_csv(output_dir / "q1_rail_table.csv",    index=False)
    results["q3_fisher"].to_csv(output_dir / "q3_fisher.csv",            index=False)

    (output_dir / "q2_torch.json").write_text(
        json.dumps(results["q2_torch"], indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output_dir / "analysis_results.json").write_text(
        json.dumps(_json_safe(results), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output_dir / "prompts.json").write_text(
        json.dumps(prompts, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    if comments:
        (output_dir / "llm_comments.json").write_text(
            json.dumps(comments, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        logger.info("LLM comments saved to %s/llm_comments.json", output_dir)

    logger.info("All outputs saved to %s", output_dir)


# ══════════════════════════════════════════════════════════════════════
# LLM inference
# ══════════════════════════════════════════════════════════════════════

def generate_comments(
    prompts: dict[str, str],
    checkpoint: str | None,
    device: str | None,
) -> dict[str, str]:
    """Load the model and generate one comment per prompt."""
    from llm_model import AmesFlanT5

    model_label = checkpoint if checkpoint else BASE_MODEL_ID
    logger.info("Loading model: %s", model_label)

    model = AmesFlanT5(checkpoint=checkpoint, device=device)

    keys   = list(prompts.keys())
    values = [prompts[k] for k in keys]

    logger.info("Generating %d comment(s) …", len(values))
    generations = model.generate_batch(values)

    return dict(zip(keys, generations))


# ══════════════════════════════════════════════════════════════════════
# Summary print
# ══════════════════════════════════════════════════════════════════════

def print_summary(
    results: dict,
    comments: dict[str, str] | None,
    checkpoint: str | None,
) -> None:
    q2  = results["q2_torch"]
    q3  = results["q3_fisher"]

    print("\n" + "=" * 60)
    print("  ETAP 3 — PIPELINE SUMMARY")
    print("=" * 60)
    print(f"  Q1 parking rows : {len(results['q1_parking_table'])}")
    print(f"  Q1 rail rows    : {len(results['q1_rail_table'])}")

    if "error" in q2:
        print(f"  Q2 model        : ERROR — {q2['error']}")
    else:
        print(
            f"  Q2 model        : train_rmse={q2['train_rmse']:.6f}"
            f"  val_rmse={q2['val_rmse']:.6f}"
        )

    if q3.empty:
        print("  Q3 Fisher       : no regions with enough observations")
    else:
        sig = int((q3["p_value_approx"] < 0.05).sum())
        print(f"  Q3 Fisher       : {sig}/{len(q3)} regions significant at p < 0.05")

    if comments:
        used_model = checkpoint if checkpoint else BASE_MODEL_ID
        print(f"\n  Model used      : {used_model}")
        print("\n" + "=" * 60)
        print("  GENERATED LLM COMMENTS")
        print("=" * 60)
        for key, text in comments.items():
            print(f"\n[{key}]\n{text}")
    else:
        print("\n  LLM skipped — prompts saved to outputs/prompts.json")
        print(
            "\n  To generate comments, run:\n"
            "    python finetune/prepare_dataset.py\n"
            "    python finetune/train.py\n"
            "    python main.py"
        )
    print("=" * 60)


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════

def main() -> None:
    args       = parse_args()
    output_dir = Path(args.output_dir)

    # ── Step 1: Data ──────────────────────────────────────────────────
    report_df = build_or_load_report()
    logger.info("Report: %d rows, %d columns", *report_df.shape)

    # ── Step 2: Analyses (Q1, Q2, Q3) ────────────────────────────────
    logger.info("Running Q1–Q3 analyses …")
    results = run_all(report_df)

    # ── Step 3: Build prompts ─────────────────────────────────────────
    prompts = build_all_prompts(results)
    logger.info("Built %d prompt(s).", len(prompts))

    # ── Step 4: LLM generation ────────────────────────────────────────
    comments: dict[str, str] | None = None

    if not args.no_llm:
        checkpoint = resolve_checkpoint(args)
        try:
            comments = generate_comments(prompts, checkpoint, args.device)
        except Exception as exc:
            logger.error(
                "LLM generation failed: %s\n"
                "Run with --no-llm to skip, or check that transformers/torch are installed.",
                exc,
            )
            raise

    # ── Step 5: Save everything ───────────────────────────────────────
    save_outputs(results, prompts, comments, output_dir)

    # ── Step 6: Print summary ─────────────────────────────────────────
    checkpoint_used = resolve_checkpoint(args) if not args.no_llm else None
    print_summary(results, comments, checkpoint_used)


if __name__ == "__main__":
    main()
