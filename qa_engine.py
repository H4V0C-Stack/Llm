"""
qa_engine.py
============
Question-Answering engine over the full ames.db database.

Architecture (two-step):
  Step 1 — Text-to-SQL:
      User question  →  FLAN-T5  →  SQL query
  Step 2 — SQL-to-Answer:
      SQL query  →  SQLite  →  result rows  →  FLAN-T5  →  natural language answer

This means the model can answer ANY question that can be expressed as a SQL
query over the 'sales' table — not just the slices used in prepare_dataset.py.

Standalone usage
----------------
    from qa_engine import AmesDatabaseQA
    qa = AmesDatabaseQA()                          # fine-tuned model (default)
    qa = AmesDatabaseQA(checkpoint="flan_t5_ames/final")
    print(qa.ask("Which region has the highest average price?"))
    print(qa.ask("How many houses were sold near the railroad?"))

Interactive CLI
---------------
    python qa_engine.py
    python qa_engine.py --checkpoint flan_t5_ames/final
    python qa_engine.py --base-model
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sqlite3
import textwrap
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────
BASE               = Path(__file__).resolve().parent
DB_PATH            = BASE / "data" / "ames.db"
DEFAULT_CHECKPOINT = BASE / "flan_t5_ames" / "final"
BASE_MODEL_ID      = "google/flan-t5-base"

# ── DB schema (injected into every SQL-generation prompt) ──────────────
_SCHEMA = """
Table: sales
Columns:
  Price           REAL     -- sale price in USD
  Sales           INTEGER  -- number of transactions in (Region, Year_Sold)
  Region          TEXT     -- neighbourhood name (25 unique values)
  Product_Category TEXT    -- 'house', 'apartment', or 'studio'
  Discount        TEXT     -- Sale_Condition|Sale_Type, e.g. 'Normal|WD'
  Discount_Flag   INTEGER  -- 1 if non-standard sale, 0 otherwise
  Price_Elasticity REAL    -- (Price - regional_median) / regional_median
  Sales_Change_Pct REAL    -- year-over-year % change in regional median price
  Year_Sold       INTEGER  -- year of sale (2006–2010)
  Garage_Cars     INTEGER  -- number of garage parking spots (0–4)
  Condition_1     TEXT     -- proximity flag: 'Norm','Feedr','Artery',
                           --   'RRAe','RRAn','RRNn','RRNe','PosA','PosN'
  Overall_Qual    INTEGER  -- overall build quality 1–10
""".strip()

# Values to help the model produce correct WHERE clauses
_SAMPLE_VALUES = """
Region examples    : 'NAmes', 'CollgCr', 'OldTown', 'Edwards', 'Somerst', 'NridgHt'
Product_Category   : 'house', 'apartment', 'studio'
Condition_1 near railroad: 'RRAe', 'RRAn', 'RRNn', 'RRNe'
Condition_1 normal : 'Norm'
Discount_Flag      : 1 = non-standard sale, 0 = normal sale
Year_Sold range    : 2006 – 2010
Overall_Qual range : 1 (worst) – 10 (best)
Price range        : ~35 000 – 755 000 USD
""".strip()

# Max rows returned from DB before summarising
_MAX_ROWS = 50
# Max chars of raw result injected into the answer prompt
_MAX_RESULT_CHARS = 600


# ══════════════════════════════════════════════════════════════════════
# Prompt builders
# ══════════════════════════════════════════════════════════════════════

# Few-shot examples shown to the model every time — teaches correct SQL pattern
_FEW_SHOT = """
Q: Which region has the highest average sale price?
A: SELECT Region, AVG(Price) as avg_price FROM sales GROUP BY Region ORDER BY avg_price DESC LIMIT 1;

Q: How many properties are near the railroad?
A: SELECT COUNT(*) FROM sales WHERE Condition_1 IN ('RRAe','RRAn','RRNn','RRNe');

Q: What is the average price for houses?
A: SELECT AVG(Price) FROM sales WHERE Product_Category = 'house';

Q: Which year had the most sales?
A: SELECT Year_Sold, COUNT(*) as cnt FROM sales GROUP BY Year_Sold ORDER BY cnt DESC LIMIT 1;

Q: What is the average price elasticity for apartments?
A: SELECT AVG(Price_Elasticity) FROM sales WHERE Product_Category = 'apartment';

Q: How many discount sales are there per category?
A: SELECT Product_Category, COUNT(*) FROM sales WHERE Discount_Flag = 1 GROUP BY Product_Category;

Q: List the top 5 most expensive regions by median price.
A: SELECT Region, AVG(Price) as avg_p FROM sales GROUP BY Region ORDER BY avg_p DESC LIMIT 5;

Q: How many properties have Overall_Qual >= 9?
A: SELECT COUNT(*) FROM sales WHERE Overall_Qual >= 9;
""".strip()


def _sql_prompt(question: str) -> str:
    return (
        "Translate the question to a SQLite SELECT query for the sales table.\n"
        "Output ONLY the SQL — no explanation, no markdown.\n\n"
        f"Schema:\n{_SCHEMA}\n\n"
        f"Examples:\n{_FEW_SHOT}\n\n"
        f"Q: {question}\n"
        "A:"
    )


def _answer_prompt(question: str, sql: str, result_text: str) -> str:
    return (
        "You are a real estate data analyst. "
        "Answer the question in 1–3 clear sentences using the query result below. "
        "Be specific — include numbers from the result. "
        "Do not mention SQL or technical details.\n\n"
        f"Question: {question}\n\n"
        f"SQL used: {sql}\n\n"
        f"Query result:\n{result_text}\n\n"
        "Answer:"
    )


def _error_answer_prompt(question: str, sql: str, error: str) -> str:
    return (
        "A SQL query was attempted but failed. "
        "Explain in 1–2 sentences what went wrong and suggest how the user "
        "could rephrase the question.\n\n"
        f"Original question: {question}\n"
        f"SQL attempted: {sql}\n"
        f"Error: {error}\n\n"
        "Explanation:"
    )


# ══════════════════════════════════════════════════════════════════════
# SQL extraction & cleaning
# ══════════════════════════════════════════════════════════════════════

def _extract_sql(raw: str) -> str:
    """
    Pull out the first SELECT … statement from model output.
    FLAN-T5 sometimes adds commentary before/after the query.
    """
    # Strip markdown code fences
    raw = re.sub(r"```(?:sql)?", "", raw, flags=re.IGNORECASE).replace("```", "")

    # Try to find a SELECT block
    match = re.search(r"(SELECT\b.*?)(?:;|$)", raw, re.IGNORECASE | re.DOTALL)
    if match:
        sql = match.group(1).strip()
        # Remove any trailing natural-language sentence the model appended
        sql = re.split(r"\n(?=[A-Z][a-z])", sql)[0].strip()
        return sql + ";"

    # Fallback: return everything (SQLite will report the error)
    return raw.strip()


def _is_safe_sql(sql: str) -> bool:
    """Block any write operations — only SELECT is allowed."""
    upper = sql.upper().lstrip()
    dangerous = ("INSERT", "UPDATE", "DELETE", "DROP", "ALTER",
                 "CREATE", "REPLACE", "TRUNCATE", "PRAGMA")
    return upper.startswith("SELECT") and not any(
        re.search(rf"\b{kw}\b", upper) for kw in dangerous
    )


# ══════════════════════════════════════════════════════════════════════
# Result formatting
# ══════════════════════════════════════════════════════════════════════

def _format_rows(rows: list[tuple], columns: list[str]) -> str:
    """Convert DB rows to a compact text representation."""
    if not rows:
        return "No rows returned."

    # Truncate to _MAX_ROWS
    truncated = len(rows) > _MAX_ROWS
    display   = rows[:_MAX_ROWS]

    lines = []
    for row in display:
        parts = []
        for col, val in zip(columns, row):
            if isinstance(val, float):
                parts.append(f"{col}={val:.4f}")
            else:
                parts.append(f"{col}={val}")
        lines.append("  " + ", ".join(parts))

    result = "\n".join(lines)
    if truncated:
        result += f"\n  … (showing {_MAX_ROWS} of {len(rows)} rows)"

    # Clip to _MAX_RESULT_CHARS so the answer prompt fits in 512 tokens
    if len(result) > _MAX_RESULT_CHARS:
        result = result[:_MAX_RESULT_CHARS] + "\n  … (truncated)"

    return result


# ══════════════════════════════════════════════════════════════════════
# DB execution
# ══════════════════════════════════════════════════════════════════════

def _run_sql(sql: str, db_path: Path) -> tuple[list[tuple], list[str], str | None]:
    """
    Execute sql against db_path.
    Returns (rows, column_names, error_message).
    error_message is None on success.
    """
    try:
        con = sqlite3.connect(str(db_path))
        cur = con.cursor()
        cur.execute(sql)
        rows    = cur.fetchall()
        columns = [desc[0] for desc in cur.description] if cur.description else []
        con.close()
        return rows, columns, None
    except sqlite3.Error as exc:
        return [], [], str(exc)


# ══════════════════════════════════════════════════════════════════════
# Main QA class
# ══════════════════════════════════════════════════════════════════════

class AmesDatabaseQA:
    """
    Two-step QA engine:
      question → SQL (via FLAN-T5) → execute on ames.db → answer (via FLAN-T5)

    Parameters
    ----------
    checkpoint : str | Path | None
        Path to a fine-tuned model directory.
        None  → use auto-detection logic (same as main.py):
                flan_t5_ames/final if it exists, otherwise google/flan-t5-base
    base_model : bool
        If True, always use google/flan-t5-base regardless of checkpoint.
    device : str | None
        'cpu', 'cuda', or None for auto-detect.
    db_path : Path | None
        Path to ames.db. Defaults to data/ames.db next to this file.
    """

    def __init__(
        self,
        checkpoint: str | Path | None = None,
        base_model: bool = False,
        device: str | None = None,
        db_path: Path | None = None,
    ) -> None:
        from llm_model import AmesFlanT5

        self.db_path = db_path or DB_PATH
        if not self.db_path.exists():
            raise FileNotFoundError(
                f"Database not found: {self.db_path}\n"
                "Run python main.py first to generate it."
            )

        # Resolve which model to load
        resolved = self._resolve_checkpoint(checkpoint, base_model)
        model_label = resolved if resolved else BASE_MODEL_ID
        logger.info("QA engine loading model: %s", model_label)

        self._model = AmesFlanT5(checkpoint=resolved, device=device)
        self.model_label = model_label

    # ── Checkpoint resolution (mirrors main.py logic) ──────────────────
    @staticmethod
    def _resolve_checkpoint(
        checkpoint: str | Path | None,
        base_model: bool,
    ) -> str | None:
        if base_model:
            return None
        if checkpoint is not None:
            cp = Path(checkpoint)
            if not cp.exists():
                raise FileNotFoundError(
                    f"Checkpoint not found: {cp}\n"
                    "Run python finetune/train.py first."
                )
            return str(cp)
        if DEFAULT_CHECKPOINT.exists() and any(DEFAULT_CHECKPOINT.iterdir()):
            return str(DEFAULT_CHECKPOINT)
        logger.warning(
            "No fine-tuned checkpoint at '%s'. Using base model '%s'.",
            DEFAULT_CHECKPOINT, BASE_MODEL_ID,
        )
        return None

    # ── Public interface ───────────────────────────────────────────────
    def ask(self, question: str) -> dict[str, Any]:
        """
        Answer a free-form question about the Ames Housing database.

        Returns a dict with keys:
          question  : original question
          sql       : generated SQL query
          rows      : raw DB result (list of dicts)
          answer    : natural language answer
          error     : error message string, or None
        """
        question = question.strip()
        if not question:
            return {"question": question, "sql": "", "rows": [],
                    "answer": "Please ask a question.", "error": None}

        # ── Step 1: generate SQL ───────────────────────────────────────
        sql_prompt = _sql_prompt(question)
        raw_sql    = self._model.generate(sql_prompt)
        sql        = _extract_sql(raw_sql)
        logger.debug("Generated SQL: %s", sql)

        # ── Safety check ───────────────────────────────────────────────
        if not _is_safe_sql(sql):
            return {
                "question": question,
                "sql":      sql,
                "rows":     [],
                "answer":   "I can only run SELECT queries on this database.",
                "error":    "unsafe SQL blocked",
            }

        # ── Step 2: execute SQL ────────────────────────────────────────
        rows, columns, db_error = _run_sql(sql, self.db_path)

        if db_error:
            # Let the model explain what went wrong
            err_prompt = _error_answer_prompt(question, sql, db_error)
            answer     = self._model.generate(err_prompt)
            return {
                "question": question,
                "sql":      sql,
                "rows":     [],
                "answer":   answer,
                "error":    db_error,
            }

        # ── Step 3: generate natural language answer ───────────────────
        result_text  = _format_rows(rows, columns)
        ans_prompt   = _answer_prompt(question, sql, result_text)
        answer       = self._model.generate(ans_prompt)

        # Build rows as list-of-dicts for JSON output
        rows_dicts = [dict(zip(columns, row)) for row in rows[:_MAX_ROWS]]

        return {
            "question": question,
            "sql":      sql,
            "rows":     rows_dicts,
            "answer":   answer,
            "error":    None,
        }

    # ── Convenience: ask multiple questions at once ────────────────────
    def ask_batch(self, questions: list[str]) -> list[dict[str, Any]]:
        return [self.ask(q) for q in questions]

    # ── Pretty print ──────────────────────────────────────────────────
    @staticmethod
    def pretty(result: dict[str, Any]) -> str:
        lines = [
            f"Question : {result['question']}",
            f"SQL      : {result['sql']}",
        ]
        if result.get("error"):
            lines.append(f"Error    : {result['error']}")
        else:
            lines.append(f"Rows     : {len(result['rows'])} returned")
        lines.append(f"Answer   : {result['answer']}")
        return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════
# Interactive CLI
# ══════════════════════════════════════════════════════════════════════

_EXAMPLE_QUESTIONS = [
    "Which region has the highest average sale price?",
    "How many houses were sold near the railroad?",
    "What is the average price elasticity for apartments?",
    "Which year had the most sales overall?",
    "List the top 5 most expensive regions by median price.",
    "How many properties have Overall_Qual >= 9?",
    "What is the average Sales_Change_Pct for houses with 3 or more garage spots?",
    "Which regions have a negative average Sales_Change_Pct?",
    "How many discount sales (Discount_Flag=1) are there per Product_Category?",
    "What is the min, max, and average price in StoneBr?",
]


def _print_welcome(model_label: str) -> None:
    print("\n" + "=" * 62)
    print("  Ames Housing — Database Q&A")
    print(f"  Model : {model_label}")
    print(f"  DB    : {DB_PATH}")
    print("=" * 62)
    print("  Ask any question about the Ames Housing database.")
    print("  Type  'examples'  to see sample questions.")
    print("  Type  'schema'    to see the database schema.")
    print("  Type  'quit'  or  'exit'  to leave.")
    print("=" * 62 + "\n")


def _cli(args: argparse.Namespace) -> None:
    logging.basicConfig(
        level=logging.WARNING,   # keep output clean in interactive mode
        format="%(levelname)s: %(message)s",
    )

    print("Loading model … (this may take a moment on first run)")
    qa = AmesDatabaseQA(
        checkpoint=args.checkpoint or None,
        base_model=args.base_model,
        device=args.device,
    )
    _print_welcome(qa.model_label)

    # Optional: save session log
    session_log: list[dict] = []
    log_path = Path(args.output) if args.output else None

    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not question:
            continue

        if question.lower() in ("quit", "exit", "q"):
            print("Goodbye.")
            break

        if question.lower() == "schema":
            print("\n" + _SCHEMA + "\n")
            print(_SAMPLE_VALUES + "\n")
            continue

        if question.lower() == "examples":
            print("\nExample questions:")
            for i, q in enumerate(_EXAMPLE_QUESTIONS, 1):
                print(f"  {i:2d}. {q}")
            print()
            continue

        # Try to run a numbered example
        if question.isdigit():
            idx = int(question) - 1
            if 0 <= idx < len(_EXAMPLE_QUESTIONS):
                question = _EXAMPLE_QUESTIONS[idx]
                print(f"You (example {idx+1}): {question}")
            else:
                print(f"  No example #{question}. Type 'examples' to list them.")
                continue

        result = qa.ask(question)
        print()
        print(qa.pretty(result))
        print()

        session_log.append(result)

        if log_path:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_text(
                json.dumps(session_log, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )


# ══════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Interactive Q&A over the full Ames Housing database",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
        Examples:
          python qa_engine.py
          python qa_engine.py --base-model
          python qa_engine.py --checkpoint flan_t5_ames/final
          python qa_engine.py --output outputs/qa_session.json
        """),
    )
    p.add_argument("--checkpoint", default=None, metavar="PATH",
                   help="Fine-tuned model directory (default: flan_t5_ames/final if it exists)")
    p.add_argument("--base-model", action="store_true",
                   help=f"Force use of '{BASE_MODEL_ID}'")
    p.add_argument("--device", default=None, choices=["cpu", "cuda"],
                   help="Inference device (default: auto)")
    p.add_argument("--output", default=None, metavar="FILE",
                   help="Save Q&A session log to this JSON file")
    return p.parse_args()


if __name__ == "__main__":
    _cli(_parse_args())
