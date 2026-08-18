"""
finetune/prepare_dataset_full.py
================================
Generates a fine-tuning dataset from the FULL ames.db database (all 1460 rows).

Unlike prepare_dataset.py (which uses only aggregated analysis results),
this script creates training examples directly from every record in the
'sales' table — covering all 12 columns, all 25 regions, and all question
types the Q&A engine may encounter.

Strategy
--------
For every row we generate several (prompt, target) pairs by:
  1. Row-level questions  — describe a single property transaction
  2. Regional aggregates  — one example per (region, question_type) combination
  3. SQL question pairs   — teach the model to translate NL → SQL for this schema
  4. Comparative questions — across years, categories, quality segments

Output
------
    finetune/train_full.json   (~3 000 – 5 000 examples)
    finetune/val_full.json     (~15% held-out)

Usage
-----
    python finetune/prepare_dataset_full.py

Then fine-tune with:
    python finetune/train.py --train finetune/train_full.json \\
                              --val   finetune/val_full.json
"""
from __future__ import annotations

import json
import math
import random
import sqlite3
import sys
from pathlib import Path
from typing import Any

ROOT    = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "ames.db"
OUT_DIR = Path(__file__).resolve().parent

SEED        = 42
TRAIN_RATIO = 0.85

# Rail-proximity codes used in the dataset
RAIL_CODES = {"RRAe", "RRAn", "RRNe", "RRNn"}


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

def _pct(v: float | None) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "n/a"
    return f"{v * 100:.2f}%"


def _usd(v: float) -> str:
    return f"${v:,.0f}"


def _sign(v: float) -> str:
    return "above" if v > 0 else "below"


def _magnitude(v: float) -> str:
    a = abs(v)
    if a < 0.03:
        return "very slightly"
    if a < 0.08:
        return "slightly"
    if a < 0.18:
        return "moderately"
    return "significantly"


def _rail(code: str) -> bool:
    return code in RAIL_CODES


def _qual_label(q: int) -> str:
    if q >= 9:
        return "excellent"
    if q >= 7:
        return "good"
    if q >= 5:
        return "average"
    if q >= 3:
        return "below-average"
    return "poor"


def _discount_text(flag: int, discount: str) -> str:
    if flag == 0:
        return "a standard market sale"
    cond, stype = (discount.split("|") + ["WD"])[:2]
    labels = {
        "Abnorml": "an abnormal sale",
        "AdjLand": "an adjoining land purchase",
        "Alloca":  "an allocation sale",
        "Family":  "a family sale",
        "Partial": "a partial sale",
        "COD":     "a cash-on-delivery transaction",
        "ConLI":   "a contract with land included",
        "ConLD":   "a contract with low down payment",
        "ConLw":   "a contract with low interest",
        "CWD":     "a warranty deed with cash",
    }
    return labels.get(cond, labels.get(stype, "a non-standard sale"))


# ──────────────────────────────────────────────────────────────────────
# 1. Row-level examples  (one property = several Q&A pairs)
# ──────────────────────────────────────────────────────────────────────

def _row_examples(row: dict[str, Any]) -> list[dict]:
    r      = row["Region"]
    price  = row["Price"]
    cat    = row["Product_Category"]
    year   = row["Year_Sold"]
    qual   = row["Overall_Qual"]
    garage = row["Garage_Cars"]
    elast  = row["Price_Elasticity"]
    chg    = row["Sales_Change_Pct"]
    near_r = _rail(row["Condition_1"])
    disc   = _discount_text(row["Discount_Flag"], row["Discount"])
    sales  = row["Sales"]
    ql     = _qual_label(qual)

    examples = []

    # ── 1a: price description ──────────────────────────────────────────
    examples.append({
        "prompt": (
            f"Describe this property transaction: "
            f"Region={r}, Price={_usd(price)}, Product_Category={cat}, "
            f"Year_Sold={year}, Overall_Qual={qual}, Garage_Cars={garage}, "
            f"Discount={disc}."
        ),
        "target": (
            f"A {cat} in {r} sold for {_usd(price)} in {year}. "
            f"The property has {ql} overall quality (score {qual}/10) "
            f"and {garage} garage parking spot{'s' if garage != 1 else ''}. "
            f"It was {disc}."
        ),
    })

    # ── 1b: elasticity interpretation ─────────────────────────────────
    if elast is not None and not math.isnan(elast):
        examples.append({
            "prompt": (
                f"Interpret the price elasticity for a {cat} in {r}: "
                f"Price_Elasticity={elast:.4f}. "
                f"The regional median is the reference point."
            ),
            "target": (
                f"This {cat} sold {_magnitude(elast)} {_sign(elast)} "
                f"the regional median price in {r} "
                f"(price elasticity={elast:.4f}, i.e. {elast*100:+.2f}% deviation). "
                + (
                    "This suggests the property commanded a premium over comparable homes."
                    if elast > 0.05
                    else "This suggests the property was priced at or near the typical market level."
                    if abs(elast) <= 0.05
                    else "This suggests the property sold at a discount relative to the neighborhood median."
                )
            ),
        })

    # ── 1c: railroad proximity ─────────────────────────────────────────
    examples.append({
        "prompt": (
            f"Does this property have railroad proximity? "
            f"Condition_1={row['Condition_1']}, Region={r}, Price={_usd(price)}."
        ),
        "target": (
            f"Yes, this property in {r} is located near a railroad "
            f"(Condition_1={row['Condition_1']}). "
            f"Railroad proximity is generally associated with lower price elasticity "
            f"and can negatively affect perceived property value."
            if near_r else
            f"No, this property in {r} is not near a railroad "
            f"(Condition_1={row['Condition_1']}). "
            f"It sold for {_usd(price)} in {year}."
        ),
    })

    # ── 1d: year-over-year price change ───────────────────────────────
    if chg is not None and not math.isnan(chg):
        direction = "increased" if chg > 0 else "decreased"
        examples.append({
            "prompt": (
                f"What does a Sales_Change_Pct of {chg:.4f} mean "
                f"for the {r} neighborhood in {year}?"
            ),
            "target": (
                f"In {r}, the regional median sale price {direction} by {_pct(chg)} "
                f"year-over-year up to {year}. "
                + (
                    f"This positive trend suggests growing demand in the neighborhood."
                    if chg > 0.05
                    else f"This slight decline may indicate temporary market softening."
                    if chg < -0.05
                    else f"Prices remained relatively stable in this period."
                )
            ),
        })

    # ── 1e: market volume ─────────────────────────────────────────────
    examples.append({
        "prompt": (
            f"How active was the {r} market in {year}? "
            f"Sales count for (Region={r}, Year_Sold={year}) is {sales}."
        ),
        "target": (
            f"In {year}, {sales} transaction{'s were' if sales != 1 else ' was'} "
            f"recorded in {r}. "
            + (
                "This is a high-volume market, indicating strong buyer activity."
                if sales >= 50
                else "This is a moderately active market."
                if sales >= 20
                else "This is a low-volume market, which may limit price comparisons."
            )
        ),
    })

    return examples


# ──────────────────────────────────────────────────────────────────────
# 2. Regional aggregate examples  (computed from full DB)
# ──────────────────────────────────────────────────────────────────────

def _regional_examples(con: sqlite3.Connection) -> list[dict]:
    examples = []
    cur = con.cursor()

    # ── 2a: avg price per region ───────────────────────────────────────
    cur.execute("""
        SELECT Region, AVG(Price) as avg_p, MIN(Price) as min_p,
               MAX(Price) as max_p, COUNT(*) as cnt
        FROM sales GROUP BY Region ORDER BY avg_p DESC
    """)
    rows = cur.fetchall()
    top3 = rows[:3]
    lines = [f"{r}: avg={_usd(a)}, range={_usd(mn)}–{_usd(mx)}, n={c}"
             for r, a, mn, mx, c in top3]
    examples.append({
        "prompt": "Which regions have the highest average property prices?",
        "target": (
            f"The three most expensive neighborhoods in Ames are: "
            + "; ".join(lines) + ". "
            "These premium areas command significantly higher prices than the city average."
        ),
    })

    # Bottom 3
    bot3 = rows[-3:]
    lines_b = [f"{r}: avg={_usd(a)}" for r, a, _, _, _ in reversed(bot3)]
    examples.append({
        "prompt": "Which neighborhoods have the lowest average sale prices?",
        "target": (
            "The most affordable neighborhoods are: "
            + "; ".join(lines_b) + ". "
            "These areas offer lower entry prices and may represent value opportunities."
        ),
    })

    # ── 2b: category breakdown ─────────────────────────────────────────
    cur.execute("""
        SELECT Product_Category, COUNT(*) as cnt,
               AVG(Price) as avg_p, AVG(Price_Elasticity) as avg_e
        FROM sales GROUP BY Product_Category ORDER BY avg_p DESC
    """)
    rows = cur.fetchall()
    lines = [f"{cat}: {cnt} sales, avg={_usd(avg)}, avg_elasticity={e:.3f}"
             for cat, cnt, avg, e in rows]
    examples.append({
        "prompt": "Compare average prices and price elasticity across property categories.",
        "target": (
            "Property category breakdown: " + "; ".join(lines) + ". "
            "Houses dominate the market by volume, while studios and apartments "
            "show distinct pricing and elasticity patterns."
        ),
    })

    # ── 2c: railroad effect ────────────────────────────────────────────
    cur.execute("""
        SELECT
            CASE WHEN Condition_1 IN ('RRAe','RRAn','RRNe','RRNn') THEN 'near railroad'
                 ELSE 'not near railroad' END as proximity,
            COUNT(*) as cnt,
            AVG(Price) as avg_p,
            AVG(Price_Elasticity) as avg_e
        FROM sales
        GROUP BY proximity
    """)
    rows = cur.fetchall()
    lines = [f"{prox}: {cnt} properties, avg price={_usd(avg)}, avg elasticity={e:.3f}"
             for prox, cnt, avg, e in rows]
    examples.append({
        "prompt": "How does railroad proximity affect property prices in Ames?",
        "target": (
            "Railroad proximity analysis: " + "; ".join(lines) + ". "
            "Properties near railroads tend to have lower price elasticity, "
            "suggesting reduced willingness to pay premium prices in those areas."
        ),
    })

    # ── 2d: quality segments ───────────────────────────────────────────
    cur.execute("""
        SELECT
            CASE WHEN Overall_Qual >= 7 THEN 'elite (7-10)'
                 WHEN Overall_Qual <= 4 THEN 'budget (1-4)'
                 ELSE 'mid-range (5-6)' END as segment,
            COUNT(*) as cnt,
            AVG(Price) as avg_p,
            AVG(Price_Elasticity) as avg_e
        FROM sales
        GROUP BY segment ORDER BY avg_p DESC
    """)
    rows = cur.fetchall()
    lines = [f"{seg}: {cnt} properties, avg={_usd(avg)}, elasticity={e:.3f}"
             for seg, cnt, avg, e in rows]
    examples.append({
        "prompt": "Compare elite, mid-range, and budget housing segments by price and elasticity.",
        "target": (
            "Quality segment comparison: " + "; ".join(lines) + ". "
            "Elite properties (Overall_Qual ≥ 7) command significantly higher prices "
            "and tend to show greater price elasticity than budget properties."
        ),
    })

    # ── 2e: year trends ────────────────────────────────────────────────
    cur.execute("""
        SELECT Year_Sold, COUNT(*) as cnt,
               AVG(Price) as avg_p, AVG(Sales_Change_Pct) as avg_chg
        FROM sales
        WHERE Sales_Change_Pct IS NOT NULL
        GROUP BY Year_Sold ORDER BY Year_Sold
    """)
    rows = cur.fetchall()
    lines = [f"{yr}: {cnt} sales, avg={_usd(avg)}, avg_change={_pct(chg)}"
             for yr, cnt, avg, chg in rows]
    examples.append({
        "prompt": "How did average sale prices and market activity evolve from 2006 to 2010?",
        "target": (
            "Annual market overview: " + "; ".join(lines) + ". "
            "The data reflects Ames market dynamics during the US financial crisis period, "
            "with notable variations in both volume and price levels across years."
        ),
    })

    # ── 2f: discount analysis ─────────────────────────────────────────
    cur.execute("""
        SELECT Discount_Flag,
               COUNT(*) as cnt,
               AVG(Price) as avg_p,
               AVG(Price_Elasticity) as avg_e
        FROM sales GROUP BY Discount_Flag
    """)
    rows = cur.fetchall()
    for flag, cnt, avg, elast in rows:
        label = "non-standard" if flag else "standard"
        examples.append({
            "prompt": (
                f"What is the average price and elasticity for {label} sales "
                f"(Discount_Flag={flag})?"
            ),
            "target": (
                f"There are {cnt} {label} sales in the dataset. "
                f"Average price: {_usd(avg)}, average elasticity: {elast:.3f}. "
                + (
                    "Non-standard sales (auctions, family transfers, etc.) "
                    "often deviate from market-rate pricing."
                    if flag else
                    "Standard market sales represent typical arm's-length transactions."
                )
            ),
        })

    # ── 2g: garage effect ─────────────────────────────────────────────
    cur.execute("""
        SELECT Garage_Cars, COUNT(*) as cnt, AVG(Price) as avg_p
        FROM sales GROUP BY Garage_Cars ORDER BY Garage_Cars
    """)
    rows = cur.fetchall()
    lines = [f"{g} spot{'s' if g!=1 else ''}: {cnt} homes, avg={_usd(avg)}"
             for g, cnt, avg in rows]
    examples.append({
        "prompt": "How does the number of garage parking spots affect sale prices?",
        "target": (
            "Garage capacity vs. price: " + "; ".join(lines) + ". "
            "Properties with more parking spots generally command higher prices, "
            "reflecting the premium buyers place on parking availability."
        ),
    })

    return examples


# ──────────────────────────────────────────────────────────────────────
# 3. SQL question-answer pairs  (teach NL→SQL mapping)
# ──────────────────────────────────────────────────────────────────────

def _sql_examples(con: sqlite3.Connection) -> list[dict]:
    """
    Each example contains a natural language question, the correct SQL,
    and the verbal answer — reinforcing the Text-to-SQL→Answer pattern
    used by qa_engine.py.
    """
    cur = con.cursor()
    examples = []

    # Helper: run SQL, get first value
    def val(sql: str) -> Any:
        cur.execute(sql)
        r = cur.fetchone()
        return r[0] if r else None

    def rows(sql: str) -> list[tuple]:
        cur.execute(sql)
        return cur.fetchall()

    # ── count total ────────────────────────────────────────────────────
    total = val("SELECT COUNT(*) FROM sales")
    examples.append({
        "prompt": (
            "Write a SQL query to count total records in the sales table, "
            "then answer: how many property transactions are in the Ames dataset?"
        ),
        "target": (
            f"SELECT COUNT(*) FROM sales;\n"
            f"The Ames Housing dataset contains {total} property transactions."
        ),
    })

    # ── most expensive region ──────────────────────────────────────────
    r = val("SELECT Region FROM sales GROUP BY Region ORDER BY AVG(Price) DESC LIMIT 1")
    p = val(f"SELECT AVG(Price) FROM sales WHERE Region='{r}'")
    examples.append({
        "prompt": "Which neighborhood has the highest average sale price? Give the SQL and the answer.",
        "target": (
            "SELECT Region, AVG(Price) FROM sales GROUP BY Region ORDER BY AVG(Price) DESC LIMIT 1;\n"
            f"The most expensive neighborhood is {r} with an average price of {_usd(p)}."
        ),
    })

    # ── near railroad count ────────────────────────────────────────────
    n_rail = val("SELECT COUNT(*) FROM sales WHERE Condition_1 IN ('RRAe','RRAn','RRNe','RRNn')")
    examples.append({
        "prompt": "How many properties in Ames are located near a railroad? Write the SQL.",
        "target": (
            "SELECT COUNT(*) FROM sales "
            "WHERE Condition_1 IN ('RRAe','RRAn','RRNe','RRNn');\n"
            f"{n_rail} properties in the dataset are located near a railroad."
        ),
    })

    # ── elite vs budget avg price ──────────────────────────────────────
    elite  = val("SELECT AVG(Price) FROM sales WHERE Overall_Qual >= 7")
    budget = val("SELECT AVG(Price) FROM sales WHERE Overall_Qual <= 4")
    examples.append({
        "prompt": (
            "Compare average prices between elite (Overall_Qual >= 7) "
            "and budget (Overall_Qual <= 4) properties using SQL."
        ),
        "target": (
            "SELECT 'elite' as segment, AVG(Price) FROM sales WHERE Overall_Qual >= 7\n"
            "UNION ALL\n"
            "SELECT 'budget', AVG(Price) FROM sales WHERE Overall_Qual <= 4;\n"
            f"Elite properties average {_usd(elite)}, "
            f"while budget properties average {_usd(budget)} — "
            f"a difference of {_usd(elite - budget)}."
        ),
    })

    # ── best year for sales ────────────────────────────────────────────
    best_yr, best_cnt = val(
        "SELECT Year_Sold, COUNT(*) as c FROM sales GROUP BY Year_Sold ORDER BY c DESC LIMIT 1"
    ), None
    r2 = rows("SELECT Year_Sold, COUNT(*) FROM sales GROUP BY Year_Sold ORDER BY COUNT(*) DESC LIMIT 1")
    if r2:
        best_yr, best_cnt = r2[0]
    examples.append({
        "prompt": "Which year had the most property sales in Ames? Provide the SQL.",
        "target": (
            "SELECT Year_Sold, COUNT(*) as cnt FROM sales "
            "GROUP BY Year_Sold ORDER BY cnt DESC LIMIT 1;\n"
            f"{best_yr} had the highest sales volume with {best_cnt} transactions."
        ),
    })

    # ── avg elasticity per category ────────────────────────────────────
    cat_rows = rows(
        "SELECT Product_Category, AVG(Price_Elasticity) "
        "FROM sales GROUP BY Product_Category ORDER BY AVG(Price_Elasticity) DESC"
    )
    cat_lines = [f"{c}: {e:.3f}" for c, e in cat_rows]
    examples.append({
        "prompt": (
            "What is the average price elasticity for each property category "
            "(house, apartment, studio)?"
        ),
        "target": (
            "SELECT Product_Category, AVG(Price_Elasticity) "
            "FROM sales GROUP BY Product_Category;\n"
            "Average price elasticity by category: " + "; ".join(cat_lines) + "."
        ),
    })

    # ── discount sales share ───────────────────────────────────────────
    disc_cnt  = val("SELECT COUNT(*) FROM sales WHERE Discount_Flag = 1")
    disc_pct  = disc_cnt / total * 100
    examples.append({
        "prompt": "What percentage of sales in Ames were non-standard (discounted) transactions?",
        "target": (
            "SELECT COUNT(*) FROM sales WHERE Discount_Flag = 1;\n"
            f"{disc_cnt} out of {total} sales ({disc_pct:.1f}%) were non-standard transactions, "
            "including auctions, family sales, and contract deals."
        ),
    })

    # ── top 5 regions by volume ────────────────────────────────────────
    top5 = rows(
        "SELECT Region, COUNT(*) as cnt FROM sales GROUP BY Region ORDER BY cnt DESC LIMIT 5"
    )
    t5_lines = [f"{r}: {c}" for r, c in top5]
    examples.append({
        "prompt": "Which 5 neighborhoods have the most property transactions?",
        "target": (
            "SELECT Region, COUNT(*) as cnt FROM sales "
            "GROUP BY Region ORDER BY cnt DESC LIMIT 5;\n"
            "Top 5 neighborhoods by transaction volume: " + ", ".join(t5_lines) + "."
        ),
    })

    # ── avg garage cars per category ──────────────────────────────────
    g_rows = rows(
        "SELECT Product_Category, AVG(Garage_Cars) "
        "FROM sales GROUP BY Product_Category ORDER BY AVG(Garage_Cars) DESC"
    )
    g_lines = [f"{c}: {g:.2f} spots avg" for c, g in g_rows]
    examples.append({
        "prompt": "How does the average number of garage parking spots differ across property types?",
        "target": (
            "SELECT Product_Category, AVG(Garage_Cars) "
            "FROM sales GROUP BY Product_Category;\n"
            "Average garage capacity by type: " + "; ".join(g_lines) + "."
        ),
    })

    # ── region-year price change ───────────────────────────────────────
    ryr = rows("""
        SELECT Region, Year_Sold, AVG(Sales_Change_Pct) as chg
        FROM sales WHERE Sales_Change_Pct IS NOT NULL
        GROUP BY Region, Year_Sold
        HAVING chg > 0.1
        ORDER BY chg DESC LIMIT 3
    """)
    if ryr:
        ryr_lines = [f"{rg} in {yr}: +{_pct(c)}" for rg, yr, c in ryr]
        examples.append({
            "prompt": "Which region-year combinations showed the strongest positive price growth?",
            "target": (
                "SELECT Region, Year_Sold, AVG(Sales_Change_Pct) FROM sales "
                "WHERE Sales_Change_Pct IS NOT NULL "
                "GROUP BY Region, Year_Sold HAVING AVG(Sales_Change_Pct) > 0.1 "
                "ORDER BY AVG(Sales_Change_Pct) DESC LIMIT 3;\n"
                "Strongest price growth: " + "; ".join(ryr_lines) + "."
            ),
        })

    return examples


# ──────────────────────────────────────────────────────────────────────
# 4. Comparative / hypothetical questions
# ──────────────────────────────────────────────────────────────────────

def _comparative_examples(con: sqlite3.Connection) -> list[dict]:
    cur = con.cursor()
    examples = []

    # Compare two specific regions
    cur.execute("""
        SELECT Region, AVG(Price) as avg_p, AVG(Price_Elasticity) as avg_e,
               COUNT(*) as cnt
        FROM sales GROUP BY Region ORDER BY avg_p DESC
    """)
    all_regions = cur.fetchall()

    for i in range(0, min(len(all_regions) - 1, 12), 2):
        r1, p1, e1, n1 = all_regions[i]
        r2, p2, e2, n2 = all_regions[i + 1]
        examples.append({
            "prompt": f"Compare the real estate market in {r1} vs {r2}.",
            "target": (
                f"{r1} vs {r2}: "
                f"{r1} has a higher average price ({_usd(p1)} vs {_usd(p2)}) "
                f"and {'higher' if e1 > e2 else 'lower'} price elasticity "
                f"({e1:.3f} vs {e2:.3f}). "
                f"Transaction volume: {n1} ({r1}) vs {n2} ({r2}). "
                + (
                    f"{r1} is the more premium market of the two."
                    if p1 > p2
                    else f"{r2} is the more affordable neighborhood."
                )
            ),
        })

    # Quality vs price gradient
    cur.execute("""
        SELECT Overall_Qual, COUNT(*) as cnt, AVG(Price) as avg_p
        FROM sales GROUP BY Overall_Qual ORDER BY Overall_Qual
    """)
    qual_rows = cur.fetchall()
    lines = [f"Q{q}: {cnt} homes, avg={_usd(avg)}" for q, cnt, avg in qual_rows]
    examples.append({
        "prompt": "Show the relationship between overall quality score and average sale price.",
        "target": (
            "Quality-to-price gradient: " + "; ".join(lines) + ". "
            "There is a clear positive relationship — each quality point increase "
            "corresponds to a notable rise in average price."
        ),
    })

    # Garage × category cross
    cur.execute("""
        SELECT Product_Category, Garage_Cars, COUNT(*) as cnt, AVG(Price) as avg_p
        FROM sales
        WHERE Garage_Cars IN (0, 1, 2, 3)
        GROUP BY Product_Category, Garage_Cars
        ORDER BY Product_Category, Garage_Cars
    """)
    cross_rows = cur.fetchall()
    lines = [f"{cat}/{g}spots: {cnt} sales, avg={_usd(avg)}"
             for cat, g, cnt, avg in cross_rows]
    examples.append({
        "prompt": (
            "How do garage spots and property category interact to affect prices? "
            "(cross-analysis)"
        ),
        "target": (
            "Category × Garage cross-analysis: " + "; ".join(lines[:8]) + ". "
            "Houses with more garage spaces consistently show higher average prices, "
            "while apartments show less variation by parking capacity."
        ),
    })

    return examples


# ──────────────────────────────────────────────────────────────────────
# 5. Sampling — control example count per row
# ──────────────────────────────────────────────────────────────────────

def _sample_row_examples(
    rows: list[dict],
    rng: random.Random,
    max_per_row: int = 3,
) -> list[dict]:
    """
    For each DB row generate up to max_per_row examples and randomly sample
    to avoid over-representing rows with many template matches.
    """
    all_ex: list[dict] = []
    for row in rows:
        ex = _row_examples(row)
        if len(ex) > max_per_row:
            ex = rng.sample(ex, max_per_row)
        all_ex.extend(ex)
    return all_ex


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main() -> None:
    rng = random.Random(SEED)

    if not DB_PATH.exists():
        print(f"ERROR: Database not found at {DB_PATH}")
        print("Run  python main.py --no-llm  first to generate the database.")
        sys.exit(1)

    print(f"Connecting to {DB_PATH} …")
    con = sqlite3.connect(str(DB_PATH))
    con.row_factory = sqlite3.Row

    # Load all rows as dicts
    cur = con.cursor()
    cur.execute("SELECT * FROM sales")
    db_rows = [dict(r) for r in cur.fetchall()]
    print(f"Loaded {len(db_rows)} rows from sales table.")

    print("Generating examples …")

    # 1. Row-level examples (~3–5 per row, then sample to 3)
    row_ex = _sample_row_examples(db_rows, rng, max_per_row=3)
    print(f"  Row-level        : {len(row_ex):>5} examples")

    # 2. Regional aggregate examples
    reg_ex = _regional_examples(con)
    print(f"  Regional agg.    : {len(reg_ex):>5} examples")

    # 3. SQL question-answer pairs
    sql_ex = _sql_examples(con)
    print(f"  SQL Q&A pairs    : {len(sql_ex):>5} examples")

    # 4. Comparative examples
    cmp_ex = _comparative_examples(con)
    print(f"  Comparative      : {len(cmp_ex):>5} examples")

    con.close()

    # Combine and shuffle
    all_examples = row_ex + reg_ex + sql_ex + cmp_ex
    rng.shuffle(all_examples)
    print(f"  Total            : {len(all_examples):>5} examples")

    # Train / val split
    split = int(len(all_examples) * TRAIN_RATIO)
    train_data = all_examples[:split]
    val_data   = all_examples[split:]

    # Write outputs
    train_path = OUT_DIR / "train_full.json"
    val_path   = OUT_DIR / "val_full.json"

    train_path.write_text(
        json.dumps(train_data, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    val_path.write_text(
        json.dumps(val_data, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"\nDone.")
    print(f"  Train : {len(train_data):>5} examples → {train_path}")
    print(f"  Val   : {len(val_data):>5} examples → {val_path}")
    print(f"\nFine-tune with:")
    print(f"  python finetune/train.py \\")
    print(f"      --train finetune/train_full.json \\")
    print(f"      --val   finetune/val_full.json")


if __name__ == "__main__":
    main()
