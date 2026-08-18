"""
prompt_builder.py
=================
Converts structured results from Q1-Q3 analyses into short prompts for FLAN-T5.
The functions keep prompts compact so they fit into the 512-token input limit.
"""
from __future__ import annotations

import math
from typing import Any

import pandas as pd


def _fmt_num(value: Any, digits: int = 3) -> str:
    try:
        value = float(value)
        if math.isnan(value):
            return "n/a"
        return f"{value:.{digits}f}"
    except Exception:
        return "n/a"


def _rows_to_lines(df: pd.DataFrame, columns: list[str], limit: int) -> str:
    if df is None or df.empty:
        return "No rows available."
    rows = []
    for _, row in df.head(limit).iterrows():
        parts = [f"{col}={row.get(col, 'n/a')}" for col in columns]
        rows.append("; ".join(parts))
    return "\n".join(f"- {line}" for line in rows)


def build_q1_prompt(parking_table: pd.DataFrame, rail_table: pd.DataFrame) -> str:
    """Prompt for infrastructure/parking effect analysis."""
    parking = parking_table.copy() if parking_table is not None else pd.DataFrame()
    rail = rail_table.copy() if rail_table is not None else pd.DataFrame()

    if not parking.empty:
        parking["abs_elasticity"] = parking["mean_elasticity"].abs()
        parking = parking.sort_values("abs_elasticity", ascending=False)

    if not rail.empty:
        rail["abs_elasticity"] = rail["mean_elasticity"].abs()
        rail = rail.sort_values("abs_elasticity", ascending=False)

    parking_lines = _rows_to_lines(
        parking[["Region", "Parking_Bin", "mean_elasticity", "count"]] if not parking.empty else parking,
        ["Region", "Parking_Bin", "mean_elasticity", "count"],
        5,
    )
    rail_lines = _rows_to_lines(
        rail[["Region", "Near_Rail", "mean_elasticity", "count"]] if not rail.empty else rail,
        ["Region", "Near_Rail", "mean_elasticity", "count"],
        4,
    )

    return (
        "Write a concise 3-sentence analytical comment in English. "
        "Interpret the effect of parking availability and railroad proximity on relative price level. "
        "In this project, Price_Elasticity means relative deviation from the neighborhood median price, "
        "not classical demand elasticity. Use only the numbers below and avoid inventing new values.\n\n"
        f"Top parking effects:\n{parking_lines}\n\n"
        f"Top railroad-proximity effects:\n{rail_lines}"
    )


def build_q2_prompt(q2_results: dict) -> str:
    """Prompt for PyTorch linear-model interpretation."""
    if not q2_results or "error" in q2_results:
        return (
            "Write a 3-sentence technical comment in English. "
            f"The PyTorch model could not be trained because: {q2_results.get('error', 'unknown error')}. "
            "Explain what this means for the reliability of the analysis and what should be improved."
        )

    weights = q2_results.get("weights", {})
    weights_text = ", ".join(f"{k}={v:+.4f}" for k, v in weights.items()) or "no weights"

    return (
        "Write a concise 3-sentence analytical comment in English. "
        "Interpret a PyTorch linear regression model predicting Sales_Change_Pct from product category "
        "and Discount_Flag. Discuss model quality and the most influential features. "
        "Use only the metrics below and avoid inventing values.\n\n"
        f"train_rmse={q2_results.get('train_rmse', 'n/a'):.6f}\n"
        f"val_rmse={q2_results.get('val_rmse', 'n/a'):.6f}\n"
        f"n_train={q2_results.get('n_train', 'n/a')}\n"
        f"n_val={q2_results.get('n_val', 'n/a')}\n"
        f"bias={q2_results.get('bias', 'n/a'):.6f}\n"
        f"weights: {weights_text}"
    )


def build_q3_prompt(fisher_df: pd.DataFrame) -> str:
    """Prompt for Fisher z-test interpretation."""
    if fisher_df is None or fisher_df.empty:
        return (
            "Write a concise 3-sentence analytical comment in English. "
            "No region had enough elite and budget observations for a Fisher z-test. "
            "Explain the limitation and what kind of additional data would be needed."
        )

    work = fisher_df.copy()
    work = work.sort_values("p_value_approx", ascending=True)
    sig_count = int((work["p_value_approx"] < 0.05).sum())
    rows = []
    for _, row in work.head(5).iterrows():
        rows.append(
            f"- Region={row['Region']}; r_elite={_fmt_num(row['r_elite'])}; "
            f"n_elite={int(row['n_elite'])}; r_budget={_fmt_num(row['r_budget'])}; "
            f"n_budget={int(row['n_budget'])}; z={_fmt_num(row['fisher_z'])}; "
            f"p={_fmt_num(row['p_value_approx'], 4)}"
        )

    return (
        "Write a concise 3-sentence analytical comment in English. "
        "Interpret the Fisher z-test comparing price-volume correlations in elite and budget housing segments. "
        "Mention whether statistically significant regional differences appear at p < 0.05. "
        "Use only the numbers below and avoid inventing values.\n\n"
        f"Significant regions: {sig_count} out of {len(work)} tested regions.\n"
        + "\n".join(rows)
    )


def build_executive_prompt(
    q1_parking: pd.DataFrame,
    q1_rail: pd.DataFrame,
    q2: dict,
    q3: pd.DataFrame,
) -> str:
    """Prompt summarising all analyses for a decision-maker."""
    best_parking = "n/a"
    if q1_parking is not None and not q1_parking.empty:
        p = q1_parking.assign(abs_elasticity=q1_parking["mean_elasticity"].abs()).sort_values(
            "abs_elasticity", ascending=False
        ).iloc[0]
        best_parking = (
            f"{p['Region']} / parking={p['Parking_Bin']} / "
            f"mean_elasticity={_fmt_num(p['mean_elasticity'])} / n={int(p['count'])}"
        )

    rail_summary = "n/a"
    if q1_rail is not None and not q1_rail.empty:
        r = q1_rail.assign(abs_elasticity=q1_rail["mean_elasticity"].abs()).sort_values(
            "abs_elasticity", ascending=False
        ).iloc[0]
        rail_summary = (
            f"{r['Region']} / near_rail={bool(r['Near_Rail'])} / "
            f"mean_elasticity={_fmt_num(r['mean_elasticity'])} / n={int(r['count'])}"
        )

    q2_text = "model training failed"
    if q2 and "error" not in q2:
        q2_text = (
            f"train_rmse={q2.get('train_rmse'):.6f}; "
            f"val_rmse={q2.get('val_rmse'):.6f}; n_train={q2.get('n_train')}; n_val={q2.get('n_val')}"
        )

    q3_text = "no tested regions"
    if q3 is not None and not q3.empty:
        q3_text = f"{int((q3['p_value_approx'] < 0.05).sum())} significant regions out of {len(q3)}"

    return (
        "Write a 4-sentence executive summary in English for a non-technical decision-maker. "
        "Summarize the most important findings from infrastructure effects, PyTorch price-change modelling, "
        "and Fisher z-test segment comparison. State limitations briefly and avoid inventing values.\n\n"
        f"Strongest parking-related effect: {best_parking}\n"
        f"Strongest railroad-proximity effect: {rail_summary}\n"
        f"Q2 PyTorch model: {q2_text}\n"
        f"Q3 Fisher test: {q3_text}"
    )


def build_all_prompts(results: dict) -> dict[str, str]:
    """Build all Etap 3 prompts from run_all() results."""
    return {
        "q1_infrastructure": build_q1_prompt(results["q1_parking_table"], results["q1_rail_table"]),
        "q2_torch_model": build_q2_prompt(results["q2_torch"]),
        "q3_fisher_segments": build_q3_prompt(results["q3_fisher"]),
        "executive_summary": build_executive_prompt(
            results["q1_parking_table"],
            results["q1_rail_table"],
            results["q2_torch"],
            results["q3_fisher"],
        ),
    }
