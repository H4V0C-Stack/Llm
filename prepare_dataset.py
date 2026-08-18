"""
finetune/prepare_dataset.py
===========================
Generates a synthetic fine-tuning dataset (prompt -> target) for FLAN-T5.
The script uses outputs of the Etap 2 analyses and creates deterministic
reference answers. It can run offline when data/ames_report.csv exists.

Run:
    python finetune/prepare_dataset.py
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis import (  # noqa: E402
    analysis1_infrastructure_elasticity,
    analysis2_torch_sales_change,
    analysis3_correlation_segments,
)
from dataset import (  # noqa: E402
    build_report_dataframe,
    ensure_openml_parquet_fallback,
    load_raw,
)

SEED = 42
TRAIN_RATIO = 0.85
OUT_DIR = Path(__file__).resolve().parent


def _sign_text(v: float) -> str:
    return "above" if v > 0 else "below"


def _magnitude(v: float) -> str:
    av = abs(float(v))
    if av < 0.05:
        return "slightly"
    if av < 0.15:
        return "moderately"
    return "strongly"


def load_report_dataframe() -> pd.DataFrame:
    report_csv = ROOT / "data" / "ames_report.csv"
    raw_csv = ROOT / "data" / "raw" / "train.csv"
    if report_csv.exists() and report_csv.stat().st_size > 0:
        return pd.read_csv(report_csv)
    if raw_csv.exists() and raw_csv.stat().st_size > 0:
        return build_report_dataframe(load_raw(raw_csv))
    train_path = ensure_openml_parquet_fallback(ROOT / "data" / "raw")
    return build_report_dataframe(load_raw(train_path))


def examples_from_parking(parking_df: pd.DataFrame) -> list[dict[str, str]]:
    examples: list[dict[str, str]] = []
    for _, row in parking_df.iterrows():
        region = row["Region"]
        parking = row["Parking_Bin"]
        elast = float(row["mean_elasticity"])
        n = int(row["count"])
        prompt = (
            f"Summarize the parking effect in {region}: Parking_Bin={parking}, "
            f"mean_elasticity={elast:.3f}, count={n}. "
            "Price_Elasticity means relative deviation from neighborhood median price."
        )
        target = (
            f"In {region}, properties with {parking} parking spot(s) are {_magnitude(elast)} "
            f"{_sign_text(elast)} the neighborhood median price (mean_elasticity={elast:.3f}, n={n}). "
            "This suggests that parking availability is associated with the relative price level in this area."
        )
        examples.append({"prompt": prompt, "target": target})
    return examples


def examples_from_rail(rail_df: pd.DataFrame) -> list[dict[str, str]]:
    examples: list[dict[str, str]] = []
    for _, row in rail_df.iterrows():
        region = row["Region"]
        near = bool(row["Near_Rail"])
        elast = float(row["mean_elasticity"])
        n = int(row["count"])
        proximity = "near railroad" if near else "not near railroad"
        prompt = (
            f"Explain railroad proximity in {region}: Near_Rail={near}, "
            f"mean_elasticity={elast:.3f}, count={n}."
        )
        target = (
            f"In {region}, properties {proximity} are {_magnitude(elast)} {_sign_text(elast)} "
            f"the neighborhood median price (mean_elasticity={elast:.3f}, n={n}). "
            "The result should be interpreted as a local association, not a universal causal effect."
        )
        examples.append({"prompt": prompt, "target": target})
    return examples


def examples_from_q2(q2: dict) -> list[dict[str, str]]:
    if not q2 or "error" in q2:
        return [{
            "prompt": "The PyTorch model for Sales_Change_Pct could not be trained. Explain the consequence.",
            "target": "The PyTorch model could not be trained because there were too few usable rows. This limits the predictive part of the analysis and indicates that more complete temporal data would be needed."
        }]

    weights = q2.get("weights", {})
    top_feature = max(weights, key=lambda k: abs(weights[k])) if weights else "unknown"
    weights_text = ", ".join(f"{k}={v:+.4f}" for k, v in weights.items())
    prompt1 = (
        f"Interpret PyTorch regression results: train_rmse={q2['train_rmse']:.4f}, "
        f"val_rmse={q2['val_rmse']:.4f}, n_train={q2['n_train']}, n_val={q2['n_val']}."
    )
    target1 = (
        f"The PyTorch linear model achieved train_rmse={q2['train_rmse']:.4f} and "
        f"val_rmse={q2['val_rmse']:.4f}. With {q2['n_train']} training rows and "
        f"{q2['n_val']} validation rows, it should be treated as a baseline model rather than a final high-accuracy predictor."
    )
    prompt2 = f"Interpret feature weights for Sales_Change_Pct model: {weights_text}."
    target2 = (
        f"The strongest feature by absolute weight is {top_feature} "
        f"({weights[top_feature]:+.4f}). Positive weights indicate association with higher predicted price change, while negative weights indicate lower predicted price change."
    )
    return [{"prompt": prompt1, "target": target1}, {"prompt": prompt2, "target": target2}]


def examples_from_q3(fisher_df: pd.DataFrame) -> list[dict[str, str]]:
    if fisher_df.empty:
        return [{
            "prompt": "No region had enough observations for Fisher z-test. Explain the limitation.",
            "target": "No Fisher z-test could be performed because the elite and budget segments did not both reach the minimum sample size. The analysis therefore requires more observations per segment and neighborhood."
        }]

    examples: list[dict[str, str]] = []
    for _, row in fisher_df.iterrows():
        sig = float(row["p_value_approx"]) < 0.05
        prompt = (
            f"Fisher z-test in {row['Region']}: r_elite={row['r_elite']:.3f}, "
            f"n_elite={int(row['n_elite'])}, r_budget={row['r_budget']:.3f}, "
            f"n_budget={int(row['n_budget'])}, p={row['p_value_approx']:.4f}."
        )
        if sig:
            target = (
                f"In {row['Region']}, the difference between elite and budget price-volume correlations "
                f"is statistically significant (p={row['p_value_approx']:.4f}). "
                "This suggests that the two market segments behave differently in this neighborhood."
            )
        else:
            target = (
                f"In {row['Region']}, the Fisher z-test is not statistically significant "
                f"(p={row['p_value_approx']:.4f}). The available data does not support a clear difference "
                "between elite and budget price-volume correlations."
            )
        examples.append({"prompt": prompt, "target": target})
    return examples


def main() -> None:
    random.seed(SEED)
    np.random.seed(SEED)

    df = load_report_dataframe()
    q1_parking, q1_rail = analysis1_infrastructure_elasticity(df)
    q2 = analysis2_torch_sales_change(df)
    q3 = analysis3_correlation_segments(df)

    examples: list[dict[str, str]] = []
    examples.extend(examples_from_parking(q1_parking))
    examples.extend(examples_from_rail(q1_rail))
    examples.extend(examples_from_q2(q2))
    examples.extend(examples_from_q3(q3))

    random.shuffle(examples)
    split = int(len(examples) * TRAIN_RATIO)
    train_data = examples[:split]
    val_data = examples[split:]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "train.json").write_text(json.dumps(train_data, indent=2, ensure_ascii=False), encoding="utf-8")
    (OUT_DIR / "val.json").write_text(json.dumps(val_data, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Done. Train examples: {len(train_data)} | Validation examples: {len(val_data)}")
    print(f"Files written to: {OUT_DIR}")


if __name__ == "__main__":
    main()
