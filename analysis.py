"""
Research analyses: infrastructure vs elasticity, torch linear model, Fisher z for correlations.
"""
from __future__ import annotations
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn


def analysis1_infrastructure_elasticity(df: pd.DataFrame) -> pd.DataFrame:
    """
    Q1: Mean Price_Elasticity by Region x parking (Garage_Cars bins) and rail proximity proxy.
    """
    work = df.copy()
    work["Parking_Bin"] = pd.cut(
        work["Garage_Cars"].clip(0, 4),
        bins=[-0.1, 0, 1, 2, 10],
        labels=["0", "1", "2", "3+"],
    )
    work["Near_Rail"] = work["Condition_1"].astype(str).str.contains("RR", na=False)

    g = (
        work.groupby(["Region", "Parking_Bin"], observed=False)["Price_Elasticity"]
        .agg(["mean", "count"])
        .reset_index()
    )
    g = g.rename(columns={"mean": "mean_elasticity"})
    g = g[g["count"] >= 3]

    rail = (
        work.groupby(["Region", "Near_Rail"], observed=False)["Price_Elasticity"]
        .agg(["mean", "count"])
        .reset_index()
    )
    rail = rail.rename(columns={"mean": "mean_elasticity"})
    rail = rail[rail["count"] >= 3]

    return g, rail


def analysis2_torch_sales_change(
    df: pd.DataFrame,
    seed: int = 42,
    epochs: int = 2000,
    lr: float = 0.05,
) -> dict:
    """
    Q2: Predict Sales_Change_Pct from Product_Category (one-hot) + Discount_Flag (torch linear).
    """
    work = df.dropna(subset=["Sales_Change_Pct"]).copy()
    if len(work) < 30:
        return {"error": "Too few rows with Sales_Change_Pct"}

    dummies = pd.get_dummies(work["Product_Category"], prefix="cat", dtype=float)
    X = pd.concat([dummies, work["Discount_Flag"].astype(float)], axis=1)
    y = work["Sales_Change_Pct"].astype(float).values

    years = work["Year_Sold"].values
    cutoff = np.quantile(years, 0.75)
    train_mask = years <= cutoff
    val_mask = ~train_mask

    X_np = X.values.astype(np.float32)
    y_np = y.astype(np.float32).reshape(-1, 1)

    torch.manual_seed(seed)
    X_t = torch.tensor(X_np, dtype=torch.float32)
    y_t = torch.tensor(y_np, dtype=torch.float32)

    n_in = X_t.shape[1]
    model = nn.Linear(n_in, 1)
    loss_fn = nn.MSELoss()
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    Xt, Xv = X_t[train_mask], X_t[val_mask]
    yt, yv = y_t[train_mask], y_t[val_mask]

    for _ in range(epochs):
        opt.zero_grad()
        pred = model(Xt)
        loss = loss_fn(pred, yt)
        loss.backward()
        opt.step()

    with torch.no_grad():
        train_rmse = float(torch.sqrt(loss_fn(model(Xt), yt)).item())
        val_rmse = float(torch.sqrt(loss_fn(model(Xv), yv)).item()) if len(Xv) else float("nan")

    weights = {n: float(v) for n, v in zip(X.columns, model.weight.squeeze().tolist())}
    bias = float(model.bias.item())

    return {
        "feature_columns": list(X.columns),
        "weights": weights,
        "bias": bias,
        "train_rmse": train_rmse,
        "val_rmse": val_rmse,
        "n_train": int(train_mask.sum()),
        "n_val": int(val_mask.sum()),
    }


def fisher_z_test_two_correlations(r1: float, n1: int, r2: float, n2: int) -> tuple[float, float]:
    """Two-tailed z-test for difference of two Pearson correlations (independent groups)."""
    if n1 < 4 or n2 < 4:
        return float("nan"), float("nan")
    z1 = np.arctanh(r1)
    z2 = np.arctanh(r2)
    se = np.sqrt(1.0 / (n1 - 3) + 1.0 / (n2 - 3))
    z_stat = (z1 - z2) / se
    # two-sided p-value: erfc(|z|/sqrt(2)) for standard-normal tail symmetry
    p = float(math.erfc(abs(z_stat) / math.sqrt(2)))
    return float(z_stat), float(p)


def analysis3_correlation_segments(df: pd.DataFrame, min_n: int = 8) -> pd.DataFrame:
    """
    Q3: Compare corr(Price, Sales) for elite (Overall_Qual >= 7) vs budget (<= 4) by Region.
    """
    rows = []
    for region, sub in df.groupby("Region", observed=True):
        elite = sub[sub["Overall_Qual"] >= 7]
        budget = sub[sub["Overall_Qual"] <= 4]
        if len(elite) < min_n or len(budget) < min_n:
            continue
        r1 = elite["Price"].corr(elite["Sales"])
        r2 = budget["Price"].corr(budget["Sales"])
        n1, n2 = len(elite), len(budget)
        z, p = fisher_z_test_two_correlations(r1, n1, r2, n2)
        rows.append(
            {
                "Region": region,
                "r_elite": r1,
                "n_elite": n1,
                "r_budget": r2,
                "n_budget": n2,
                "fisher_z": z,
                "p_value_approx": p,
            }
        )
    return pd.DataFrame(rows)


def run_all(df: pd.DataFrame) -> dict:
    p1, p1_rail = analysis1_infrastructure_elasticity(df)
    p2 = analysis2_torch_sales_change(df)
    p3 = analysis3_correlation_segments(df)
    return {
        "q1_parking_table": p1,
        "q1_rail_table": p1_rail,
        "q2_torch": p2,
        "q3_fisher": p3,
    }
