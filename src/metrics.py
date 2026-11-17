"""Evaluation metrics for the Favorita demand forecasting project.

All predictions and targets are stored in log1p scale; WAPE and bias are
computed back in units (expm1).
"""
import numpy as np
import pandas as pd

PERISHABLE_WEIGHT = 1.25


def perishable_weights(perishable) -> np.ndarray:
    """Official Kaggle weights: 1.25 for perishable items, 1.0 otherwise."""
    return np.where(np.asarray(perishable) == 1, PERISHABLE_WEIGHT, 1.0)


def nwrmsle(y_true_log, y_pred_log, weights=None) -> float:
    """Normalized Weighted Root Mean Squared Logarithmic Error (official metric).

    Inputs are already in log1p scale. Lower is better.
    """
    y_true_log = np.asarray(y_true_log, dtype="float64")
    y_pred_log = np.asarray(y_pred_log, dtype="float64")
    w = np.ones_like(y_true_log) if weights is None else np.asarray(weights, dtype="float64")
    return float(np.sqrt(np.sum(w * (y_pred_log - y_true_log) ** 2) / np.sum(w)))


def wape(y_true, y_pred) -> float:
    """Weighted Absolute Percentage Error: share of the volume that was mispredicted."""
    y_true = np.asarray(y_true, dtype="float64")
    y_pred = np.asarray(y_pred, dtype="float64")
    return float(np.sum(np.abs(y_pred - y_true)) / np.sum(y_true))


def bias(y_true, y_pred) -> float:
    """Relative bias: > 0 means over-forecast (excess stock), < 0 under-forecast (stockout)."""
    y_true = np.asarray(y_true, dtype="float64")
    y_pred = np.asarray(y_pred, dtype="float64")
    return float((np.sum(y_pred) - np.sum(y_true)) / np.sum(y_true))


def summarize(long: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    """Compare models by fortnight.

    ``long`` has one row per store x item x horizon with columns
    store_nbr, item_nbr, h, perishable, y_true and one column per model (log1p).

    Metrics
    -------
    NWRMSLE          : official metric, daily level (log scale)
    WAPE diário      : % of daily volume mispredicted
    WAPE quinzenal   : % of the fortnight total (per store x item) mispredicted
    Viés             : over (+) or under (-) forecast of total volume
    """
    long = long.assign(quinzena=np.where(long["h"] <= 15, "1ª quinzena (d1–15)",
                                         "2ª quinzena (d16–30)"))
    groups = [*long.groupby("quinzena"), ("30 dias", long)]
    rows = []
    for period, g in groups:
        w = perishable_weights(g["perishable"])
        true_units = np.expm1(g["y_true"].to_numpy("float64"))
        for m in models:
            pred_units = np.expm1(g[m].to_numpy("float64").clip(0))
            totals = (pd.DataFrame({"pair": g["store_nbr"].astype("int64") * 10_000_000 + g["item_nbr"],
                                    "q": g["quinzena"].to_numpy(), "t": true_units, "p": pred_units})
                      .groupby(["pair", "q"])[["t", "p"]].sum())
            rows.append({
                "Período": period, "Modelo": m,
                "NWRMSLE": nwrmsle(g["y_true"], g[m].clip(lower=0), w),
                "WAPE diário": wape(true_units, pred_units),
                "WAPE quinzenal": wape(totals["t"], totals["p"]),
                "Viés": bias(true_units, pred_units),
            })
    return pd.DataFrame(rows).set_index(["Período", "Modelo"])