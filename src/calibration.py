"""Bias calibration: bring log-scale forecasts back to the right volume in units.

Models trained on log1p(sales) predict something close to a geometric mean,
which is systematically BELOW the average sales (Jensen's inequality).
A multiplicative factor = actual units / predicted units, estimated on a
period the model did not learn from, corrects the total volume.
"""
import numpy as np
import pandas as pd

from src.features import N_HORIZONS

_Y = [f"y_d{h}" for h in range(1, N_HORIZONS + 1)]
_FORTNIGHT = np.where(np.arange(1, N_HORIZONS + 1) <= 15, 0, 1)  # 0 = 1st, 1 = 2nd


def fit_calibration(df: pd.DataFrame, preds_log: np.ndarray, by: str | None = None,
                    clip: tuple[float, float] = (0.5, 3.0)) -> pd.DataFrame:
    """Estimate multiplicative factors (actual units / predicted units).

    by=None   -> one factor per horizon (h = 1 ... 30)
    by="col"  -> one factor per group of ``col`` and fortnight (1st / 2nd)
    """
    true = np.expm1(df[_Y].to_numpy("float64"))
    pred = np.expm1(preds_log.astype("float64"))
    if by is None:
        factor = true.sum(axis=0) / pred.sum(axis=0)
        return pd.DataFrame({"fator": factor}, index=pd.RangeIndex(1, N_HORIZONS + 1, name="h"))

    keys = df[by].to_numpy()
    t = pd.DataFrame(true).groupby(keys).sum().T.groupby(_FORTNIGHT).sum().T
    p = pd.DataFrame(pred).groupby(keys).sum().T.groupby(_FORTNIGHT).sum().T
    factors = (t / p).clip(*clip).fillna(1.0)
    factors.columns = ["1ª quinzena", "2ª quinzena"]
    factors.index.name = by
    return factors


def apply_calibration(df: pd.DataFrame, preds_log: np.ndarray, factors: pd.DataFrame,
                      by: str | None = None) -> np.ndarray:
    """Multiply predicted units by the factors and return to log1p scale."""
    pred = np.expm1(preds_log.astype("float64"))
    if by is None:
        mult = factors["fator"].to_numpy()[None, :]
    else:
        per_row = factors.reindex(df[by].to_numpy()).fillna(1.0).to_numpy()  # rows x 2
        mult = per_row[:, _FORTNIGHT]
    return np.log1p(pred * mult).astype("float32")