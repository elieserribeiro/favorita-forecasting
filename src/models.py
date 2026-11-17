"""Baselines and direct multi-horizon LightGBM models."""
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.features import N_HORIZONS
from src.metrics import perishable_weights

CAT_FEATURES = ["store_nbr", "family", "store_type", "city", "state", "cluster"]
_NOT_FEATURES = ("item_nbr", "cutoff")

DEFAULT_PARAMS = {
    "objective": "regression",   # L2 on log1p(sales) = optimizes the official metric
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_data_in_leaf": 200,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "metric": "l2",
    "verbose": -1,
    "seed": 42,
}


def national_holidays(holidays: pd.DataFrame) -> pd.DatetimeIndex:
    """Dates of national holidays/events actually celebrated (excludes compensated work days)."""
    h = holidays[(holidays["locale"] == "National") & (~holidays["transferred"])
                 & (holidays["type"] != "Work Day")]
    return pd.DatetimeIndex(h["date"].unique())


def base_features(df: pd.DataFrame) -> list[str]:
    """Shared features: everything except ids, targets and day-specific promo columns."""
    return [c for c in df.columns
            if c not in _NOT_FEATURES and not c.startswith(("y_d", "promo_d"))]


def horizon_frame(df: pd.DataFrame, h: int, holidays: pd.DatetimeIndex) -> pd.DataFrame:
    """Features for horizon h: shared features + promotion and calendar of the target day."""
    target_date = df["cutoff"] + pd.Timedelta(days=h - 1)
    dow = target_date.dt.dayofweek.to_numpy()
    X = df[base_features(df)].copy()
    X["promo_target"] = df[f"promo_d{h}"]
    X["target_dow"] = dow.astype("int8")
    X["target_dom"] = target_date.dt.day.astype("int8")
    X["target_month_start"] = (target_date.dt.day <= 5).astype("int8")
    X["target_holiday"] = target_date.isin(holidays).astype("int8")
    rows = np.arange(len(df))
    for weeks in (4, 16):
        cols = df[[f"dow{d}_mean_{weeks}w" for d in range(7)]].to_numpy()
        X[f"target_dow_mean_{weeks}w"] = cols[rows, dow]
    return X


def baselines(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """Simple benchmarks (log1p scale), each an array of shape (rows, N_HORIZONS)."""
    n = len(df)
    dow_cols = df[[f"dow{d}_mean_4w" for d in range(7)]].to_numpy()
    target_dow = np.stack([(df["cutoff"] + pd.Timedelta(days=h)).dt.dayofweek.to_numpy()
                           for h in range(N_HORIZONS)], axis=1)
    return {
        "Média 7 dias": np.repeat(df["mean_7d"].to_numpy()[:, None], N_HORIZONS, axis=1),
        "Média 28 dias": np.repeat(df["mean_28d"].to_numpy()[:, None], N_HORIZONS, axis=1),
        "Média do dia da semana (4 sem.)": dow_cols[np.arange(n)[:, None], target_dow],
    }


def train_direct(df_train: pd.DataFrame, df_val: pd.DataFrame, holidays: pd.DatetimeIndex,
                 horizons=range(1, N_HORIZONS + 1), params: dict | None = None,
                 num_boost_round: int = 1000, models_dir: Path | None = None):
    """Train one LightGBM per horizon (direct strategy).

    The LAST training cutoff is used only for early stopping, so the validation
    period stays completely unseen.

    Returns
    -------
    preds : array (rows of df_val, len(horizons)) in log1p scale
    importance : DataFrame (features x horizons) with gain importance
    """
    params = {**DEFAULT_PARAMS, **(params or {})}
    es_cutoff = df_train["cutoff"].max()
    fit = (df_train["cutoff"] < es_cutoff).to_numpy()
    w = perishable_weights(df_train["perishable"])
    preds, importance = [], {}

    for h in horizons:
        start = time.time()
        X = horizon_frame(df_train, h, holidays)
        y = df_train[f"y_d{h}"].to_numpy()
        dtrain = lgb.Dataset(X[fit], y[fit], weight=w[fit], categorical_feature=CAT_FEATURES)
        dvalid = lgb.Dataset(X[~fit], y[~fit], weight=w[~fit], reference=dtrain)
        model = lgb.train(params, dtrain, num_boost_round=num_boost_round, valid_sets=[dvalid],
                          callbacks=[lgb.early_stopping(50, verbose=False)])
        del X, dtrain, dvalid

        Xv = horizon_frame(df_val, h, holidays)
        preds.append(model.predict(Xv, num_iteration=model.best_iteration).clip(0))
        importance[h] = pd.Series(model.feature_importance("gain"), index=Xv.columns)
        if models_dir is not None:
            model.save_model(str(Path(models_dir) / f"lgbm_h{h:02d}.txt"),
                             num_iteration=model.best_iteration)
        print(f"h = {h:2d} | árvores: {model.best_iteration:4d} | {time.time() - start:5.1f} s")

    return np.column_stack(preds), pd.DataFrame(importance)


def to_long(df: pd.DataFrame, preds: dict[str, np.ndarray]) -> pd.DataFrame:
    """One row per store x item x horizon with the true value and every model's prediction."""
    n = len(df)
    out = pd.DataFrame({
        "store_nbr": np.repeat(df["store_nbr"].to_numpy(), N_HORIZONS),
        "item_nbr": np.repeat(df["item_nbr"].to_numpy(), N_HORIZONS),
        "family": np.repeat(df["family"].to_numpy(), N_HORIZONS),
        "perishable": np.repeat(df["perishable"].to_numpy(), N_HORIZONS),
        "h": np.tile(np.arange(1, N_HORIZONS + 1, dtype="int8"), n),
        "date": (np.repeat(df["cutoff"].to_numpy(), N_HORIZONS)
                 + np.tile(np.arange(N_HORIZONS), n).astype("timedelta64[D]")),
        "y_true": df[[f"y_d{h}" for h in range(1, N_HORIZONS + 1)]].to_numpy().ravel(),
    })
    for name, arr in preds.items():
        out[name] = arr.astype("float32").ravel()
    return out