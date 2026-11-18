"""Final forecast: the next two fortnights after the end of the data."""
import numpy as np
import pandas as pd

from src.features import N_HORIZONS


def _pair_key(store, item) -> np.ndarray:
    return np.asarray(store, dtype="int64") * 10_000_000 + np.asarray(item, dtype="int64")


def extend_promo(promo: pd.DataFrame, test: pd.DataFrame, start,
                 n_days: int = N_HORIZONS, fill_unknown: str = "zero",
                 lag: int = 14) -> pd.DataFrame:
    """Append the planned promotions of the future days (from Kaggle's test.csv).

    Days after the last date in ``test`` have no promotion information:
    - fill_unknown="zero"   : assume no promotion;
    - fill_unknown="repeat" : repeat the promotion calendar of ``lag`` days earlier
                              (14 = same weekday, two weeks before).
    Only the store x item pairs already in the history matrix are kept.
    """
    dates = pd.date_range(start, periods=n_days, freq="D")
    test = test[test["date"].isin(dates)]

    index_keys = pd.Index(_pair_key(promo.index.get_level_values("store_nbr"),
                                    promo.index.get_level_values("item_nbr")))
    rows = index_keys.get_indexer(_pair_key(test["store_nbr"], test["item_nbr"]))
    cols = (test["date"] - dates[0]).dt.days.to_numpy()
    known = rows >= 0

    flags = np.zeros((len(promo), n_days), dtype="int8")
    flags[rows[known], cols[known]] = test["onpromotion"].to_numpy("int8")[known]

    if fill_unknown == "repeat":
        for c in np.flatnonzero(dates > test["date"].max()):
            source = dates[c] - pd.Timedelta(days=lag)
            flags[:, c] = (flags[:, c - lag] if c >= lag else promo[source].to_numpy())

    future = pd.DataFrame(flags, index=promo.index, columns=dates)
    return pd.concat([promo, future], axis=1)


def forecast_long(df: pd.DataFrame, preds_units: np.ndarray) -> pd.DataFrame:
    """One row per store x item x future day with the forecast in units."""
    n = len(df)
    return pd.DataFrame({
        "date": (np.repeat(df["cutoff"].to_numpy(), N_HORIZONS)
                 + np.tile(np.arange(N_HORIZONS), n).astype("timedelta64[D]")),
        "h": np.tile(np.arange(1, N_HORIZONS + 1, dtype="int8"), n),
        "store_nbr": np.repeat(df["store_nbr"].to_numpy(), N_HORIZONS),
        "item_nbr": np.repeat(df["item_nbr"].to_numpy(), N_HORIZONS),
        "family": np.repeat(df["family"].to_numpy(), N_HORIZONS),
        "perishable": np.repeat(df["perishable"].to_numpy(), N_HORIZONS).astype("int8"),
        "previsto": preds_units.astype("float32").ravel(),
    })