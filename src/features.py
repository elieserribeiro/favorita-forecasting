"""Feature engineering for direct multi-horizon demand forecasting.

Strategy
--------
For each cutoff date ``t`` (forecast origin), every store x item row receives:
- features computed ONLY from data before ``t`` (no leakage);
- promotions planned for the next days (known in advance);
- targets: log1p(sales) for each of the next ``n_horizons`` days (y_d1 ... y_d30).

One model per horizon is trained later (direct strategy).
"""
import numpy as np
import pandas as pd

WINDOWS = (3, 7, 14, 28, 56, 112)  # history windows (days)
DOW_WEEKS = (4, 16)                # weeks used in day-of-week means
N_HORIZONS = 30                    # days ahead to forecast


def build_wide(train: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pivot long sales into wide matrices (rows = store x item, columns = dates).

    Returns
    -------
    sales : log1p(unit_sales), float32. Days without a sale are filled with 0.
    promo : onpromotion flag (0/1), int8.
    """
    start = train["date"].min()
    dates = pd.date_range(start, train["date"].max(), freq="D")

    # Row position of each store x item pair and column position of each date
    # (numpy filling uses far less memory than pandas unstack on ~24M rows)
    key = train["store_nbr"].to_numpy("int64") * 10_000_000 + train["item_nbr"].to_numpy("int64")
    row, pairs = pd.factorize(key, sort=True)
    col = (train["date"] - start).dt.days.to_numpy()
    index = pd.MultiIndex.from_arrays([(pairs // 10_000_000).astype("int8"),
                                       (pairs % 10_000_000).astype("int32")],
                                      names=["store_nbr", "item_nbr"])

    values = np.zeros((len(pairs), len(dates)), dtype="float32")
    values[row, col] = np.log1p(train["unit_sales"].clip(lower=0).to_numpy("float32"))
    sales = pd.DataFrame(values, index=index, columns=dates)

    flags = np.zeros((len(pairs), len(dates)), dtype="int8")
    flags[row, col] = train["onpromotion"].to_numpy("int8")
    promo = pd.DataFrame(flags, index=index, columns=dates)
    return sales, promo


def _days(cutoff: pd.Timestamp, back: int) -> pd.DatetimeIndex:
    """The ``back`` days immediately before the cutoff."""
    return pd.date_range(cutoff - pd.Timedelta(days=back), periods=back, freq="D")


def make_features(sales: pd.DataFrame, promo: pd.DataFrame, cutoff,
                  n_horizons: int = N_HORIZONS,
                  with_target: bool = True) -> pd.DataFrame:
    """Build one row per store x item for a given cutoff date."""
    cutoff = pd.Timestamp(cutoff)
    X = {}

    # 1) Recent history: mean of log1p(sales) in several windows
    for w in WINDOWS:
        X[f"mean_{w}d"] = sales[_days(cutoff, w)].mean(axis=1).values
    X["last_day"] = sales[cutoff - pd.Timedelta(days=1)].values

    # 2) Intermittency: days with sales and days since the last sale
    for w in (28, 112):
        X[f"days_with_sales_{w}d"] = (sales[_days(cutoff, w)] > 0).sum(axis=1).values
    hist = sales[_days(cutoff, 112)].values > 0
    from_end = np.argmax(hist[:, ::-1], axis=1)           # 0 = day before cutoff
    X["days_since_last_sale"] = np.where(hist.any(axis=1), from_end + 1, 113)

    # 3) Weekly pattern: mean of each weekday over the last 4 and 16 weeks
    for weeks in DOW_WEEKS:
        start = cutoff - pd.Timedelta(days=7 * weeks)
        for d in range(7):
            days = pd.date_range(start + pd.Timedelta(days=d), periods=weeks, freq="7D")
            X[f"dow{days[0].dayofweek}_mean_{weeks}w"] = sales[days].mean(axis=1).values

    # 4) Promotions: recent history and planned promotions in the forecast window
    for w in (14, 56):
        X[f"promo_sum_{w}d"] = promo[_days(cutoff, w)].sum(axis=1).values
    future = pd.date_range(cutoff, periods=n_horizons, freq="D")
    promo_future = promo.reindex(columns=future, fill_value=0).values
    for h in range(n_horizons):
        X[f"promo_d{h + 1}"] = promo_future[:, h]
    X["promo_next_1_15"] = promo_future[:, :15].sum(axis=1)
    X["promo_next_16_30"] = promo_future[:, 15:30].sum(axis=1)

    out = pd.DataFrame(X, index=sales.index)

    # 5) Item level across the whole chain (all stores)
    for col in ("mean_7d", "mean_28d"):
        out[f"item_{col}"] = out.groupby(level="item_nbr")[col].transform("mean")

    # 6) Targets: log1p(sales) on each of the next days
    if with_target:
        missing = future.difference(sales.columns)
        if len(missing):
            raise ValueError(f"Target dates not available in data: {missing[0].date()} ...")
        target = sales[future].values
        for h in range(n_horizons):
            out[f"y_d{h + 1}"] = target[:, h]

    out.insert(0, "cutoff", cutoff)
    return out.astype({c: "float32" for c in out.columns if c != "cutoff"})


def add_static(df: pd.DataFrame, items: pd.DataFrame,
               stores: pd.DataFrame) -> pd.DataFrame:
    """Add product and store attributes as integer-coded categorical features."""
    it = items.set_index("item_nbr")[["family", "class", "perishable"]].copy()
    it["family"] = it["family"].astype("category").cat.codes
    st = stores.set_index("store_nbr")[["city", "state", "type", "cluster"]].copy()
    st = st.rename(columns={"type": "store_type"})
    for c in ("city", "state", "store_type"):
        st[c] = st[c].astype("category").cat.codes

    df = df.reset_index()
    df = df.merge(it, left_on="item_nbr", right_index=True, how="left")
    df = df.merge(st, left_on="store_nbr", right_index=True, how="left")
    static = ["family", "class", "perishable", "city", "state", "store_type", "cluster"]
    return df.astype({c: "int16" for c in static})


def build_dataset(sales: pd.DataFrame, promo: pd.DataFrame, cutoffs,
                  items: pd.DataFrame, stores: pd.DataFrame,
                  with_target: bool = True) -> pd.DataFrame:
    """Stack the feature rows of several cutoffs into one training table."""
    parts = [make_features(sales, promo, c, with_target=with_target) for c in cutoffs]
    return add_static(pd.concat(parts), items, stores)