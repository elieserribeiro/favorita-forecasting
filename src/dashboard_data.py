"""Build the light datasets used by the Streamlit dashboard (app/data).

The full prediction tables have ~5 million rows each and stay out of GitHub.
The dashboard uses two aggregated views (a few MB in total):
- store x family x day : 2017 history, validation forecast and future forecast
- item x day (whole chain) : last 90 days of history, validation and future forecast

Run from the project root:  python -m src.dashboard_data
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
OUT = ROOT / "app" / "data"
ITEM_HISTORY_DAYS = 90


def _with_family(df: pd.DataFrame, family: pd.Series) -> pd.DataFrame:
    df = df.copy()
    df["family"] = df["item_nbr"].map(family).astype("category")
    return df


def build(proc: Path = PROC, out: Path = OUT) -> None:
    out.mkdir(parents=True, exist_ok=True)
    items = pd.read_parquet(proc / "items.parquet")
    stores = pd.read_parquet(proc / "stores.parquet")
    train = pd.read_parquet(proc / "train_2017.parquet",
                            columns=["date", "store_nbr", "item_nbr", "unit_sales"])
    val = pd.read_parquet(proc / "painel_validacao.parquet",
                          columns=["date", "store_nbr", "item_nbr", "previsto"])
    fut = pd.read_parquet(proc / "previsao_futura.parquet",
                          columns=["date", "store_nbr", "item_nbr", "previsto", "abc"])
    family = items.set_index("item_nbr")["family"]
    last_hist = train["date"].max()

    # 1) Store x family x day
    keys = ["store_nbr", "family", "date"]
    cube = pd.concat([
        _with_family(train, family).groupby(keys, observed=True)["unit_sales"].sum().rename("real"),
        _with_family(val, family).groupby(keys, observed=True)["previsto"].sum().rename("prev_val"),
        _with_family(fut, family).groupby(keys, observed=True)["previsto"].sum().rename("prev_fut"),
    ], axis=1).reset_index()
    hist_days = cube["date"] <= last_hist
    cube.loc[hist_days, "real"] = cube.loc[hist_days, "real"].fillna(0)
    cube = cube.astype({"store_nbr": "int8", "real": "float32",
                        "prev_val": "float32", "prev_fut": "float32"})
    cube.to_parquet(out / "loja_familia_dia.parquet", index=False)

    # 2) Item x day for the whole chain (sum of all stores)
    start = last_hist - pd.Timedelta(days=ITEM_HISTORY_DAYS - 1)
    recent = train[train["date"] >= start]
    item_day = pd.concat([
        recent.groupby(["item_nbr", "date"])["unit_sales"].sum().rename("real"),
        val.groupby(["item_nbr", "date"])["previsto"].sum().rename("prev_val"),
        fut.groupby(["item_nbr", "date"])["previsto"].sum().rename("prev_fut"),
    ], axis=1).reset_index()
    item_days = item_day["date"] <= last_hist
    item_day.loc[item_days, "real"] = item_day.loc[item_days, "real"].fillna(0)
    item_day = item_day.astype({"item_nbr": "int32", "real": "float32",
                                "prev_val": "float32", "prev_fut": "float32"})
    item_day.to_parquet(out / "item_dia.parquet", index=False)

    # 3) Item and store attributes
    abc = fut.drop_duplicates("item_nbr").set_index("item_nbr")["abc"].astype(str)
    volume = recent.groupby("item_nbr")["unit_sales"].sum()
    meta = items[items["item_nbr"].isin(item_day["item_nbr"].unique())].copy()
    meta["abc"] = meta["item_nbr"].map(abc).fillna("C")
    meta["volume_90d"] = meta["item_nbr"].map(volume).fillna(0).astype("float32")
    meta.to_parquet(out / "itens.parquet", index=False)
    stores.to_parquet(out / "lojas.parquet", index=False)

    for f in sorted(out.glob("*.parquet")):
        print(f"{f.name:28s} {f.stat().st_size / 1e6:6.1f} MB")


if __name__ == "__main__":
    build()