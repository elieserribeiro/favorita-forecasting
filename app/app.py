"""Painel de previsão de demanda — Corporación Favorita.

Run locally from the project root:  streamlit run app/app.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data import FAMILY_PT  # noqa: E402

DATA = Path(__file__).parent / "data"
REPO_URL = "https://github.com/elieserribeiro/favorita-forecasting"
COLORS = {"real": "#1f2937", "val": "#f59e0b", "fut": "#ea580c", "q1": "#fdba74", "q2": "#93c5fd"}

st.set_page_config(page_title="Previsão de Demanda · Favorita", page_icon="🛒", layout="wide")


# ---------------------------------------------------------------- data
@st.cache_data
def load_data():
    cube = pd.read_parquet(DATA / "loja_familia_dia.parquet")
    item_day = pd.read_parquet(DATA / "item_dia.parquet")
    items = pd.read_parquet(DATA / "itens.parquet")
    stores = pd.read_parquet(DATA / "lojas.parquet")
    cube["family"] = cube["family"].astype(str)
    return cube, item_day, items, stores


def periods(df: pd.DataFrame) -> dict:
    """Key dates derived from the data itself."""
    fut_dates = df.loc[df["prev_fut"].notna(), "date"]
    val_dates = df.loc[df["prev_val"].notna(), "date"]
    f0, v0 = fut_dates.min(), val_dates.min()
    return {
        "last_hist": df.loc[df["real"].notna(), "date"].max(),
        "val": (v0, val_dates.max()),
        "fut_q1": (f0, f0 + pd.Timedelta(days=14)),
        "fut_q2": (f0 + pd.Timedelta(days=15), fut_dates.max()),
        "val_q1": (v0, v0 + pd.Timedelta(days=14)),
        "val_q2": (v0 + pd.Timedelta(days=15), val_dates.max()),
    }


def between(s: pd.Series, period: tuple) -> pd.Series:
    return s.between(period[0], period[1])


def fmt_units(x: float) -> str:
    if x >= 1e6:
        return f"{x / 1e6:,.2f} mi".replace(",", "X").replace(".", ",").replace("X", ".")
    return f"{x:,.0f}".replace(",", ".")


def fmt_pct(x: float, sign: bool = True) -> str:
    if x is None or not np.isfinite(x):
        return "—"
    return (f"{x:+.1%}" if sign else f"{x:.1%}").replace(".", ",")


def ratio(a: float, b: float) -> float:
    """a / b, or NaN when there is nothing to compare against (b = 0)."""
    return a / b if b > 0 else np.nan


def delta_text(value: float, reference: float) -> str | None:
    r = ratio(value, reference)
    return None if np.isnan(r) else fmt_pct(r - 1) + " vs. ritmo recente"


def family_label(f: str) -> str:
    return f"{f} — {FAMILY_PT.get(f, '')}"


def forecast_chart(daily: pd.DataFrame, p: dict, title: str, start=None) -> go.Figure:
    if start is not None:
        daily = daily[daily.index >= start]
    fig = go.Figure()
    for (name, period, color) in (("1ª quinzena", p["fut_q1"], COLORS["q1"]),
                                  ("2ª quinzena", p["fut_q2"], COLORS["q2"])):
        fig.add_vrect(x0=period[0] - pd.Timedelta(hours=12), x1=period[1] + pd.Timedelta(hours=12),
                      fillcolor=color, opacity=0.18, line_width=0,
                      annotation_text=name, annotation_position="top left")
    fig.add_trace(go.Scatter(x=daily.index, y=daily["real"], name="Realizado",
                             line=dict(color=COLORS["real"], width=2)))
    fig.add_trace(go.Scatter(x=daily.index, y=daily["prev_val"], name="Previsto na validação",
                             line=dict(color=COLORS["val"], width=1.5, dash="dash")))
    fig.add_trace(go.Scatter(x=daily.index, y=daily["prev_fut"], name="Previsão oficial",
                             line=dict(color=COLORS["fut"], width=2.5)))
    fig.update_layout(title=title, height=430, hovermode="x unified",
                      yaxis_title="Unidades / dia", margin=dict(t=60, b=10),
                      legend=dict(orientation="h", y=-0.15),
                      xaxis=dict(tickformat="%d/%m", hoverformat="%d/%m/%Y"))
    return fig


def accuracy(df: pd.DataFrame, p: dict, keys: list[str]) -> pd.DataFrame:
    """WAPE, bias, shortage and excess of the validation, per fortnight."""
    v = df[between(df["date"], p["val"])].copy()
    v["Quinzena"] = np.where(between(v["date"], p["val_q1"]), "1ª quinzena", "2ª quinzena")
    tot = v.groupby(keys + ["Quinzena"], observed=True)[["real", "prev_val"]].sum()
    rows = []
    for q, g in tot.groupby(level="Quinzena"):
        real = g["real"].sum()
        rows.append({"Quinzena": q,
                     "WAPE": ratio((g["prev_val"] - g["real"]).abs().sum(), real),
                     "Viés": ratio(g["prev_val"].sum(), real) - 1,
                     "Falta": ratio((g["real"] - g["prev_val"]).clip(lower=0).sum(), real),
                     "Excesso": ratio((g["prev_val"] - g["real"]).clip(lower=0).sum(), real)})
    return pd.DataFrame(rows).set_index("Quinzena")


# ---------------------------------------------------------------- layout
cube, item_day, items, stores = load_data()
P = periods(cube)

st.title("🛒 Previsão de Demanda — Corporación Favorita")
st.caption(f"Rede de supermercados do Equador · 54 lojas · ~4 mil produtos · "
           f"previsão de {P['fut_q1'][0]:%d/%m} a {P['fut_q2'][1]:%d/%m/%Y} · "
           f"[código no GitHub]({REPO_URL})")

# Sidebar filters
with st.sidebar:
    st.header("🔎 Filtros")
    types = sorted(stores["type"].unique())
    sel_types = st.multiselect("Tipo de loja", types, default=types)
    st_f = stores[stores["type"].isin(sel_types)]
    cities = sorted(st_f["city"].unique())
    sel_cities = st.multiselect("Cidade", cities, placeholder="Todas as cidades")
    st_f = st_f[st_f["city"].isin(sel_cities or cities)].sort_values("store_nbr")
    store_opts = ["Todas"] + [f"Loja {r.store_nbr} · {r.city} · tipo {r.type}" for r in st_f.itertuples()]
    sel_store = st.selectbox("Loja", store_opts)
    store_ids = (st_f["store_nbr"].tolist() if sel_store == "Todas"
                 else [int(sel_store.split()[1])])

    families = sorted(cube["family"].unique())
    sel_fams = st.multiselect("Família de produtos", families, format_func=family_label,
                              placeholder="Todas as famílias")
    fams = sel_fams or families
    st.divider()
    st.caption("A **previsão oficial** cobre as duas quinzenas seguintes ao fim dos dados. "
               "O **previsto na validação** mostra como o modelo se saiu num período em que "
               "já sabíamos a resposta.")

sel = cube[cube["store_nbr"].isin(store_ids) & cube["family"].isin(fams)]
if sel.empty:
    st.warning("Nenhum dado para os filtros escolhidos.")
    st.stop()
daily = sel.groupby("date")[["real", "prev_val", "prev_fut"]].sum(min_count=1)

tab_prev, tab_prec, tab_item, tab_sobre = st.tabs(
    ["📈 Previsão", "🎯 Precisão do modelo", "🛒 Produto", "ℹ️ Sobre o projeto"])

# ---------------------------------------------------------------- tab 1
with tab_prev:
    q1 = daily.loc[between(daily.index.to_series(), P["fut_q1"]), "prev_fut"].sum()
    q2 = daily.loc[between(daily.index.to_series(), P["fut_q2"]), "prev_fut"].sum()
    last30 = daily.loc[daily.index > P["last_hist"] - pd.Timedelta(days=30), "real"].sum() / 2

    c1, c2, c3 = st.columns(3)
    c1.metric(f"Previsão 1ª quinzena ({P['fut_q1'][0]:%d/%m}–{P['fut_q1'][1]:%d/%m})",
              fmt_units(q1), delta_text(q1, last30), delta_color="off")
    c2.metric(f"Previsão 2ª quinzena ({P['fut_q2'][0]:%d/%m}–{P['fut_q2'][1]:%d/%m})",
              fmt_units(q2), delta_text(q2, last30), delta_color="off")
    c3.metric("Média quinzenal dos últimos 30 dias", fmt_units(last30))

    st.plotly_chart(forecast_chart(daily, P, "Histórico recente, validação e previsão",
                                   start=pd.Timestamp("2017-06-01")), width="stretch")

    st.subheader("Previsão por família de produtos")
    fam_tot = sel.assign(
        q1=np.where(between(sel["date"], P["fut_q1"]), sel["prev_fut"], 0),
        q2=np.where(between(sel["date"], P["fut_q2"]), sel["prev_fut"], 0),
        r30=np.where(sel["date"] > P["last_hist"] - pd.Timedelta(days=30), sel["real"], 0),
    ).groupby("family")[["q1", "q2", "r30"]].sum()
    fam_tot = fam_tot[fam_tot["r30"] > 0]   # only families sold in the last 30 days
    table = pd.DataFrame({
        "Família": [family_label(f) for f in fam_tot.index],
        "1ª quinzena (unid.)": fam_tot["q1"].values,
        "2ª quinzena (unid.)": fam_tot["q2"].values,
        "Variação 1ª quinz.": (fam_tot["q1"] / (fam_tot["r30"] / 2) - 1).values,
        "Variação 2ª quinz.": (fam_tot["q2"] / (fam_tot["r30"] / 2) - 1).values,
    }).sort_values("1ª quinzena (unid.)", ascending=False)
    st.dataframe(table, hide_index=True, width="stretch", column_config={
        "1ª quinzena (unid.)": st.column_config.NumberColumn(format="%,.0f"),
        "2ª quinzena (unid.)": st.column_config.NumberColumn(format="%,.0f"),
        "Variação 1ª quinz.": st.column_config.NumberColumn(format="percent"),
        "Variação 2ª quinz.": st.column_config.NumberColumn(format="percent"),
    })
    st.caption("Variação em relação à média quinzenal dos últimos 30 dias de histórico.")

    export = (sel.loc[sel["prev_fut"].notna(), ["date", "store_nbr", "family", "prev_fut"]]
                 .rename(columns={"date": "data", "store_nbr": "loja", "family": "familia",
                                  "prev_fut": "previsao_unidades"}))
    st.download_button("⬇️ Baixar previsão filtrada (CSV)",
                       export.to_csv(index=False).encode("utf-8"),
                       file_name="previsao_favorita.csv", mime="text/csv")

# ---------------------------------------------------------------- tab 2
with tab_prec:
    st.markdown(f"Desempenho do modelo no **período de validação** "
                f"({P['val'][0]:%d/%m} a {P['val'][1]:%d/%m/%Y}), um teste \"às cegas\": "
                "o modelo não conhecia esses dados ao ser treinado.")
    acc = accuracy(sel, P, ["store_nbr", "family"])
    for q, col in zip(acc.index, st.columns(len(acc))):
        with col:
            st.markdown(f"**{q}**")
            a, b = st.columns(2)
            a.metric("Erro (WAPE)", fmt_pct(acc.loc[q, "WAPE"], sign=False))
            b.metric("Viés", fmt_pct(acc.loc[q, "Viés"]))
            a.metric("Falta (ruptura)", fmt_pct(acc.loc[q, "Falta"], sign=False))
            b.metric("Excesso", fmt_pct(acc.loc[q, "Excesso"], sign=False))
    st.caption("Erros medidos no total da quinzena por **loja × família**. No nível de cada "
               "produto em cada loja o erro é maior (~30%); na compra por família na rede, "
               "menor (~6%). **Falta**: demanda não coberta pela previsão. "
               "**Excesso**: previsão acima da demanda.")

    val_daily = daily[between(daily.index.to_series(), P["val"])]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=val_daily.index, y=val_daily["real"], name="Realizado",
                             line=dict(color=COLORS["real"], width=2)))
    fig.add_trace(go.Scatter(x=val_daily.index, y=val_daily["prev_val"], name="Previsto",
                             line=dict(color=COLORS["val"], width=2)))
    fig.update_layout(title="Previsto × realizado na validação", height=380,
                      hovermode="x unified", yaxis_title="Unidades / dia",
                      legend=dict(orientation="h", y=-0.15), margin=dict(t=60, b=10),
                      xaxis=dict(tickformat="%d/%m", hoverformat="%d/%m/%Y"))
    st.plotly_chart(fig, width="stretch")

    st.subheader("Erro por família")
    rows = []
    for f, g in sel.groupby("family"):
        v = g[between(g["date"], P["val"])]
        if v["real"].sum() <= 0:          # family with no sales in the selection
            continue
        a = accuracy(g, P, ["store_nbr"])
        rows.append({"Família": family_label(f), "Volume na validação": v["real"].sum(),
                     "WAPE 1ª quinz.": a["WAPE"].iloc[0], "WAPE 2ª quinz.": a["WAPE"].iloc[-1],
                     "Viés (30 dias)": ratio(v["prev_val"].sum(), v["real"].sum()) - 1})
    if rows:
        fam_acc = pd.DataFrame(rows).sort_values("Volume na validação", ascending=False)
        st.dataframe(fam_acc, hide_index=True, width="stretch", column_config={
            "Volume na validação": st.column_config.NumberColumn(format="%,.0f"),
            "WAPE 1ª quinz.": st.column_config.NumberColumn(format="percent"),
            "WAPE 2ª quinz.": st.column_config.NumberColumn(format="percent"),
            "Viés (30 dias)": st.column_config.NumberColumn(format="percent"),
        })
    else:
        st.info("Não houve vendas no período de validação para a seleção atual.")

# ---------------------------------------------------------------- tab 3
with tab_item:
    st.markdown("Consulta de um produto específico, somando **todas as lojas da rede**.")
    c1, c2 = st.columns([1, 2])
    fam_item = c1.selectbox("Família", families, format_func=family_label,
                            index=families.index("GROCERY I") if "GROCERY I" in families else 0)
    cand = (items[items["family"] == fam_item]
            .sort_values("volume_90d", ascending=False))
    labels = {r.item_nbr: (f"Item {r.item_nbr} · classe {r.abc}"
                           f"{' · perecível' if r.perishable == 1 else ''}"
                           f" · {fmt_units(r.volume_90d)} unid. em 90 dias")
              for r in cand.itertuples()}
    item = c2.selectbox("Produto (ordenado por volume vendido)", list(labels),
                        format_func=labels.get)
    idf = item_day[item_day["item_nbr"] == item].set_index("date").sort_index()

    m1, m2, m3 = st.columns(3)
    iq1 = idf.loc[between(idf.index.to_series(), P["fut_q1"]), "prev_fut"].sum()
    iq2 = idf.loc[between(idf.index.to_series(), P["fut_q2"]), "prev_fut"].sum()
    ir30 = idf.loc[idf.index > P["last_hist"] - pd.Timedelta(days=30), "real"].sum() / 2
    m1.metric("Previsão 1ª quinzena", fmt_units(iq1), delta_text(iq1, ir30), delta_color="off")
    m2.metric("Previsão 2ª quinzena", fmt_units(iq2), delta_text(iq2, ir30), delta_color="off")
    m3.metric("Média quinzenal recente", fmt_units(ir30))
    st.plotly_chart(forecast_chart(idf, P, f"Item {item} — rede inteira"),
                    width="stretch")

# ---------------------------------------------------------------- tab 4
with tab_sobre:
    st.markdown(f"""
### O projeto
Previsão de demanda para a **Corporación Favorita**, rede de supermercados do Equador,
usando o dataset público da competição *Corporación Favorita Grocery Sales Forecasting*
(Kaggle). O objetivo é apoiar decisões de **compras, estoque, distribuição e redução de
perdas** com previsões para as **duas quinzenas seguintes**.

### Como funciona
| Etapa | Descrição |
|---|---|
| **Dados** | 23,8 milhões de vendas diárias de 2017 (54 lojas × ~4 mil produtos) |
| **Modelo** | 30 modelos LightGBM — um para cada dia à frente (*previsão direta*) |
| **Variáveis** | histórico recente, padrão semanal, intermitência, promoções e calendário |
| **Validação** | teste "às cegas" de 17/jul a 15/ago/2017 |
| **Calibração** | ajuste do volume previsto, reduzindo o viés de −20,7% para −2,6% |

### Resultados
- **9% menos erro** do que os métodos simples (média das últimas semanas).
- Erro de **~6%** na compra por família de produtos para a quinzena.
- Previsão: **11,9 milhões** de unidades (16–30/ago) e **12,8 milhões** (31/ago–14/set).

### Principais padrões da demanda
- Sábado e domingo concentram **~35%** das vendas da semana.
- O **início do mês** (pagamento de salários) vende 13–28% acima da média.
- **3 famílias** respondem por 67% do volume; **Quito** por 51%.

[📂 Código completo, notebooks e análises no GitHub]({REPO_URL})
""")