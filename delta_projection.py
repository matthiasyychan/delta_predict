import math
import time
from datetime import datetime, timezone
from io import BytesIO

import altair as alt
import pandas as pd
import requests
import streamlit as st

DERIBIT_HTTP = "https://www.deribit.com/api/v2"  # switch to testnet if needed


def rpc(method: str, params: dict | None = None, timeout=10):
    payload = {"jsonrpc": "2.0", "id": int(time.time() * 1000), "method": method, "params": params or {}}
    r = requests.post(DERIBIT_HTTP, json=payload, timeout=timeout)
    if not r.ok:
        raise RuntimeError(f"HTTP {r.status_code} from Deribit: {r.text}")
    data = r.json()
    if "error" in data:
        raise RuntimeError(f"Deribit error: {data['error']}")
    return data["result"]


@st.cache_data(ttl=3)
def get_ticker(instr: str):
    return rpc("public/ticker", {"instrument_name": instr})


@st.cache_data(ttl=3)
def get_index_price(currency: str):
    index_name = f"{currency.lower()}_usd"
    return rpc("public/get_index_price", {"index_name": index_name})["index_price"]


@st.cache_data(ttl=60)
def load_option_instruments(currency: str):
    inst = rpc("public/get_instruments", {"currency": currency.upper(), "kind": "option", "expired": False})
    out = []
    for x in inst:
        out.append(
            {
                "instrument_name": x["instrument_name"],
                "expiration_timestamp": x["expiration_timestamp"],
                "strike": x["strike"],
                "option_type": x["option_type"],  # "call" / "put"
            }
        )
    out.sort(key=lambda z: (z["expiration_timestamp"], z["strike"], z["option_type"]))
    return out


def inv_norm_cdf(p: float) -> float:
    """Approximate inverse CDF for standard normal (Acklam)."""
    if p <= 0.0:
        return -1e9
    if p >= 1.0:
        return 1e9

    a = [
        -3.969683028665376e01,
        2.209460984245205e02,
        -2.759285104469687e02,
        1.383577518672690e02,
        -3.066479806614716e01,
        2.506628277459239e00,
    ]
    b = [
        -5.447609879822406e01,
        1.615858368580409e02,
        -1.556989798598866e02,
        6.680131188771972e01,
        -1.328068155288572e01,
    ]
    c = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e00,
        -2.549732539343734e00,
        4.374664141464968e00,
        2.938163982698783e00,
    ]
    d = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e00,
        3.754408661907416e00,
    ]

    plow = 0.02425
    phigh = 1 - plow

    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (
            (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
        )
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(
            (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
        )

    q = p - 0.5
    r = q * q
    return (
        (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
        / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
    )


def get_atm_call_iv_for_expiry(expiry_ts, instruments, currency, index_price):
    filtered = [x for x in instruments if x["expiration_timestamp"] == expiry_ts]
    strikes = sorted({int(x["strike"]) for x in filtered})
    sample_calls = [x for x in filtered if x["option_type"] == "call"]
    if not sample_calls:
        return None, None

    sample_ticker = get_ticker(sample_calls[0]["instrument_name"])
    F_live = float(sample_ticker.get("underlying_price") or 0.0)
    if F_live <= 0:
        F_live = float(index_price)

    if not strikes:
        return None, F_live

    atm_strike = min(strikes, key=lambda k: abs(k - int(round(F_live))))
    atm_call_candidates = [
        x for x in filtered if x["option_type"] == "call" and int(x["strike"]) == int(atm_strike)
    ]
    if not atm_call_candidates:
        return None, F_live

    atm_call_ticker = get_ticker(atm_call_candidates[0]["instrument_name"])
    return atm_call_ticker.get("mark_iv", None), F_live


st.set_page_config(page_title="BTC Delta Projection", layout="wide")
st.title("BTC Delta Projection (ATM Mark IV)")

currency = st.selectbox("Currency", ["BTC", "ETH"], index=0)
instruments = load_option_instruments(currency)
index_price = get_index_price(currency)

expiries = sorted({x["expiration_timestamp"] for x in instruments})

# Only keep the last expiry in the current month
now = datetime.now(tz=timezone.utc)
current_month_expiries = [
    ts
    for ts in expiries
    if datetime.fromtimestamp(ts / 1000, tz=timezone.utc).year == now.year
    and datetime.fromtimestamp(ts / 1000, tz=timezone.utc).month == now.month
]
if current_month_expiries:
    last_current_month = max(current_month_expiries)
    expiries = [ts for ts in expiries if ts not in current_month_expiries or ts == last_current_month]
expiry_labels = {
    ts: datetime.fromtimestamp(ts / 1000, tz=timezone.utc).strftime("%d %b %Y")
    for ts in expiries
}
avg_key = "__AVG__"
expiry_options = [avg_key] + expiries
expiry_sel = st.selectbox(
    "Expiry",
    expiry_options,
    format_func=lambda ts: "Average (all expiries)" if ts == avg_key else expiry_labels[ts],
)

decay_options = [0, 2.5, 5, 7.5, 10, 12.5, 15, 20]
decay_pct = st.selectbox(
    "Volatility decrement per year (%)",
    decay_options,
    index=2,
    format_func=lambda v: f"{v:g}%",
)

atm_call_iv_pct = None
F_live = float(index_price)
using_average = expiry_sel == avg_key

if using_average:
    iv_values = []
    fwd_values = []
    for ts in expiries:
        iv, fwd = get_atm_call_iv_for_expiry(ts, instruments, currency, index_price)
        if iv is not None:
            iv_values.append(float(iv))
        if fwd is not None:
            fwd_values.append(float(fwd))

    if iv_values:
        atm_call_iv_pct = sum(iv_values) / len(iv_values)
    if fwd_values:
        F_live = sum(fwd_values) / len(fwd_values)
else:
    expiry_ts = expiry_sel
    filtered_by_expiry = [x for x in instruments if x["expiration_timestamp"] == expiry_ts]
    strikes = sorted({int(x["strike"]) for x in filtered_by_expiry})

    sample_calls = [x for x in filtered_by_expiry if x["option_type"] == "call"]
    if not sample_calls:
        st.error("No call options found for the selected expiry.")
        st.stop()

    sample_ticker = get_ticker(sample_calls[0]["instrument_name"])
    F_live = float(sample_ticker.get("underlying_price") or 0.0)
    if F_live <= 0:
        F_live = float(index_price)

    atm_strike = min(strikes, key=lambda k: abs(k - int(round(F_live)))) if strikes else None
    if atm_strike is not None:
        atm_call_candidates = [
            x for x in filtered_by_expiry if x["option_type"] == "call" and int(x["strike"]) == int(atm_strike)
        ]
        if atm_call_candidates:
            atm_call_ticker = get_ticker(atm_call_candidates[0]["instrument_name"])
            atm_call_iv_pct = atm_call_ticker.get("mark_iv", None)
st.caption("Live data")
m1, m2, m3 = st.columns(3)
m1.metric(f"{currency} Index ({currency.lower()}_usd)", f"{index_price:,.2f}")
m2.metric("Underlying future", f"{F_live:,.2f}")
if atm_call_iv_pct is not None:
    iv_label = "Avg ATM Call Mark IV (%)" if using_average else "ATM Call Mark IV (%)"
    m3.metric(iv_label, f"{float(atm_call_iv_pct):.2f}")

st.divider()

if atm_call_iv_pct is None:
    st.warning("ATM call Mark IV unavailable for this selection; choose a different expiry.")
    st.stop()

sigma_atm = float(atm_call_iv_pct) / 100.0
target_vix_pct = 17.28
target_sigma = target_vix_pct / 100.0
decay_rate = float(decay_pct) / 100.0
deltas = [0.5, 0.4, 0.3, 0.2, 0.1, 0.6, 0.7, 0.8, 0.9, 0.999]

start_year = datetime.now(tz=timezone.utc).year
years = [start_year + i for i in range(1, 11)]
data = {}
year_sigmas = {}

for delta in deltas:
    p = min(0.999999, max(0.000001, float(delta)))
    d1 = inv_norm_cdf(p)
    series = []
    for i in range(1, 11):
        T = float(i)
        sigma_year = max(target_sigma, sigma_atm * ((1 - decay_rate) ** i))
        year_sigmas[start_year + i] = sigma_year
        strike = F_live * math.exp(-d1 * sigma_year * math.sqrt(T) + 0.5 * sigma_year * sigma_year * T)
        series.append(strike)
    data[f"Δ {delta:g}"] = series

df = pd.DataFrame(data, index=years).round(0).astype(int)
st.subheader("Projected BTC Price by Delta (Next 10 Years)")

df_long = df.reset_index().melt(id_vars="index", var_name="Delta", value_name="Price")
df_long.rename(columns={"index": "Year"}, inplace=True)
df_long["Multiple"] = df_long["Price"] / float(index_price)
df_long["YearsOut"] = df_long["Year"].astype(int) - start_year
df_long["APR"] = (df_long["Multiple"] ** (1 / df_long["YearsOut"])) - 1
df_long["Volatility"] = df_long["Year"].map(lambda y: year_sigmas.get(int(y), sigma_atm)) * 100.0

base = alt.Chart(df_long).encode(
    x=alt.X("Year:O", title="Year"),
    y=alt.Y("Price:Q", title="BTC Price (USD)", axis=alt.Axis(format="~s")),
    color=alt.Color("Delta:N", title="Delta"),
)

hover = alt.selection_point(fields=["Year", "Delta"], nearest=True, on="mouseover", empty="none")

line = base.mark_line()

points = (
    base.mark_point(size=80)
    .encode(
        opacity=alt.condition(hover, alt.value(1), alt.value(0)),
        tooltip=[
            alt.Tooltip("Year:O"),
            alt.Tooltip("Delta:N"),
            alt.Tooltip("Price:Q", format=",.0f"),
            alt.Tooltip("Volatility:Q", format=".2f", title="Volatility (%)"),
            alt.Tooltip("APR:Q", format=".2%", title="APR"),
            alt.Tooltip("Multiple:Q", format=".2f", title="Multiple (x)"),
        ],
    )
    .add_params(hover)
)

rule = (
    alt.Chart(df_long)
    .mark_rule(color="#999")
    .encode(
        x="Year:O",
        tooltip=[
            alt.Tooltip("Year:O"),
            alt.Tooltip("Delta:N"),
            alt.Tooltip("Price:Q", format=",.0f"),
            alt.Tooltip("Volatility:Q", format=".2f", title="Volatility (%)"),
            alt.Tooltip("APR:Q", format=".2%", title="APR"),
            alt.Tooltip("Multiple:Q", format=".2f", title="Multiple (x)"),
        ],
        opacity=alt.condition(hover, alt.value(0.6), alt.value(0)),
    )
    .transform_filter(hover)
)

chart = (line + points + rule)
st.altair_chart(chart, use_container_width=True)

# Download chart data as Excel (deltas ordered 0.1 -> 1.0)
excel_buffer = BytesIO()
ordered_cols = sorted(
    df.columns,
    key=lambda c: float(c.replace("Δ", "").strip()),
)
df_excel = df[ordered_cols]
with pd.ExcelWriter(excel_buffer, engine="xlsxwriter") as writer:
    df_excel.to_excel(writer, sheet_name="Delta Projection")
    workbook = writer.book
    worksheet = writer.sheets["Delta Projection"]
    num_format = workbook.add_format({"num_format": "#,##0"})
    worksheet.set_column(1, 1 + len(df_excel.columns), 14, num_format)
    # Export tooltip data
    tooltip_cols = ["Year", "Delta", "Price", "Volatility", "APR", "Multiple"]
    tooltip_df = df_long[tooltip_cols].copy()
    tooltip_df["DeltaNum"] = tooltip_df["Delta"].str.replace("Δ", "").str.strip().astype(float)
    tooltip_df = tooltip_df.sort_values(["DeltaNum", "Year"]).drop(columns=["DeltaNum"])
    tooltip_df.to_excel(writer, sheet_name="Tooltip Data", index=False)
    tooltip_ws = writer.sheets["Tooltip Data"]
    tooltip_ws.set_column(0, 0, 8)
    tooltip_ws.set_column(1, 1, 8)
    tooltip_ws.set_column(2, 2, 14, num_format)
    tooltip_ws.set_column(3, 3, 14, workbook.add_format({"num_format": "0.00"}))
    tooltip_ws.set_column(4, 4, 14, workbook.add_format({"num_format": "0.00%"}))
    tooltip_ws.set_column(5, 5, 14, workbook.add_format({"num_format": "0.00"}))
excel_buffer.seek(0)
st.download_button(
    label="Download chart data (Excel)",
    data=excel_buffer,
    file_name="delta_projection.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)
