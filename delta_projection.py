import math
import time
from datetime import datetime, timezone

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


st.set_page_config(page_title="BTC Delta Projection", layout="wide")
st.title("BTC Delta Projection (ATM Mark IV)")

currency = st.selectbox("Currency", ["BTC", "ETH"], index=0)
instruments = load_option_instruments(currency)

expiries = sorted({x["expiration_timestamp"] for x in instruments})
expiry_labels = {
    ts: datetime.fromtimestamp(ts / 1000, tz=timezone.utc).strftime("%d %b %Y")
    for ts in expiries
}
expiry_ts = st.selectbox("Expiry", expiries, format_func=lambda ts: expiry_labels[ts])

filtered_by_expiry = [x for x in instruments if x["expiration_timestamp"] == expiry_ts]
strikes = sorted({int(x["strike"]) for x in filtered_by_expiry})

sample_calls = [x for x in filtered_by_expiry if x["option_type"] == "call"]
if not sample_calls:
    st.error("No call options found for the selected expiry.")
    st.stop()

sample_ticker = get_ticker(sample_calls[0]["instrument_name"])
F_live = float(sample_ticker.get("underlying_price") or 0.0)
if F_live <= 0:
    F_live = float(get_index_price(currency))

atm_strike = min(strikes, key=lambda k: abs(k - int(round(F_live)))) if strikes else None

atm_call_iv_pct = None
if atm_strike is not None:
    atm_call_candidates = [
        x for x in filtered_by_expiry if x["option_type"] == "call" and int(x["strike"]) == int(atm_strike)
    ]
    if atm_call_candidates:
        atm_call_ticker = get_ticker(atm_call_candidates[0]["instrument_name"])
        atm_call_iv_pct = atm_call_ticker.get("mark_iv", None)

index_price = get_index_price(currency)
st.caption("Live data")
m1, m2, m3 = st.columns(3)
m1.metric(f"{currency} Index ({currency.lower()}_usd)", f"{index_price:,.2f}")
m2.metric("Underlying future", f"{F_live:,.2f}")
if atm_call_iv_pct is not None:
    m3.metric("ATM Call Mark IV (%)", f"{float(atm_call_iv_pct):.2f}")

st.divider()

if atm_call_iv_pct is None:
    st.warning("ATM call Mark IV unavailable for this expiry; select another expiry.")
    st.stop()

sigma_atm = float(atm_call_iv_pct) / 100.0
deltas = [0.5, 0.4, 0.3, 0.2, 0.1, 0.6, 0.7, 0.8, 0.9, 1.0]

start_year = datetime.now(tz=timezone.utc).year
years = [start_year + i for i in range(1, 11)]
data = {}

for delta in deltas:
    p = min(0.999999, max(0.000001, float(delta)))
    d1 = inv_norm_cdf(p)
    series = []
    for i in range(1, 11):
        T = float(i)
        strike = F_live * math.exp(-d1 * sigma_atm * math.sqrt(T) + 0.5 * sigma_atm * sigma_atm * T)
        series.append(strike)
    data[f"Δ {delta:g}"] = series

df = pd.DataFrame(data, index=years).round(0).astype(int)
st.subheader("Projected BTC Price by Delta (Next 10 Years)")

df_long = df.reset_index().melt(id_vars="index", var_name="Delta", value_name="Price")
df_long.rename(columns={"index": "Year"}, inplace=True)

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
        ],
        opacity=alt.condition(hover, alt.value(0.6), alt.value(0)),
    )
    .transform_filter(hover)
)

chart = (line + points + rule)
st.altair_chart(chart, use_container_width=True)
