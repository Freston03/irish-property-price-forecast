import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Irish Property Price Forecast", layout="wide")


@st.cache_data
def load():
    hist = pd.read_csv("data/rppi_clean.csv", index_col=0, parse_dates=True).iloc[:, 0]
    fc = pd.read_csv("data/forecast.csv", index_col=0, parse_dates=True)
    by_win = pd.read_csv("data/mape_by_window.csv", index_col=0)
    nested = pd.read_csv("data/sarima_nested_windows.csv", index_col=0)
    return hist, fc, by_win, nested


hist, fc, by_win, nested = load()

st.title("Irish Residential Property Price Forecast")
st.caption("CSO Residential Property Price Index (HPM09) · SARIMA model · 12-month outlook")

tab1, tab2, tab3 = st.tabs(["Forecast", "Model comparison", "Limitations"])

# ---------- Tab 1: forecast ----------
with tab1:
    c1, c2, c3 = st.columns(3)
    years = c1.slider("Years of history to show", 2, 20, 5)
    months = c2.slider("Forecast horizon (months)", 1, 12, 12)
    mult = c3.slider(
        "Uncertainty band width", 1.0, 2.0, 1.25, 0.05,
        help="1.0 = the model's own 95% band. 1.25 reached about 95% coverage in backtesting.",
    )

    f = fc.iloc[:months].copy()
    half = (f["upper_model"] - f["lower_model"]) / 2
    lo, hi = f["forecast"] - half * mult, f["forecast"] + half * mult
    h = hist[hist.index >= hist.index[-1] - pd.DateOffset(years=years)]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=h.index, y=h.values, name="History",
                             line=dict(color="black")))
    fig.add_trace(go.Scatter(
        x=list(f.index) + list(f.index[::-1]),
        y=list(hi) + list(lo[::-1]),
        fill="toself", fillcolor="rgba(31,119,180,0.2)",
        line=dict(width=0), name="Uncertainty band", hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(x=f.index, y=f["forecast"], name="Forecast",
                             line=dict(color="rgb(31,119,180)", dash="dash")))
    fig.update_layout(yaxis_title="Index points", hovermode="x unified",
                      margin=dict(l=0, r=0, t=10, b=0), height=450)
    st.plotly_chart(fig, use_container_width=True)

    last, end = hist.iloc[-1], f["forecast"].iloc[-1]
    m1, m2, m3 = st.columns(3)
    m1.metric("Latest index", f"{last:.1f}")
    m2.metric(f"Forecast in {months} months", f"{end:.1f}", f"{(end / last - 1) * 100:+.1f}%")
    m3.metric("Plausible range", f"{lo.iloc[-1]:.1f} – {hi.iloc[-1]:.1f}")
    st.info(
        "This forecast assumes the market keeps behaving as it has recently. "
        "It cannot anticipate policy changes, interest rate shocks or recessions."
    )

# ---------- Tab 2: model comparison ----------
with tab2:
    st.subheader("Average error across 8 rolling backtest windows (MAPE %, lower is better)")
    avg = by_win.mean().sort_values().round(2)
    st.bar_chart(avg)
    st.dataframe(by_win.round(2), use_container_width=True)
    st.markdown(
        f"**SARIMA** was the best model overall. **Prophet** looked best over the most recent "
        f"windows but fell behind once older windows, including turning points, were added. "
        f"With nested tuning, SARIMA's average MAPE was **{nested['MAPE %'].mean():.2f}%** "
        f"and its 95% band covered **{nested['95% band coverage %'].mean():.0f}%** of months "
        f"before recalibration."
    )
    st.dataframe(nested, use_container_width=True)

# ---------- Tab 3: limitations ----------
with tab3:
    st.markdown(
        """
- **Turning points are the weak spot.** Errors were small in calm periods (about 0.3–2.5%)
  but reached 5–6% in the windows starting September 2020 and September 2023.
- **Statistical, not causal.** The model learns from past patterns only. It does not know
  about interest rates, housing supply, or government policy.
- **National index.** This covers Ireland overall, not individual counties or neighbourhoods.
- **Small test set.** Eight 12-month windows is useful but not conclusive.
- **Not financial advice.**
"""
    )

st.caption(
    "Data: Central Statistics Office Ireland, Residential Property Price Index (HPM09), "
    "licensed CC BY 4.0."
)
