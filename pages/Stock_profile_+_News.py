# pages/Stock_profile.py
import streamlit as st
import plotly.graph_objects as go
import streamlit as st
import streamlit as st
try:
    from utils import yf_history_1y, add_indicators, yf_company_info, human_market_cap, news_for_ticker
except Exception as e:
    st.error("Import failed. See details below.")
    st.exception(e)
    st.stop()
key = st.secrets["OPENAI_API_KEY"]
key = st.secrets["TAVILY_API"]
from utils import yf_history_1y, add_indicators, yf_company_info, human_market_cap, news_for_ticker
from tools import summarize_news_items, DEFAULT_MODEL

st.set_page_config(page_title="Stock Profile", page_icon="📈", layout="wide")
st.title("📊 Stock Profile")

# -------- Sidebar: one input + Apply --------
if "selected_ticker" not in st.session_state:
    st.session_state.selected_ticker = None

with st.sidebar.form("find_stock_form"):
    q = st.text_input("Ticker or company name", value="", placeholder="e.g., AMAT or Applied Materials")
    applied = st.form_submit_button("Apply", use_container_width=True)

if applied and q.strip():
    # keep it simple: use what user typed (yfinance needs an actual ticker!)
    st.session_state.selected_ticker = q.strip().upper()

ticker = st.session_state.selected_ticker
if not ticker:
    st.info("Enter a ticker or company name and click **Apply**.")
    st.stop()

# -------- Charts (1Y, fixed) --------
@st.cache_data(ttl=60*30, show_spinner=False)
def _load_px(t: str):
    return yf_history_1y(t)

px = _load_px(ticker)
if px.empty:
    st.error("Could not load price data (check the ticker symbol).")
    st.stop()

px = add_indicators(px)
info = yf_company_info(ticker)

# KPIs
curr = float(px["Close"].iloc[-1])
prev = float(px["Close"].iloc[-2]) if len(px) > 1 else curr
chg = (curr - prev) / prev * 100 if prev else 0
k1, k2, k3, k4 = st.columns(4)
k1.metric("Price", f"${curr:,.2f}", f"{chg:+.2f}%")
k2.metric("High (1Y)", f"${px['High'].max():,.2f}")
k3.metric("Low (1Y)", f"${px['Low'].min():,.2f}")
k4.metric("Market Cap", human_market_cap(info.get("market_cap")))

# Charts
c1, c2 = st.columns([2, 1], gap="large")
with c1:
    st.subheader(f"{ticker} — Price (Candles) + MAs")
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=px["date"], open=px["Open"], high=px["High"],
                                 low=px["Low"], close=px["Close"], name="Price"))
    for col, name in [("MA20", "MA 20"), ("MA50", "MA 50"), ("MA200", "MA 200")]:
        if px[col].notna().sum():
            fig.add_trace(go.Scatter(x=px["date"], y=px[col], mode="lines", name=name))
    fig.update_layout(margin=dict(l=10, r=10, t=30, b=10), height=520, yaxis_title="USD")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("RSI (14)")
    rsi = go.Figure()
    rsi.add_trace(go.Scatter(x=px["date"], y=px["RSI14"], mode="lines", name="RSI14"))
    rsi.add_hrect(y0=70, y1=70, line_width=0)
    rsi.add_hrect(y0=30, y1=30, line_width=0)
    rsi.update_yaxes(range=[0, 100])
    rsi.update_layout(margin=dict(l=10, r=10, t=20, b=10), height=220)
    st.plotly_chart(rsi, use_container_width=True)

with c2:
    st.subheader("Close")
    line = go.Figure()
    line.add_trace(go.Scatter(x=px["date"], y=px["Close"], mode="lines"))
    line.update_layout(margin=dict(l=10, r=10, t=20, b=10), height=260, yaxis_title="USD")
    st.plotly_chart(line, use_container_width=True)

    st.subheader("Volume")
    vol = go.Figure()
    vol.add_trace(go.Bar(x=px["date"], y=px["Volume"]))
    vol.update_layout(margin=dict(l=10, r=10, t=20, b=10), height=260, yaxis_title="Shares")
    st.plotly_chart(vol, use_container_width=True)

# Company snapshot
with st.expander("ℹ️ Company snapshot", expanded=False):
    st.write(f"**{info.get('name', ticker)}**")
    st.write(f"Sector: {info.get('sector','—')} · Industry: {info.get('industry','—')}")
    if info.get("summary"):
        st.write(info["summary"])

# -------- News + summaries (Tavily basic) --------
st.markdown("## 📰 Latest News & Summaries")

@st.cache_data(ttl=60*20, show_spinner=False)
def _fetch_news(t: str):
    return news_for_ticker(t, max_results=3)

with st.spinner("Fetching news & generating summaries…"):
    items = _fetch_news(ticker)          # Tavily (basic)
    if not items:
        st.caption("No recent news found.")
    else:
        summaries = summarize_news_items(items, max_chars=320, model=DEFAULT_MODEL)
        for s in summaries:
            st.markdown(f"- **[{s['title']}]({s['url']})**")
            # hide empty / 'skip' responses automatically
            if s["summary"] and not s["summary"].strip().lower().startswith("skip"):
                st.write(s["summary"])
        st.divider()