# pages/Stock_profile.py
from datetime import date, timedelta
import plotly.graph_objects as go
import streamlit as st

from utils import (
    yf_get_history, add_indicators, yf_company_info, human_market_cap,
    get_top_news_for_tickers, resolve_symbol
)
from tools import summarize_news_items, DEFAULT_NEWS_PROMPT, DEFAULT_MODEL

st.set_page_config(page_title="Stock Profile", page_icon="📈", layout="wide")
st.title("📊 Stock Profile")

# ---------------- Sidebar: single search (with Apply) ----------------
st.sidebar.header("🔎 Find a stock")

# keep last confirmed ticker
if "selected_ticker" not in st.session_state:
    st.session_state.selected_ticker = None

with st.sidebar.form(key="sp_find_stock_form", clear_on_submit=False):
    search_query = st.text_input(
        "Type a company name or ticker",
        value="",
        placeholder="e.g., Applied Materials or AMAT",
        key="sp_search",
    )

    @st.cache_data(ttl=60, show_spinner=False)
    def _suggest(q: str):
        return resolve_symbol(q)

    suggestions = _suggest(search_query) if search_query.strip() else []
    picked_symbol = None

    if suggestions:
        labels = [
            f"{s['symbol']} — {s['shortname'] or s['longname'] or '—'} ({s['exchDisp']})"
            for s in suggestions
        ]
        picked_idx = st.radio(
            "Suggestions",
            options=list(range(len(suggestions))),
            format_func=lambda i: labels[i],
            key="sp_pick",
        )
        picked_symbol = suggestions[picked_idx]["symbol"]
    else:
        st.caption("Start typing to see suggestions…")

    applied = st.form_submit_button("Apply", use_container_width=True)

if applied and picked_symbol:
    st.session_state.selected_ticker = picked_symbol

ticker = st.session_state.selected_ticker

if ticker:
    st.sidebar.success(f"Selected: {ticker}")
else:
    st.sidebar.info("No ticker selected yet.")

# ---------------- Guard: need a chosen ticker ----------------
if not ticker:
    st.info("Pick a stock in the sidebar and click **Apply**.")
    st.stop()

# ---------------- Fixed defaults (kept simple) ----------------
MONTHS_BACK = 12          # 1Y chart
INTERVAL = "1d"
MAX_NEWS = 3
SEARCH_DEPTH = "basic"
SUMMARY_LEN = 320
PROMPT_TEMPLATE = DEFAULT_NEWS_PROMPT
MODEL = DEFAULT_MODEL

# ---------------- Data: price + company ----------------
@st.cache_data(show_spinner=False, ttl=60 * 30)
def _prices(t, months_back, interval):
    end = date.today()
    start = end - timedelta(days=months_back * 30)
    return yf_get_history(t, start, end, interval)

px = _prices(ticker, MONTHS_BACK, INTERVAL)
if px.empty:
    st.error("Could not load price data for this ticker.")
    st.stop()

px = add_indicators(px)
info = yf_company_info(ticker)

# ---------------- KPIs ----------------
curr = float(px["Close"].iloc[-1])
prev = float(px["Close"].iloc[-2]) if len(px) > 1 else curr
chg = (curr - prev) / prev * 100 if prev else 0

k1, k2, k3, k4 = st.columns(4)
k1.metric("Price", f"${curr:,.2f}", f"{chg:+.2f}%")
k2.metric("High (1Y)", f"${px['High'].max():,.2f}")
k3.metric("Low (1Y)", f"${px['Low'].min():,.2f}")
k4.metric("Market Cap", human_market_cap(info.get("market_cap")))

# ---------------- Charts ----------------
c1, c2 = st.columns([2, 1], gap="large")

with c1:
    st.subheader(f"{ticker} — Price (Candles) + MAs")
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=px["date"], open=px["Open"], high=px["High"], low=px["Low"], close=px["Close"], name="Price"
    ))
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
    close_fig = go.Figure()
    close_fig.add_trace(go.Scatter(x=px["date"], y=px["Close"], mode="lines", name="Close"))
    close_fig.update_layout(margin=dict(l=10, r=10, t=20, b=10), height=260, yaxis_title="USD")
    st.plotly_chart(close_fig, use_container_width=True)

    st.subheader("Volume")
    vol = go.Figure()
    vol.add_trace(go.Bar(x=px["date"], y=px["Volume"], name="Volume"))
    vol.update_layout(margin=dict(l=10, r=10, t=20, b=10), height=260, yaxis_title="Shares")
    st.plotly_chart(vol, use_container_width=True)

# ---------------- Company snapshot ----------------
with st.expander("ℹ️ Company snapshot", expanded=False):
    st.write(f"**{info.get('shortName', ticker)}**")
    st.write(f"Sector: {info.get('sector','—')} · Industry: {info.get('industry','—')}")
    if info.get("summary"):
        st.write(info["summary"])

# ---------------- Tavily + OpenAI summaries (with spinner) ----------------
st.markdown("## 📰 Latest News & Summaries")

@st.cache_data(ttl=60 * 30, show_spinner=False)
def _news(t, n, depth):
    return get_top_news_for_tickers([t], max_results_per_ticker=n, search_depth=depth).get(t, [])

@st.cache_data(ttl=60 * 10, show_spinner=False)
def _summarize(items, max_chars, model, template):
    return summarize_news_items(items, max_chars=max_chars, model=model, prompt_template=template)

with st.spinner("Fetching news & generating summaries…"):
    try:
        items = _news(ticker, MAX_NEWS, SEARCH_DEPTH)
    except Exception as e:
        st.error(f"News fetch error: {e}")
        st.stop()

    if not items:
        st.caption("No recent news found.")
    else:
        try:
            summaries = _summarize(items, SUMMARY_LEN, MODEL, PROMPT_TEMPLATE)
        except Exception as e:
            summaries = [{"title": it.get("title","Untitled"), "url": it.get("url",""),
                          "summary": f"(summarization error: {e})"} for it in items]

        for s in summaries:
            st.markdown(f"- **[{s['title']}]({s['url']})**")
            st.write(s["summary"])
        st.divider()