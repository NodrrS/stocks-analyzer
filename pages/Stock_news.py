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

# ---------------- Sidebar: single search + controls ----------------

st.sidebar.header("🔎 Find a stock")

search_query = st.sidebar.text_input(
    "Type a company name or ticker",
    value="",
    placeholder="e.g., Applied Materials or AMAT",
    key="sp_search"
)

# Suggestion list (cached for snappiness)
@st.cache_data(ttl=60, show_spinner=False)
def _suggest(q: str):
    return resolve_symbol(q)

suggestions = _suggest(search_query) if search_query.strip() else []

if suggestions:
    # Render suggestions as radio buttons
    labels = [
        f"{s['symbol']} — {s['shortname'] or s['longname'] or '—'} ({s['exchDisp']})"
        for s in suggestions
    ]
    idx = st.sidebar.radio("Suggestions", options=list(range(len(suggestions))), format_func=lambda i: labels[i], key="sp_pick")
    ticker = suggestions[idx]["symbol"]
else:
    ticker = None

# Price range
st.sidebar.markdown("---")
rng = st.sidebar.select_slider("Price range", options=["3M", "6M", "1Y", "2Y", "5Y"], value="1Y", key="sp_range")
months = {"3M": 3, "6M": 6, "1Y": 12, "2Y": 24, "5Y": 60}
interval = "1d"
end = date.today()
start = end - timedelta(days=months[rng] * 30)

# News controls
st.sidebar.subheader("📰 News & Summaries")
max_news = st.sidebar.number_input("Max news", 1, 10, 3, step=1, key="sp_max_news")
summary_len = st.sidebar.number_input("Summary length (chars)", 120, 1000, 320, step=20, key="sp_sum_len")
search_depth = st.sidebar.selectbox("Search depth", ["basic", "advanced"], index=0, key="sp_depth")
model = st.sidebar.text_input("OpenAI model", value=DEFAULT_MODEL, key="sp_model")
with st.sidebar.expander("✏️ Prompt template", expanded=False):
    prompt_template = st.text_area(
        "Template (use {max_chars} {title} {url} {content})",
        value=DEFAULT_NEWS_PROMPT,
        height=220,
        key="sp_prompt",
    )

# ---------------- Guard: need a chosen ticker ----------------
if not ticker:
    st.info("Start typing a company name or ticker in the sidebar, then pick one from the suggestions.")
    st.stop()

# ---------------- Data: price + company ----------------
@st.cache_data(show_spinner=False, ttl=60*30)
def _prices(t, s, e, i):
    return yf_get_history(t, s, e, i)

px = _prices(ticker, start, end, interval)
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
k2.metric("High (period)", f"${px['High'].max():,.2f}")
k3.metric("Low (period)", f"${px['Low'].min():,.2f}")
k4.metric("Market Cap", human_market_cap(info.get("market_cap")))

# ---------------- Charts ----------------
c1, c2 = st.columns([2, 1], gap="large")

with c1:
    st.subheader(f"{ticker} — Price (Candles) + MAs")
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=px["date"], open=px["Open"], high=px["High"], low=px["Low"], close=px["Close"], name="Price"))
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

# ---------------- Tavily + OpenAI summaries ----------------
st.markdown("## 📰 Latest News & Summaries")

@st.cache_data(ttl=60*30, show_spinner=False)
def _news(t, n, depth):
    from utils import get_top_news_for_tickers
    return get_top_news_for_tickers([t], max_results_per_ticker=n, search_depth=depth).get(t, [])

@st.cache_data(ttl=60*10, show_spinner=False)
def _summarize(items, max_chars, model, template):
    return summarize_news_items(items, max_chars=max_chars, model=model, prompt_template=template)

try:
    items = _news(ticker, max_news, search_depth)
except Exception as e:
    st.error(f"News fetch error: {e}")
    st.stop()

if not items:
    st.caption("No recent news found.")
else:
    try:
        summaries = _summarize(items, summary_len, model, prompt_template)
    except Exception as e:
        summaries = [{"title": it.get("title","Untitled"), "url": it.get("url",""), "summary": f"(summarization error: {e})"} for it in items]

    for s in summaries:
        title = s["title"]
        url = s["url"]
        st.markdown(f"- **[{title}]({url})**")
        st.write(s["summary"])
    st.divider()