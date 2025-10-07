# pages/Stock_profile_+_News.py
import streamlit as st
import plotly.graph_objects as go
from streamlit_searchbox import st_searchbox

from utils import (
    resolve_symbol,
    yf_history_1y,
    add_indicators,
    yf_company_info,
    human_market_cap,
    news_for_ticker,
)
from tools import summarize_news_items, DEFAULT_MODEL

st.set_page_config(page_title="Stock Profile", page_icon="📈", layout="wide")
st.title("📊 Stock Profile")
# --- Mobile sidebar hint (blink the toggle + floating pill) ---
def sidebar_hint(show: bool):
    if not show:
        return
    st.markdown("""
    <style>
      /* Try multiple selectors to catch the header toggle across versions */
      button[kind="header"],
      button[title*="sidebar"],
      [data-testid="stSidebarCollapseButton"] button,
      [data-testid="stSidebarCollapseButton"] {
        animation: sbPulse 1.2s ease-in-out infinite !important;
        outline: 2px solid #ffbd2e !important;
        border-radius: 12px !important;
      }
      @keyframes sbPulse {
        0%   { box-shadow: 0 0 0 rgba(255,189,46,0.0); }
        50%  { box-shadow: 0 0 16px rgba(255,189,46,0.8); }
        100% { box-shadow: 0 0 0 rgba(255,189,46,0.0); }
      }
      /* Floating hint, only on small screens */
      .sb-hint {
        position: fixed; top: 14px; left: 14px;
        z-index: 9999;
        background: #ffffff;
        border: 2px solid #ffbd2e;
        border-radius: 999px;
        padding: 6px 10px;
        font-weight: 600;
        box-shadow: 0 6px 18px rgba(0,0,0,0.15);
        animation: sbPulse 1.2s ease-in-out infinite;
      }
      @media (min-width: 800px) { .sb-hint { display: none; } }  /* desktop: hide pill */
    </style>
    <div class="sb-hint">👈 Tap to open sidebar</div>
    """, unsafe_allow_html=True)

# One-time, show the hint until a ticker is chosen or user dismisses it
if "hide_sidebar_hint" not in st.session_state:
    st.session_state.hide_sidebar_hint = False

# Show when there is no selected ticker yet (first-time mobile landing)
_should_hint = (not st.session_state.get("selected_ticker")) and (not st.session_state.hide_sidebar_hint)
sidebar_hint(_should_hint)

# Optional: small dismiss control (appears only while hint is visible)
if _should_hint:
    st.button("Got it", key="dismiss_sb_hint", on_click=lambda: st.session_state.update(hide_sidebar_hint=True))

# --- Guide: blink CSS + toast if user arrived from Welcome ---
GUIDE_FLAG = st.session_state.get("guide")

def guide_active_for_this_page(flag, expected_filename: str) -> bool:
    return bool(flag and flag.get("target_page") == expected_filename)

# CSS for blinking highlight (border + subtle glow)
def guide_css():
    st.markdown("""
    <style>
      .guide-blink {
        animation: guideBlink 1.2s ease-in-out infinite;
        border: 2px solid #ffbd2e !important;
        border-radius: 10px;
        box-shadow: 0 0 0 rgba(255,189,46,0);
        padding: 8px 12px;
      }
      @keyframes guideBlink {
        0%   { box-shadow: 0 0 0 rgba(255,189,46,0.0); }
        50%  { box-shadow: 0 0 16px rgba(255,189,46,0.75); }
        100% { box-shadow: 0 0 0 rgba(255,189,46,0.0); }
      }
      .guide-caption { color: #c27c08; font-weight: 600; }
    </style>
    """, unsafe_allow_html=True)

# If guide is for this page, show toast + enable CSS once
if guide_active_for_this_page(GUIDE_FLAG, "Stock_profile_+_News.py"):
    guide_css()
    st.toast(GUIDE_FLAG.get("hint", "Start by entering a ticker or company name."))

# -------- Sidebar: type-ahead search + Apply (with auto-select if single match) --------
st.sidebar.header("🔎 Find a stock")

# keep last confirmed ticker
if "selected_ticker" not in st.session_state:
    st.session_state.selected_ticker = None

def _symbol_search_cb(query: str):
    """
    searchbox callback — returns a list of dicts with 'id' (returned on select)
    and 'name' (shown in dropdown). We also stash last query/suggestions in
    session_state so we can auto-select when there's a single match.
    """
    q = (query or "").strip()
    if not q:
        st.session_state["_typeahead_last_query"] = ""
        st.session_state["_typeahead_last_suggestions"] = []
        return []
    results = resolve_symbol(q, limit=8) or []
    st.session_state["_typeahead_last_query"] = q
    st.session_state["_typeahead_last_suggestions"] = results
    return [
        {
            "id": r["symbol"],
            "name": f"{r['symbol']} — {r.get('shortname') or r.get('longname') or ''}"
                    + (f" ({r.get('exchDisp')})" if r.get('exchDisp') else "")
        }
        for r in results
    ]

with st.sidebar:
    # Blink highlight if welcome guide is targeting this control
    highlight = guide_active_for_this_page(GUIDE_FLAG, "Stock_profile_+_News.py") and \
                (GUIDE_FLAG.get("target") == "ticker_input")
    if highlight:
        st.markdown('<div class="guide-blink">', unsafe_allow_html=True)

    selected_item = st_searchbox(
        _symbol_search_cb,
        key="ticker_typeahead",
        placeholder="Type a ticker or company name…",
        clear_on_submit=False,
    )

    if highlight:
        st.caption("💡 **Give here the name or ticker of the stock.**")
        st.markdown('</div>', unsafe_allow_html=True)

    # Extract a pure ticker string from selected_item (dict or str)
    selected_symbol = None
    if isinstance(selected_item, dict):
        selected_symbol = (selected_item.get("id") or selected_item.get("symbol") or "").upper()
    elif isinstance(selected_item, str):
        selected_symbol = selected_item.strip().upper()

    # Auto-select if there's exactly one suggestion
    _last_q = st.session_state.get("_typeahead_last_query", "")
    _last_sugs = st.session_state.get("_typeahead_last_suggestions", [])
    if not selected_symbol and _last_q and isinstance(_last_sugs, list) and len(_last_sugs) == 1:
        selected_symbol = (_last_sugs[0].get("symbol") or "").upper()
        if selected_symbol:
            st.toast(f"Auto-selected: {selected_symbol}")

    # Apply button so heavy charts/news load only when confirmed
    applied = st.button("Apply", use_container_width=True)

# Clear the guide after first render of this page (one-shot hint)
if guide_active_for_this_page(GUIDE_FLAG, "Stock_profile_+_News.py"):
    del st.session_state["guide"]

# Persist user choice on Apply
if applied and selected_symbol:
    st.session_state.selected_ticker = selected_symbol

ticker = st.session_state.selected_ticker
if not ticker:
    st.info("Start typing to search and pick a stock, then click **Apply**.")
    st.stop()

# -------- Charts (1Y, fixed) --------
@st.cache_data(ttl=60 * 30, show_spinner=False)
def _load_px(t: str):
    return yf_history_1y(t)

px = _load_px(ticker)
if px.empty:
    st.error("Could not load price data (check the ticker symbol).")
    st.stop()

px = add_indicators(px)
info = yf_company_info(ticker)

# -------- KPIs --------
curr = float(px["Close"].iloc[-1])
prev = float(px["Close"].iloc[-2]) if len(px) > 1 else curr
chg = (curr - prev) / prev * 100 if prev else 0
k1, k2, k3, k4 = st.columns(4)
k1.metric("Price", f"${curr:,.2f}", f"{chg:+.2f}%")
k2.metric("High (1Y)", f"${px['High'].max():,.2f}")
k3.metric("Low (1Y)", f"${px['Low'].min():,.2f}")
k4.metric("Market Cap", human_market_cap(info.get("market_cap")))

# -------- Charts --------
c1, c2 = st.columns([2, 1], gap="large")
with c1:
    st.subheader(f"{ticker} — Price (Candles) + MAs")
    fig = go.Figure()
    fig.add_trace(
        go.Candlestick(
            x=px["date"],
            open=px["Open"],
            high=px["High"],
            low=px["Low"],
            close=px["Close"],
            name="Price",
        )
    )
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

# -------- Company snapshot --------
with st.expander("ℹ️ Company snapshot", expanded=False):
    st.write(f"**{info.get('name', ticker)}**")
    st.write(f"Sector: {info.get('sector','—')} · Industry: {info.get('industry','—')}")
    if info.get("summary"):
        st.write(info["summary"])

# -------- News + summaries (Tavily basic) --------
st.markdown("## 📰 Latest News & Summaries")

@st.cache_data(ttl=60 * 20, show_spinner=False)
def _fetch_news(t: str):
    return news_for_ticker(t, max_results=3)

with st.spinner("Fetching news & generating summaries…"):
    try:
        items = _fetch_news(ticker)
    except RuntimeError as e:
        # Most likely missing Tavily key; handled in utils.tavily_client()
        st.warning("Tavily API key is missing or not visible to the app. Add `TAVILY_API` in Secrets.")
        items = []

    if not items:
        st.caption("No recent news found.")
    else:
        summaries = summarize_news_items(items, max_chars=320, model=DEFAULT_MODEL)
        for s in summaries:
            st.markdown(f"- **[{s['title']}]({s['url']})**")
            if s["summary"] and not s["summary"].strip().lower().startswith("skip"):
                st.write(s["summary"])
        st.divider()