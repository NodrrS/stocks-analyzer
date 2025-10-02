# utils.py
import os
from datetime import timedelta
from typing import List, Dict, Optional

import pandas as pd
import yfinance as yf
import requests

# --- Optional imports (don’t crash if not installed at import-time) ---
try:
    from tavily import TavilyClient
except Exception:
    TavilyClient = None  # handled in tavily_client()

# Try Streamlit secrets if available
def _get_secret(name: str) -> Optional[str]:
    try:
        import streamlit as st  # only available in Streamlit runtime
        return st.secrets.get(name)
    except Exception:
        return None

# ---- Keys (env first, then secrets.toml) ----
TAVILY_API = (
    os.getenv("TAVILY_API")
    or os.getenv("TAVILY_API_KEY")
    or _get_secret("TAVILY_API")
    or _get_secret("TAVILY_API_KEY")
)

# ---------------- Tavily helpers (lesson-style) ----------------
def tavily_client() -> "TavilyClient":
    if TavilyClient is None:
        raise RuntimeError("Install tavily-python: pip install tavily-python")
    if not TAVILY_API:
        raise RuntimeError("Tavily key missing. Set TAVILY_API in .env or .streamlit/secrets.toml")
    return TavilyClient(api_key=TAVILY_API)

def search_web(query: str) -> str:
    """Simple web search (basic depth). Returns a joined content string."""
    tavily = tavily_client()
    data = tavily.search(query, search_depth="basic")
    return "\n".join(r.get("content", "") for r in data.get("results", []))

def news_for_ticker(ticker_or_name: str, max_results: int = 3) -> List[Dict]:
    """Top news for a ticker/company. Tavily basic depth, lesson-style."""
    tavily = tavily_client()
    q = f"{ticker_or_name} stock news earnings guidance product announcement"
    data = tavily.search(q, search_depth="basic", max_results=max_results, include_answer=False)
    return data.get("results", [])[:max_results]

# ---------------- yfinance helpers ----------------
def yf_history_1y(ticker: str, interval: str = "1d") -> pd.DataFrame:
    """1-year OHLCV with tidy columns + date index as column."""
    end = pd.Timestamp.today().normalize()
    start = end - pd.Timedelta(days=365)
    df = yf.download(
        ticker, start=start, end=end + pd.Timedelta(days=1),
        interval=interval, auto_adjust=True, progress=False
    )
    if df.empty:
        return df
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]  # ('Close','AAPL') -> 'Close'
    df = df.rename(columns=str.capitalize).reset_index().rename(columns={"Date": "date"})
    return df

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """MA20/50/200 + RSI14 (simple, stable)."""
    if df.empty:
        return df
    out = df.copy()
    close = pd.to_numeric(out["Close"], errors="coerce")

    # MAs
    out["MA20"] = close.rolling(20).mean()
    out["MA50"] = close.rolling(50).mean()
    out["MA200"] = close.rolling(200).mean()

    # RSI(14) with EWM
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    ag = gain.ewm(alpha=1/14, adjust=False).mean()
    al = loss.ewm(alpha=1/14, adjust=False).mean().replace(0, pd.NA)
    rs = ag / al
    out["RSI14"] = (100 - (100 / (1 + rs))).fillna(50)

    return out

def human_market_cap(v) -> str:
    try:
        x = float(v)
    except Exception:
        return "—"
    units = ["", "K", "M", "B", "T"]
    k = 0
    while abs(x) >= 1000 and k < len(units) - 1:
        x /= 1000.0
        k += 1
    return f"{x:.2f}{units[k]}"

def yf_company_info(ticker: str) -> Dict:
    """Basic company snapshot from yfinance."""
    try:
        t = yf.Ticker(ticker)
        info = getattr(t, "info", {}) or {}
        fast = getattr(t, "fast_info", {}) or {}
        return {
            "name": info.get("shortName") or info.get("longName") or ticker,
            "sector": info.get("sector"),
            "industry": info.get("industry"),
            "summary": info.get("longBusinessSummary"),
            "market_cap": fast.get("market_cap") or info.get("marketCap"),
        }
    except Exception:
        return {"name": ticker}