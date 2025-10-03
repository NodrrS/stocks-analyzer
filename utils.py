# utils.py
import os
from typing import List, Dict, Optional
import pandas as pd
import yfinance as yf
import streamlit as st
from tavily import TavilyClient


# ---------------- Tavily helpers ----------------
def _get_tavily_key() -> Optional[str]:
    """Get Tavily API key from env or Streamlit secrets."""
    return (
        os.getenv("TAVILY_API")
        or os.getenv("TAVILY_API_KEY")
        or st.secrets.get("TAVILY_API")
        or st.secrets.get("TAVILY_API_KEY")
    )

def tavily_client() -> TavilyClient:
    """Create a Tavily client with API key from secrets/env."""
    key = _get_tavily_key()
    if not key:
        raise RuntimeError("Tavily key missing. Add TAVILY_API in Streamlit Secrets or env.")
    return TavilyClient(api_key=key)

def search_web(query: str) -> str:
    """Simple web search (basic depth). Returns a joined content string."""
    tavily = tavily_client()
    data = tavily.search(query, search_depth="basic")
    return "\n".join(r.get("content", "") for r in data.get("results", []))

def news_for_ticker(ticker_or_name: str, max_results: int = 3) -> List[Dict]:
    """Top news for a ticker/company using Tavily (basic depth)."""
    tavily = tavily_client()
    q = f"{ticker_or_name} stock news earnings guidance product announcement"
    data = tavily.search(q, search_depth="basic", max_results=max_results, include_answer=False)
    return data.get("results", [])[:max_results]


# ---------------- yfinance helpers ----------------
def yf_history_1y(ticker: str, interval: str = "1d") -> pd.DataFrame:
    """Download 1-year OHLCV with tidy columns and date as column."""
    end = pd.Timestamp.today().normalize()
    start = end - pd.Timedelta(days=365)
    df = yf.download(
        ticker,
        start=start,
        end=end + pd.Timedelta(days=1),
        interval=interval,
        auto_adjust=True,
        progress=False,
    )
    if df.empty:
        return df
    if isinstance(df.columns, pd.MultiIndex):
        # Flatten e.g. ('Close','AAPL') -> 'Close'
        df.columns = [c[0] for c in df.columns]
    df = df.rename(columns=str.capitalize).reset_index().rename(columns={"Date": "date"})
    return df

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add MA20/50/200 and RSI14 indicators."""
    if df.empty:
        return df
    out = df.copy()
    close = pd.to_numeric(out["Close"], errors="coerce")

    # Moving averages
    out["MA20"] = close.rolling(20).mean()
    out["MA50"] = close.rolling(50).mean()
    out["MA200"] = close.rolling(200).mean()

    # RSI(14)
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    ag = gain.ewm(alpha=1/14, adjust=False).mean()
    al = loss.ewm(alpha=1/14, adjust=False).mean().replace(0, pd.NA)
    rs = ag / al
    out["RSI14"] = (100 - (100 / (1 + rs))).fillna(50)

    return out

def human_market_cap(v) -> str:
    """Format large numbers like 1.23B."""
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