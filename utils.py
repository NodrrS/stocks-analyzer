# utils.py
import os
from datetime import timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import yfinance as yf
from dotenv import load_dotenv

# Tavily
try:
    from tavily import TavilyClient
except Exception:
    TavilyClient = None  # handled below

load_dotenv()

# -------------------- Finance helpers --------------------
def yf_get_history(ticker: str, start, end, interval: str = "1d", auto_adjust: bool = True) -> pd.DataFrame:
    df = yf.download(
        ticker, start=start, end=end + timedelta(days=1),
        interval=interval, auto_adjust=auto_adjust, progress=False,
    )
    if not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame()

    # Flatten MultiIndex like ('Close','AAPL') -> 'Close'
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [str(c[0]) for c in df.columns]

    df = df.rename(columns=str.capitalize).reset_index().rename(columns={"Date": "date"})
    return df

def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    # ensure 1-D numeric Series
    s = pd.Series(series).squeeze()
    s = pd.to_numeric(s, errors="coerce")

    delta = s.diff()
    gain  = delta.clip(lower=0)
    loss  = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean().replace(0, np.nan)

    rs  = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    out["Close"] = pd.to_numeric(out["Close"], errors="coerce")
    out["MA20"]  = out["Close"].rolling(20).mean()
    out["MA50"]  = out["Close"].rolling(50).mean()
    out["MA200"] = out["Close"].rolling(200).mean()
    out["RSI14"] = _rsi(out["Close"], 14)
    return out

def yf_company_info(ticker: str) -> Dict:
    try:
        t = yf.Ticker(ticker)
        fast = getattr(t, "fast_info", {}) or {}
        info = getattr(t, "info", {}) or {}
        return {
            "shortName": info.get("shortName") or info.get("longName") or ticker,
            "sector": info.get("sector"),
            "industry": info.get("industry"),
            "summary": info.get("longBusinessSummary"),
            "market_cap": fast.get("market_cap") or info.get("marketCap"),
        }
    except Exception:
        return {"shortName": ticker, "market_cap": None}

def human_market_cap(v) -> str:
    try:
        v = float(v)
    except Exception:
        return "—"
    units = ["", "K", "M", "B", "T"]
    k = 0
    while abs(v) >= 1000 and k < len(units) - 1:
        v /= 1000.0
        k += 1
    return f"{v:.2f}{units[k]}"

# -------------------- Tavily helpers --------------------
def _get_tavily_key() -> Optional[str]:
    key = os.getenv("TAVILY_API_KEY") or os.getenv("TAVILY_API")
    if key:
        return key
    try:
        import streamlit as st  # secrets on Streamlit Cloud
        return st.secrets.get("TAVILY_API_KEY") or st.secrets.get("TAVILY_API")
    except Exception:
        return None

_tavily_client: Optional["TavilyClient"] = None

def tavily_client() -> "TavilyClient":
    global _tavily_client
    if _tavily_client is None:
        if TavilyClient is None:
            raise RuntimeError("Install tavily-python: pip install tavily-python")
        key = _get_tavily_key()
        if not key:
            raise RuntimeError("Tavily key not found. Set TAVILY_API_KEY in .env or Streamlit secrets.")
        _tavily_client = TavilyClient(api_key=key)
    return _tavily_client

def news_query_for_ticker(ticker: str) -> str:
    # short, high-signal query
    return f"{ticker} stock news earnings guidance product announcement"

def fetch_news_for_ticker(
    ticker: str, max_results: int = 3, search_depth: str = "basic"
) -> List[dict]:
    client = tavily_client()
    q = news_query_for_ticker(ticker.upper())
    resp = client.search(q, search_depth=search_depth, max_results=max_results, include_answer=False)
    return resp.get("results", [])[:max_results]

def get_top_news_for_tickers(
    tickers: List[str], max_results_per_ticker: int = 3, search_depth: str = "basic"
) -> Dict[str, List[dict]]:
    out: Dict[str, List[dict]] = {}
    for t in tickers:
        try:
            out[t.upper()] = fetch_news_for_ticker(t, max_results=max_results_per_ticker, search_depth=search_depth)
        except Exception as e:
            out[t.upper()] = [{"title": f"Error fetching news for {t}", "url": "", "content": str(e)}]
    return out

# utils.py (append these imports + functions)
import re
import requests

_TICKER_RE = re.compile(r"^[A-Z.\-]{1,10}$")

def is_probable_ticker(s: str) -> bool:
    return bool(_TICKER_RE.match(s.upper()))

def yahoo_symbol_search(query: str, limit: int = 8) -> list[dict]:
    """
    Call Yahoo Finance search API to resolve names/tickers.
    Returns list of suggestions: [{"symbol","shortname","longname","exchDisp","typeDisp"}]
    """
    url = "https://query2.finance.yahoo.com/v1/finance/search"
    params = {"q": query, "quotesCount": limit, "newsCount": 0, "listsCount": 0}
    headers = {"User-Agent": "Mozilla/5.0"}  # avoid 403
    try:
        r = requests.get(url, params=params, headers=headers, timeout=10)
        r.raise_for_status()
        data = r.json() or {}
        out = []
        for q in data.get("quotes", [])[:limit]:
            if (q.get("quoteType") or "").upper() not in {"EQUITY", "ETF", "MUTUALFUND"}:
                continue
            out.append({
                "symbol": (q.get("symbol") or "").upper(),
                "shortname": q.get("shortname") or "",
                "longname": q.get("longname") or "",
                "exchDisp": q.get("exchDisp") or "",
                "typeDisp": q.get("typeDisp") or "",
            })
        return out
    except Exception:
        return []

def resolve_symbol(query: str) -> list[dict]:
    """
    If user typed a probable ticker, return it first; otherwise search by name.
    Output format (same as yahoo_symbol_search)
    """
    q = (query or "").strip()
    if not q:
        return []
    if is_probable_ticker(q):
        # Put the exact ticker guess first, plus enrich with Yahoo data if available
        results = yahoo_symbol_search(q, limit=6)
        seen = {r["symbol"] for r in results}
        head = [{"symbol": q.upper(), "shortname": "", "longname": "", "exchDisp": "", "typeDisp": "Ticker"}]
        return head + [r for r in results if r["symbol"] not in {q.upper()}]
    # otherwise treat as company/name search
    return yahoo_symbol_search(q, limit=8)