"""
Single-page Streamlit app (no PDF upload) that analyzes stocks using yfinance.

Features
1) Daily Top-5 lists from a user-editable watchlist (defaults provided)
   ➜ Click **Analyze** to compute: Top 5 Buy / Hold / Sell + Top 5 Dividend Kings
2) Manual Ticker Input ➜ enter one or more tickers and get analysis cards

Run:
  pip install streamlit yfinance pandas numpy plotly
  streamlit run app.py

Notes:
- This is not financial advice.
- Momentum windows: 3M ≈ 63 trading days, 6M ≈ 126 days
- BUY rule (default): mom_3m & mom_6m ≥ 3% and vol_60d ≤ 35%
- SELL rule (default): mom_3m & mom_6m ≤ −3%
"""
import math
import time
import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st
import plotly.express as px
import re

st.set_page_config(page_title="Stocks Analyzer — Retail Top 5", layout="wide")


# ----------------------------
# Config & helpers
# ----------------------------
DEFAULT_WATCHLIST = "AAPL, MSFT, NVDA, GOOGL, AMZN, META, TSLA, JPM, V, MA, UNH, JNJ, PG, XOM, CVX, LVMUY, SAP, ASML, ORCL, COST"
# NEW: curated universes you can expand anytime
PRESETS = {
    # 0. None - for manual selection only
    "None": [],
    
    # 1. Big Tech / US Megacaps
    "US Big Tech": [
        "AAPL","MSFT","NVDA","GOOGL","GOOG","AMZN","META","TSLA","BRK-B","AVGO"
    ],

    # 2. Dow Jones 30 (selection)
    "Dow Jones 30": [
        "AAPL","MSFT","GS","JPM","V","MA","HD","MCD","DIS","INTC","IBM","NKE",
        "CVX","XOM","CAT","UNH","JNJ","PG","KO","WMT"
    ],

    # 3. S&P 100 (selection)
    "S&P 100": [
        "AAPL","MSFT","AMZN","GOOGL","META","BRK-B","JNJ","XOM","PG","JPM",
        "V","MA","HD","CVX","ABBV","LLY","PEP","PFE","MRK","BAC"
    ],

    # 4. NASDAQ 100 (selection, already in your code)
    "NASDAQ 100": [
        "AAPL","MSFT","NVDA","GOOGL","AMZN","META","TSLA","ADBE","NFLX","AVGO",
        "AMD","INTC","AMAT","CSCO","QCOM","TXN","PYPL","BKNG","REGN","ASML"
    ],

    # 5. DAX 40 (selection)
    "DAX 40": [
        "SAP.DE","ALV.DE","BMW.DE","BAS.DE","BAYN.DE","DPW.DE","DBK.DE","DTE.DE","RWE.DE","MUV2.DE",
        "ADS.DE","VOW3.DE","HEN3.DE","IFX.DE","SIE.DE","FRE.DE","HEI.DE","PUM.DE","BEI.DE","VNA.DE"
    ],

    # 6. FTSE 100 (selection)
    "FTSE 100": [
        "HSBA.L","ULVR.L","SHEL.L","BP.L","AZN.L","GSK.L","BATS.L","RIO.L","LLOY.L","BARC.L",
        "TSCO.L","VOD.L","DGE.L","BHP.L","REL.L","GLEN.L","NG.L","BA.L","PRU.L","STAN.L"
    ],

    # 7. CAC 40 (France)
    "CAC 40": [
        "MC.PA","OR.PA","SAN.PA","BNP.PA","AIR.PA","AI.PA","GLE.PA","EN.PA","ENGI.PA","DG.PA",
        "EL.PA","SU.PA","HO.PA","VIV.PA","KER.PA","RI.PA","ACA.PA","SGO.PA","STM.PA","CAP.PA"
    ],

    # 8. Euro Stoxx 50 (selection)
    "Euro Stoxx 50": [
        "ASML.AS","SAP.DE","LIN","AIR.PA","SAN.PA","OR.PA","MC.PA","ENEL.MI","IBE.MC","ITX.MC",
        "TTE.PA","BMW.DE","DAI.DE","AD.AS","ADYEN.AS","BAS.DE","BNP.PA","AI.PA","SU.PA","VIV.PA"
    ],

    # 9. Nikkei 225 (selection)
    "Nikkei 225": [
        "7203.T","6758.T","9984.T","8306.T","9432.T","7974.T","6861.T","6954.T","8035.T","8058.T",
        "4063.T","8766.T","6752.T","4502.T","4661.T","6367.T","6501.T","6594.T","6952.T","8031.T"
    ],

    # 10. Emerging Markets (selection)
    "Emerging Markets": [
        "BABA","PDD","TCEHY","JD","MELI","INFY","HDB","TSM","VALE","PETR4.SA",
        "ITUB4.SA","SBER.ME","NPN.JO","MTN.JO","TATAMOTORS.NS","RELIANCE.NS","ICICIBANK.NS"
    ],
}
PRESETS["All"] = sorted(set(sum(PRESETS.values(), [])))
# 
# ---- Timeframe helpers ----
TIMEFRAMES = {
    "Today":   {"short": 1,   "long": 5,   "period": "3mo"},   # 1-day vs 5-day
    "Week":    {"short": 5,   "long": 20,  "period": "6mo"},
    "Month":   {"short": 21,  "long": 63,  "period": "1y"},
    "Year":    {"short": 63,  "long": 126, "period": "2y"},    # roughly 3m/6m
}
def get_windows(tf: str):
    cfg = TIMEFRAMES.get(tf, TIMEFRAMES["Month"])
    return cfg["short"], cfg["long"], cfg["period"]
# 
def parse_tickers(text: str) -> list[str]:
    raw = re.split(r"[,\s;]+", (text or "").upper())
    return [t for t in raw if t and t not in {".", ",", ";", ":"}]

@st.cache_data(ttl=60*60*12, show_spinner=False)  # cache up to ~half-day
def fetch_history(ticker: str, period: str = "1y"):
    # Use Ticker.history with auto_adjust; work with the 'Close' column
    for _ in range(2):  # tiny retry to mitigate transient empties
        try:
            t = yf.Ticker(ticker)
            hist = t.history(period=period, interval="1d", auto_adjust=True)
            if hist is not None and not hist.empty:
                if "Close" in hist.columns:
                    return hist[["Close"]].dropna()
                # Some data sources may name it 'Adj Close' or lowercase
                for col in ["Adj Close", "close", "adjclose", "adj close"]:
                    if col in hist.columns:
                        s = hist[[col]].dropna()
                        s.columns = ["Close"]
                        return s
        except Exception:
            pass
        time.sleep(0.2)
    return None

@st.cache_data(ttl=60*60*12, show_spinner=False)
def fetch_dividends(ticker: str):
    try:
        t = yf.Ticker(ticker)
        return t.dividends
    except Exception:
        return None

@st.cache_data(ttl=60*60*12, show_spinner=False)
def fetch_info(ticker: str):
    try:
        t = yf.Ticker(ticker)
        info = getattr(t, "get_info", None)
        info = info() if callable(info) else getattr(t, "info", {})
        return info if isinstance(info, dict) else {}
    except Exception:
        return {}

# Compute metrics for a single ticker
@st.cache_data(ttl=60*60*12, show_spinner=False)
def compute_metrics(ticker: str, short_days=21, long_days=63, period="1y"):
    hist = fetch_history(ticker, period=period)   # <<< pass period here
    if hist is None or hist.empty:
        return None
    s = hist["Close"].dropna()
    r = s.pct_change().dropna()

    def pct_change_days(days):
        return s.pct_change(days).iloc[-1] if len(s) > days else np.nan

    mom_short = pct_change_days(short_days)
    mom_long  = pct_change_days(long_days)
    vol_60d   = r.rolling(60).std().iloc[-1] if len(r) > 60 else np.nan
    peak = s.cummax()
    drawdown_1y = (s/peak - 1.0).min() if len(s) else np.nan

    # Get dividend information
    ttm_div = 0.0
    try:
        t = yf.Ticker(ticker)
        div = t.dividends
        if not div.empty:
            end = div.index.max()
            if pd.notna(end):
                start = end - pd.Timedelta(days=365)
                ttm_div = float(div.loc[div.index >= start].sum())
    except Exception:
        # Fallback: sum of last 4 payments
        try:
            ttm_div = float(div.tail(4).sum()) if not div.empty else 0.0
        except:
            ttm_div = 0.0

    last_price = float(s.iloc[-1]) if len(s) else np.nan
    div_yield = (ttm_div / last_price) if last_price and not math.isnan(last_price) and last_price>0 else 0.0

    info = fetch_info(ticker)
    sector = info.get("sector")
    name = info.get("shortName") or info.get("longName") or ticker

    return {
        "symbol": ticker,
        "name": name,
        "sector": sector,
        "last_price": last_price,
        "mom_short": mom_short,   # <<< renamed fields
        "mom_long":  mom_long,
        "vol_60d": vol_60d,
        "drawdown_1y": drawdown_1y,
        "div_yield": div_yield,
    }


# Decide Buy / Hold / Sell for a row of metrics

def decide_action(row, min_mom=0.03, max_vol=0.35):
    mS, mL, vol = row["mom_short"], row["mom_long"], row["vol_60d"]
    if any(pd.isna([mS, mL, vol])): return "hold", 0.2, "insufficient data"
    if (mS <= -min_mom) and (mL <= -min_mom): return "sell", 0.7, "negative momentum"
    if (mS >=  min_mom) and (mL >=  min_mom) and (vol <= max_vol): return "buy", 0.7, "positive momentum & acceptable risk"
    return "hold", 0.5, "mixed signals"
    


# Compute metrics for many tickers
@st.cache_data(ttl=60*60*12, show_spinner=False)
def analyze_universe(tickers: list[str],
                     min_mom=0.03, max_vol=0.35,
                     short_days=21, long_days=63, period="1y"):
    rows, failed = [], []
    for t in tickers:
        m = compute_metrics(t, short_days=short_days, long_days=long_days, period=period)
        if m: rows.append(m)
        else: failed.append(t)
    
    df = pd.DataFrame(rows)
    if df.empty:
        return df, {"buy": df, "hold": df, "sell": df, "kings": df}
    
    acts, confs, whys = [], [], []
    for _, r in df.iterrows():
        a, c, w = decide_action(r, min_mom=min_mom, max_vol=max_vol)
        acts.append(a); confs.append(c); whys.append(w)
    df["action"], df["confidence"], df["rationale"] = acts, confs, whys

    # Top 5 lists
    buy  = df[df.action=="buy"].sort_values(["mom_long","mom_short"], ascending=False).head(5)
    sell = df[df.action=="sell"].sort_values(["mom_long","mom_short"], ascending=True).head(5)
    hold = df[df.action=="hold"].sort_values(["vol_60d"], ascending=True).head(5)
    kings= df.sort_values("div_yield", ascending=False).head(5)
    
    return df, {"buy": buy, "hold": hold, "sell": sell, "kings": kings}
    return df, lists

# ----------------------------
# UI
# ----------------------------
st.title("Stocks Analyzer — Retail Top 5")
st.caption("For research only — not financial advice.")

with st.expander("ℹ️ How to Work with This App"):
    st.markdown("""
        **1. Use the sidebar to set up your analysis:**
        - **Choose a Preset Universe** → e.g., US Big Tech, DAX 40, or None (manual only).  
        - **Select Stocks** → pick from the preset or open the global list to add more tickers.  
        - **Min Momentum (%)** → minimum % growth in both short- and long-term momentum for a stock to qualify as Buy.  
        - **Max 60D Volatility (%)** → filters out highly volatile stocks (risk control).  
        - **Timeframe** → sets the lookback period for momentum (Today / Week / Month / Year).  
        - Click **Analyze Selection** to run the analysis.

        **2. Interpreting the results below:**
        - **Top 5 Tables** → lists the strongest Buy, Hold, Sell, and Dividend King stocks from your selection.  
        - **Growth Charts** → show normalized performance of Top 5 Buy, Hold, and Sell stocks over time.  
        - **Risk vs Return Scatter** → compares volatility (x-axis) vs momentum (y-axis); colors = signals.  
        - **Market Map Treemap** → visual overview of Top-5 categories: size = Market Cap, color = Momentum.  

        This workflow helps you filter stocks, understand trends, and quickly identify opportunities or risks.
        """)

with st.sidebar:
    st.header("Universe & Rules")

    # --- preset selector ---
    preset = st.selectbox("Choose a preset universe", list(PRESETS.keys()))

    # --- preset-based selection (only show if preset is not "None") ---
    selected_preset = []
    if preset != "None":
        options = PRESETS[preset]
        selected_preset = st.multiselect(
            "Select stocks from preset",
            options=options,
            default=options[:15],
            help="Type to search; tick the boxes you want to include."
        )
        st.caption(f"Selected from preset: **{len(selected_preset)}** of {len(options)}")

    # --- individual stocks button ---
    if "show_all_picker" not in st.session_state:
        st.session_state.show_all_picker = False
    st.markdown("""
    <style>
    div.stButton > button:nth-child(1) {
        background-color: #4CAF50;  /* Green background */
        color: white;               /* White text */
        border-radius: 8px;         /* Rounded corners */
        height: 3em;
        width: 100%;
    }
    </style>
    """, unsafe_allow_html=True)
    if st.button("Choose individual stocks to analyze"):
        st.session_state.show_all_picker = not st.session_state.show_all_picker

    # --- GLOBAL picker from PRESETS['All'] (shown when toggled) ---
    selected_all = []
    if st.session_state.show_all_picker:
        st.markdown("**Global picker (All symbols)**")

        # keep a persistent selection state for the global list
        if "selected_all_state" not in st.session_state:
            st.session_state.selected_all_state = []

        cc1, cc2 = st.columns(2)
        all_options = PRESETS["All"]

        if cc1.button("Select all (All)"):
            st.session_state.selected_all_state = all_options[:]  # copy entire list
        if cc2.button("Clear all (All)"):
            st.session_state.selected_all_state = []

        selected_all = st.multiselect(
            "Pick additional symbols from the global list",
            options=all_options,
            default=st.session_state.selected_all_state,
            help="Adds to your preset selection."
        )
        # sync state if user changed selection manually
        st.session_state.selected_all_state = selected_all
        st.caption(f"Selected from All: **{len(selected_all)}** of {len(all_options)}")

    # --- thresholds ---
    c1, c2 = st.columns(2)
    min_mom = c1.number_input(
    "Min momentum (%)", value=3.0, step=0.5, 
    help="Minimum momentum threshold (short & long). If both are above this % and volatility is acceptable, stock is a Buy."
) / 100.0
    max_vol = c2.number_input(
    "Max 60D vol (%)", value=35.0, step=1.0, 
    help="Maximum allowed volatility over the past 60 days. Higher = riskier stock."
) / 100.0
# 
    # Timeframe selector
    timeframe = st.selectbox(
    "Timeframe", list(TIMEFRAMES.keys()), index=2,
    help="Controls the analysis window: Today, Week, Month, Year → maps to different momentum lookback periods."
)
    short_days, long_days, period = get_windows(timeframe)
#  
    # --- analyze button ---
    analyze_selection_btn = st.button("Analyze Selection", type="primary")

# OUTSIDE the sidebar: use this for analysis (preset ∪ global)
combined_selected = sorted(set(selected_preset + selected_all))

st.markdown("---")

st.subheader("Top-5 from Selected Stocks")
st.caption(
    "The app ranks your selected stocks into **Buy / Hold / Sell** signals and "
    "lists the **Top 5 Dividend Kings** with the highest trailing dividend yield. "
    "This helps you quickly see potential opportunities and risks in your watchlist."
)

if analyze_selection_btn:
    if not combined_selected:
        st.warning("No stocks selected. Tick some from the preset and/or the global picker.")
    else:
        with st.spinner("Analyzing selection..."):
            df_all, tops = analyze_universe(
                combined_selected,
                min_mom=min_mom, max_vol=max_vol,
                short_days=short_days, long_days=long_days, period=period
            )
            # Store results in session state
            st.session_state.df_all = df_all
            st.session_state.tops = tops
            st.session_state.analysis_done = True
        if df_all.empty:
            st.warning("No data fetched. Check symbols or network.")
        else:
            # Column explanations (expander)
            with st.expander("ℹ️ Column Explanations"):
                st.markdown("""
                - **Ticker** → Stock symbol used on the exchange (e.g., AAPL)  
                - **Company** → Full company name  
                - **Momentum (X d)** → % change in stock price over X days (short = near-term, long = trend)  
                - **Volatility (60d)** → Price fluctuation risk measured as 60-day rolling standard deviation  
                - **Dividend Yield (TTM)** → Dividends paid in the last 12 months divided by stock price  
                - **Price (Last Close)** → Last market closing price  
                - **Dividend Yield %** → Dividend yield expressed as a percentage  
                """)
            # Stack all tables vertically
            rename_map = {
                "symbol": "Ticker",
                "name": "Company",
                "mom_short": f"Momentum ({short_days}d)",
                "mom_long": f"Momentum ({long_days}d)",
                "vol_60d": "Volatility (60d)",
                "div_yield": "Dividend Yield (TTM)",
                "last_price": "Price (Last Close)",
                "yield_%": "Dividend Yield %",
            }

            st.markdown("### Top 5 Buy")
            st.dataframe(
                tops["buy"][["symbol","name","mom_short","mom_long","vol_60d","div_yield"]]
                .rename(columns=rename_map),
                use_container_width=True
            )

            st.markdown("### Top 5 Hold")
            st.dataframe(
                tops["hold"][["symbol","name","mom_short","mom_long","vol_60d","div_yield"]]
                .rename(columns=rename_map),
                use_container_width=True
            )

            st.markdown("### Top 5 Sell")
            st.dataframe(
                tops["sell"][["symbol","name","mom_short","mom_long","vol_60d","div_yield"]]
                .rename(columns=rename_map),
                use_container_width=True
            )
            
            st.markdown("### Top 5 Dividend Kings (by trailing yield)")
            k = tops["kings"].copy()
            k["yield_%"] = (k["div_yield"].fillna(0) * 100).round(2)
            st.dataframe(
                k[["symbol","name","yield_%","last_price"]]
                .rename(columns=rename_map),
                use_container_width=True
            )

            # All charts in 2x2 layout
            try:
                st.subheader("Charts & Analysis")
                
                # Create 2x2 grid for all charts
                chart_cols = st.columns(2)
                
                # Top row: Growth charts for Buy and Hold
                with chart_cols[0]:
                    st.subheader("Top 5 Buy — Growth Over Time")
                    st.caption(
                        "Shows how the **Top 5 Buy** stocks performed over the selected timeframe. "
                        "Growth is normalized (% change from the first day). "
                        "Useful to confirm that Buy candidates are trending up."
                    )
                    if not tops["buy"].empty:
                        growth_data = []
                        for symbol in tops["buy"]["symbol"].head(5):
                            hist = fetch_history(symbol, period=period)
                            if hist is not None and not hist.empty:
                                close_prices = hist["Close"]
                                if len(close_prices) > 0:
                                    start_price = close_prices.iloc[0]
                                    growth = ((close_prices / start_price) - 1) * 100
                                    symbol_df = pd.DataFrame({
                                        'Date': hist.index,
                                        'Growth_%': growth,
                                        'Symbol': symbol,
                                        'Name': tops["buy"][tops["buy"]["symbol"] == symbol]["name"].iloc[0] if not tops["buy"][tops["buy"]["symbol"] == symbol]["name"].empty else symbol
                                    })
                                    growth_data.append(symbol_df)
                        
                        if growth_data:
                            combined_df = pd.concat(growth_data, ignore_index=True)
                            fig_growth = px.line(
                                combined_df,
                                x="Date",
                                y="Growth_%",
                                color="Symbol",
                                title=f"Top 5 Buy - Stock Growth ({timeframe})",
                                hover_data={"Name": True, "Growth_%": ":.1f%"},
                                markers=True
                            )
                            fig_growth.update_layout(yaxis_title="Growth %", height=300, showlegend=True)
                            fig_growth.update_traces(line=dict(width=2))
                            st.plotly_chart(fig_growth, use_container_width=True)
                
                with chart_cols[1]:
                    st.subheader("Top 5 Hold — Growth Over Time")
                    st.caption(
                        "Tracks **Top 5 Hold** stocks over the selected period. "
                        "Normalized performance helps see if they are stabilizing or drifting. "
                        "Hold = mixed signals (not strong enough to Buy, not weak enough to Sell)."
                    )
                    if not tops["hold"].empty:
                        growth_data = []
                        for symbol in tops["hold"]["symbol"].head(5):
                            hist = fetch_history(symbol, period=period)
                            if hist is not None and not hist.empty:
                                close_prices = hist["Close"]
                                if len(close_prices) > 0:
                                    start_price = close_prices.iloc[0]
                                    growth = ((close_prices / start_price) - 1) * 100
                                    symbol_df = pd.DataFrame({
                                        'Date': hist.index,
                                        'Growth_%': growth,
                                        'Symbol': symbol,
                                        'Name': tops["hold"][tops["hold"]["symbol"] == symbol]["name"].iloc[0] if not tops["hold"][tops["hold"]["symbol"] == symbol]["name"].empty else symbol
                                    })
                                    growth_data.append(symbol_df)
                        
                        if growth_data:
                            combined_df = pd.concat(growth_data, ignore_index=True)
                            fig_growth = px.line(
                                combined_df,
                                x="Date",
                                y="Growth_%",
                                color="Symbol",
                                title=f"Top 5 Hold - Stock Growth ({timeframe})",
                                hover_data={"Name": True, "Growth_%": ":.1f%"},
                                markers=True
                            )
                            fig_growth.update_layout(yaxis_title="Growth %", height=300, showlegend=True)
                            fig_growth.update_traces(line=dict(width=2))
                            st.plotly_chart(fig_growth, use_container_width=True)
                
                # Bottom row: Scatter plot and Sell growth chart
                
                with chart_cols[0]:
                    st.subheader("Risk vs Return Scatter")
                    st.caption (
                    "Each point represents one stock. **X-axis = volatility (risk)**, **Y-axis = momentum (return)**.  \n"
                    "- Top-left quadrant: strong performance with low volatility → attractive candidates  \n"
                    "- Bottom-right quadrant: weak momentum with high volatility → risky candidates  \n"
                    "- Colors show the Buy / Hold / Sell signal classification."
                    )
                    fig = px.scatter(
                        df_all, x="vol_60d", y="mom_long",
                        hover_data=["symbol","name","sector"],
                        title="Risk vs Return (long vs 60D vol)",
                        color="action"
                    )
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)
                
                
                with chart_cols[1]:
                    st.subheader("Top 5 Sell — Growth Over Time")
                    st.caption(
                        "Tracks **Top 5 Sell** stocks. "
                        
                        
                        "Normalized performance highlights sustained weakness or failed rebounds—"
                        "useful to validate Sell signals."
                    )
                    if not tops["sell"].empty:
                        growth_data = []
                        for symbol in tops["sell"]["symbol"].head(5):
                            hist = fetch_history(symbol, period=period)
                            if hist is not None and not hist.empty:
                                close_prices = hist["Close"]
                                if len(close_prices) > 0:
                                    start_price = close_prices.iloc[0]
                                    growth = ((close_prices / start_price) - 1) * 100
                                    symbol_df = pd.DataFrame({
                                        'Date': hist.index,
                                        'Growth_%': growth,
                                        'Symbol': symbol,
                                        'Name': tops["sell"][tops["sell"]["symbol"] == symbol]["name"].iloc[0] if not tops["sell"][tops["sell"]["symbol"] == symbol]["name"].empty else symbol
                                    })
                                    growth_data.append(symbol_df)
                        
                        if growth_data:
                            combined_df = pd.concat(growth_data, ignore_index=True)
                            fig_growth = px.line(
                                combined_df,
                                x="Date",
                                y="Growth_%",
                                color="Symbol",
                                title=f"Top 5 Sell - Stock Growth ({timeframe})",
                                hover_data={"Name": True, "Growth_%": ":.1f%"},
                                markers=True
                            )
                            fig_growth.update_layout(yaxis_title="Growth %", height=300, showlegend=True)
                            fig_growth.update_traces(line=dict(width=2))
                            st.plotly_chart(fig_growth, use_container_width=True)
                            
            except Exception as e:
                st.error(f"Error creating charts: {str(e)}")
                pass

            # Market Map - integrated into main analysis
            try:
                st.subheader(f"Market Map — {timeframe}")
                st.caption(
                "The treemap mirrors the **Top-5 categories**.  \n"
                "- **Size** = Market Cap (larger boxes = bigger companies).  \n"
                "- **Color** = Momentum (green = strong upward trend, red = weak/negative)  \n"
                "- Hierarchy = Action → Symbol (Buy, Hold, Sell, Dividend Kings).  \n"
                "This provides a quick overview of relative weight and performance."
                )
                
                # Build category → symbols from 'tops' dict (all categories)
                cat_to_syms = {
                    "Top 5 Buy": list(tops["buy"]["symbol"]) if not tops["buy"].empty else [],
                    "Top 5 Hold": list(tops["hold"]["symbol"]) if not tops["hold"].empty else [],
                    "Top 5 Sell": list(tops["sell"]["symbol"]) if not tops["sell"].empty else [],
                    "Top 5 Dividend Kings": list(tops["kings"]["symbol"]) if not tops["kings"].empty else [],
                }

                # Include all categories
                allowed_syms = set()
                for symbols in cat_to_syms.values():
                    allowed_syms.update(symbols)

                df_map = df_all[df_all["symbol"].isin(allowed_syms)].copy()

                # Ensure we have 'action' (Buy/Hold/Sell); kings inherit their existing action
                if "action" not in df_map.columns:
                    df_map["action"] = "hold"  # safety default

                # Ensure market cap exists (fetch via cached fetch_info if missing)
                if "market_cap" not in df_map.columns:
                    def _get_cap(sym: str):
                        info = fetch_info(sym) or {}
                        return info.get("marketCap")
                    df_map["market_cap"] = df_map["symbol"].apply(_get_cap)

                # Fallback market cap to avoid zero-sized boxes
                df_map["market_cap"] = df_map["market_cap"].fillna(1_000_000_000)  # 1B fallback

                if df_map.empty:
                    st.info("No symbols to plot for the Top-5 categories.")
                else:
                    # Hierarchy: Action -> Symbol (no sectors, to mirror Top-5 lists)
                    path = ["action", "symbol"]

                    # Use momentum coloring by default
                    fig_map = px.treemap(
                        df_map,
                        path=path,
                        values="market_cap",
                        color="mom_long",
                        color_continuous_scale="RdYlGn",
                        hover_data={
                            "name": True,
                            "mom_long": ":.2%",
                            "mom_short": ":.2%",
                            "div_yield": ":.2%",
                            "vol_60d": ":.2f",
                            "market_cap": ":,.0f",
                            "action": True,
                        },
                        title=None,
                    )
                    fig_map.update_layout(coloraxis_colorbar_title="Momentum")
                    st.plotly_chart(fig_map, use_container_width=True)
            except Exception as e:
                st.error(f"Error creating market map: {str(e)}")
                pass

else:
    st.info("Choose a preset, tick stocks, set thresholds, then click Analyze Selection.")

st.markdown("---")


