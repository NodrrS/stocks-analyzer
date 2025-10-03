# 📊 Stocks Analyzer — Insights for Retail Investors

An interactive **Streamlit dashboard** that helps retail investors quickly identify opportunities and stay updated with stock news.  
The app now has **two parts**:

1. **Top 5 Analyzer** → Quickly spot Buy / Hold / Sell / Dividend opportunities  
2. **Stock Profile + News** → Deep dive into a single stock with charts, KPIs, and news summaries  

Built with **Python, Streamlit, yfinance, Plotly, Tavily, and OpenAI**.

---

## 🚀 Features

### Part 1: Top 5 Analyzer
- **Presets** → Analyze universes like US Big Tech, DAX 40, NASDAQ 100, or custom stock lists.  
- **Filters** → Adjust **Min Momentum (%)** and **Max 60D Volatility (%)** to fine-tune Buy/Hold/Sell signals.  
- **Signals**:
  - ✅ **Top 5 Buy** → strong momentum, low risk  
  - 🟡 **Top 5 Hold** → mixed signals or too volatile  
  - ❌ **Top 5 Sell** → clear downward trend  
  - 👑 **Top 5 Dividend Kings** → highest income yielders  
- **Timeframes** → Switch between Today, Week, Month, and Year.  
- **Visuals** → Growth charts, Risk vs Return scatter, Finviz-style Market Map.  
- **For retail investors** → Lightweight, simple, and easy to interpret.  

---

### Part 2: Stock Profile + News
An extension that combines **technical analysis** with **fundamental insights**.

#### 🔎 Sidebar
- Single input field for ticker/company name  
- **Apply** button → user confirms before loading data  
- Session state keeps the selection persistent  

#### 📈 Stock Profile
- **Data source:** [yfinance](https://pypi.org/project/yfinance/)  
- Historical data: **1-year, daily interval**  
- Indicators: MA20, MA50, MA200, RSI(14)  
- KPIs: Current price, 1-year high, 1-year low, Market Cap  
- Visuals (Plotly):  
  - Candlestick chart with MAs  
  - RSI chart  
  - Closing price line chart  
  - Volume bar chart  

#### 📰 News Section
- **Tavily API** → fetches top 3 relevant stock news  
- **OpenAI API** → summarizes for retail investors  
- Summaries:
  - ≤ 320 characters  
  - Focus on catalysts, guidance, product/macro impacts  
  - Skips irrelevant entries  
- Shows title (linked to source) + summary text  

---

## ⚙️ Setup

### Requirements
Install dependencies:
```bash
pip install -r requirements.txt