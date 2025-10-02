# 📊 Stocks Analyzer — Top 5 Insights for Retail Investors

A simple, interactive **Streamlit dashboard** that helps retail investors quickly identify:

- ✅ **Top 5 Buy candidates** (strong momentum, low risk)  
- 🟡 **Top 5 Hold candidates** (mixed signals or too volatile)  
- ❌ **Top 5 Sell candidates** (clear downward trend)  
- 👑 **Top 5 Dividend Kings** (highest income yielders)  

Built with **Python, Streamlit, yfinance, and Plotly**.

---

## 🚀 Features
- **Presets** → Analyze universes like US Big Tech, DAX 40, NASDAQ 100, or custom stock lists.  
- **Filters** → Adjust **Min Momentum (%)** and **Max 60D Volatility (%)** to fine-tune Buy/Hold/Sell signals.  
- **Timeframes** → Switch between Today, Week, Month, and Year.  
- **Visuals** → Growth charts, Risk vs Return scatter, and a Finviz-style Market Map.  
- **For retail investors** → Designed to be lightweight, simple, and easy to interpret.  

## ⚙️ Installation

Clone the repo and install dependencies:

```bash
git clone https://github.com/your-username/stocks-analyzer.git
cd stocks-analyzer
pip install -r requirements.txt

# conda activate tr
# pip3 install -r requirements.txt
# 
# streamlit run Stock_summary.py 