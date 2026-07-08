# ============================================================
#  universe.py
#  A ~100-stock universe spanning US + Indian large caps across
#  sectors, so validation isn't run on 20 hand-picked names.
#  Swap/extend freely -- the only requirement is that yfinance
#  recognizes the ticker.
# ============================================================

US_TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA",
    "JPM", "V", "MA", "HD", "UNH", "XOM", "CVX",
    "KO", "PEP", "WMT", "COST", "MCD", "NKE", "DIS", "NFLX",
    "ADBE", "CRM", "ORCL", "INTC", "AMD", "QCOM", "CSCO", "IBM",
    "TXN", "AVGO", "PYPL", "BA", "CAT", "GE", "HON", "MMM",
    "LMT", "RTX", "UPS", "FDX", "T", "VZ", "TMO", "ABBV",
    "PFE", "MRK", "LLY", "BMY", "GILD", "AMGN", "SBUX", "TGT",
    "LOW", "CMG", "BKNG", "ABT", "JNJ", "PG",
]

INDIA_TICKERS = [
    "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS",
    "HINDUNILVR.NS", "ITC.NS", "KOTAKBANK.NS", "LT.NS", "SBIN.NS",
    "BHARTIARTL.NS", "ASIANPAINT.NS", "MARUTI.NS", "AXISBANK.NS",
    "BAJFINANCE.NS", "WIPRO.NS", "HCLTECH.NS", "ULTRACEMCO.NS",
    "SUNPHARMA.NS", "TITAN.NS", "NESTLEIND.NS", "TATAMOTORS.NS",
    "TATASTEEL.NS", "POWERGRID.NS", "NTPC.NS", "ONGC.NS",
    "COALINDIA.NS", "ADANIENT.NS", "ADANIPORTS.NS", "JSWSTEEL.NS",
    "GRASIM.NS", "HDFCLIFE.NS", "SBILIFE.NS", "DIVISLAB.NS",
    "DRREDDY.NS", "CIPLA.NS", "BRITANNIA.NS", "EICHERMOT.NS",
    "HEROMOTOCO.NS", "BAJAJ-AUTO.NS",
]

FULL_UNIVERSE = US_TICKERS + INDIA_TICKERS  # 100 tickers total

if __name__ == "__main__":
    print(f"US tickers:    {len(US_TICKERS)}")
    print(f"India tickers: {len(INDIA_TICKERS)}")
    print(f"Total:         {len(FULL_UNIVERSE)}")
