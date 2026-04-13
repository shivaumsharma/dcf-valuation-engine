# ============================================================
#  Stock Valuation Mini DCF Engine
# ============================================================

import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
#  SECTION 1: ASSUMPTIONS
#  All key inputs live here so they're easy to change/discuss
# ============================================================

TICKER          = "MSFT"      # Stock to analyse
DISCOUNT_RATE   = 0.10        # Required rate of return / WACC (10%)
TERMINAL_GROWTH = 0.025       # Long-term stable growth rate after Year 5 (2.5%)
PROJECTION_YEARS = 5          # How many years we project into the future


# ============================================================
#  FUNCTION 1: get_stock_data()
#  Fetches current price and last known Free Cash Flow from yfinance
# ============================================================

def get_stock_data(ticker: str) -> dict:
    """
    Downloads stock financials using yfinance.
    Returns current price and free cash flow.
    FCF = Operating Cash Flow - Capital Expenditures
    """
    stock = yf.Ticker(ticker)

    # --- Current market price ---
    info = stock.info
    current_price = info.get("currentPrice") or info.get("regularMarketPrice")
    if current_price is None:
        raise ValueError(f"Could not fetch price for {ticker}")

    # --- Free Cash Flow ---
    # yfinance sometimes exposes FCF directly; otherwise we compute it
    cash_flow = stock.cashflow  # DataFrame: rows = line items, cols = dates
    # --- Latest Free Cash Flow (for valuation) ---
    try:
        if "Free Cash Flow" in cash_flow.index:
         fcf = float(cash_flow.loc["Free Cash Flow"].iloc[0])
        else:
            operating_cf = float(cash_flow.loc["Operating Cash Flow"].iloc[0])
            capex = float(cash_flow.loc["Capital Expenditure"].iloc[0])
            fcf = operating_cf + capex
    except KeyError:
        raise ValueError("Required cash flow line items not found.")

   # --- Historical FCF for growth calculation ---
    try:
        if "Free Cash Flow" in cash_flow.index:
            fcf_series = cash_flow.loc["Free Cash Flow"].dropna()
        else:
            operating_cf = cash_flow.loc["Operating Cash Flow"]
            capex = cash_flow.loc["Capital Expenditure"]
            fcf_series = (operating_cf + capex).dropna()

        # Take last 4 years (latest first)
        fcf_values = fcf_series.values[:4]

        if len(fcf_values) < 2:
            raise ValueError("Not enough data for growth calculation")
        # CAGR calculation
        start_fcf = fcf_values[-1]
        end_fcf = fcf_values[0]
        years = len(fcf_values) - 1

        growth_rate = (end_fcf / start_fcf) ** (1 / years) - 1

    except Exception:
        raise ValueError("Failed to compute historical FCF")

    # Shares outstanding — needed to convert total FCF → per-share value
    shares = info.get("sharesOutstanding")
    if shares is None or shares == 0:
        raise ValueError("Shares outstanding data unavailable.")

    return {
        "ticker":        ticker,
        "current_price": current_price,
        "fcf_total":     fcf,           # Total FCF in dollars
        "fcf_per_share": fcf / shares,  # FCF per share (like EPS but for cash)
        "shares":        shares,
        "growth_rate":growth_rate
    }


# ============================================================
#  FUNCTION 2: project_cash_flows()
#  Grows FCF year-by-year using the assumed growth rate
# ============================================================

def project_cash_flows(
    base_fcf:       float,
    growth_rate:    float,
    years:          int
) -> list[float]:
    """
    Starting from base_fcf (Year 0), grows it forward for 'years' periods.
    Formula: FCF_n = FCF_0 * (1 + g)^n
    Returns a list of projected FCF values for Years 1 through N.
    """
    projected = []
    for year in range(1, years + 1):
        future_fcf = base_fcf * (1 + growth_rate) ** year
        projected.append(future_fcf)
    return projected


# ============================================================
#  FUNCTION 3: calculate_dcf()
#  The core valuation engine — discounts future cash flows to today
# ============================================================

def calculate_dcf(
    projected_fcfs:  list[float],
    discount_rate:   float,
    terminal_growth: float,
    years:           int
) -> dict:
    """
    Discounts each projected FCF back to present value (PV).
    Then calculates Terminal Value using the Gordon Growth Model.
    Sums everything to get total intrinsic value.

    PV formula:   PV = CF / (1 + r)^t
    Terminal Val: TV = FCF_last * (1 + g) / (r - g)
    """

    # --- Step A: Discount each year's FCF to present value ---
    pv_cash_flows = []
    for t, fcf in enumerate(projected_fcfs, start=1):
        pv = fcf / (1 + discount_rate) ** t
        pv_cash_flows.append(pv)

    sum_pv_fcf = sum(pv_cash_flows)  # Total PV of operating cash flows

    # --- Step B: Terminal Value ---
    # After Year 5 the business doesn't stop — it keeps growing at terminal_growth forever.
    # We capture all that future value in a single "terminal value" number.
    last_fcf       = projected_fcfs[-1]
    terminal_value = last_fcf * (1 + terminal_growth) / (discount_rate - terminal_growth)

    # Discount terminal value back to today (it's received at the end of Year N)
    pv_terminal = terminal_value / (1 + discount_rate) ** years

    # --- Step C: Intrinsic Value = PV of FCFs + PV of Terminal Value ---
    intrinsic_value = sum_pv_fcf + pv_terminal

    return {
        "pv_cash_flows":  pv_cash_flows,
        "sum_pv_fcf":     sum_pv_fcf,
        "terminal_value": terminal_value,
        "pv_terminal":    pv_terminal,
        "intrinsic_value": intrinsic_value,
    }


# ============================================================
#  FUNCTION 4: sensitivity_analysis()
#  Shows how intrinsic value changes as discount rate varies
# ============================================================

def sensitivity_analysis(
    projected_fcfs:  list[float],
    terminal_growth: float,
    years:           int,
    base_rate:       float,
    steps:           list[float] = [-0.02, -0.01, 0, 0.01, 0.02]
) -> pd.DataFrame:
    """
    Runs calculate_dcf() across a range of discount rates.
    Helps show how sensitive the valuation is to this key assumption.
    """
    rows = []
    for delta in steps:
        rate = base_rate + delta
        result = calculate_dcf(projected_fcfs, rate, terminal_growth, years)
        rows.append({
            "Discount Rate": f"{rate:.0%}",
            "Intrinsic Value (per share)": round(result["intrinsic_value"], 2),
        })
    return pd.DataFrame(rows)


# ============================================================
#  FUNCTION 5: main()
#  Orchestrates everything and prints clean output
# ============================================================

def main():
    print("=" * 55)
    print(f"  DCF Valuation Engine — {TICKER}")
    print("=" * 55)

    # Step 1 — Fetch data
    print("\n[1] Fetching stock data...")
    data = get_stock_data(TICKER)

    current_price = data["current_price"]
    fcf_per_share = data["fcf_per_share"]

    print(f"    Current Price   : ${current_price:,.2f}")
    print(f"    FCF (total)     : ${data['fcf_total']:,.0f}")
    print(f"    FCF per share   : ${fcf_per_share:,.4f}")
    print(f"    Shares Out.     : {data['shares']:,.0f}")

    # Step 2 — Project cash flows
    print("\n[2] Projecting Free Cash Flows (per share)...")
    growth_rate = data["growth_rate"]

    print(f"    Estimated Growth Rate: {growth_rate:.2%}")

    projected = project_cash_flows(fcf_per_share, growth_rate, PROJECTION_YEARS)

    for i, fcf in enumerate(projected, start=1):
        print(f"    Year {i}: ${fcf:,.4f}")

    # Step 3 — DCF calculation
    print("\n[3] Discounting cash flows...")
    result = calculate_dcf(projected, DISCOUNT_RATE, TERMINAL_GROWTH, PROJECTION_YEARS)

    print(f"    PV of FCFs      : ${result['sum_pv_fcf']:,.2f}")
    print(f"    Terminal Value  : ${result['terminal_value']:,.2f}")
    print(f"    PV of Terminal  : ${result['pv_terminal']:,.2f}")

    # Step 4 — Output verdict
    intrinsic = result["intrinsic_value"]
    margin    = ((intrinsic - current_price) / current_price) * 100

    print("\n" + "=" * 55)
    print(f"  Current Price   : ${current_price:,.2f}")
    print(f"  Intrinsic Value : ${intrinsic:,.2f}")
    print(f"  Difference      : {margin:+.1f}%")

    if intrinsic > current_price:
        print(f"\n  VERDICT: UNDERVALUED — potential upside of {margin:.1f}%")
    else:
        print(f"\n  VERDICT: OVERVALUED — trading {abs(margin):.1f}% above fair value")
    print("=" * 55)

    # Step 5 — Sensitivity table
    print("\n[4] Sensitivity Analysis — Discount Rate vs Intrinsic Value")
    sensitivity = sensitivity_analysis(
        projected, TERMINAL_GROWTH, PROJECTION_YEARS, DISCOUNT_RATE
    )
    print(sensitivity.to_string(index=False))
    print()
    years = list(range(1, len(projected) + 1))

    pv = result["pv_cash_flows"]
    plt.plot(years, projected, marker='o', label="Future FCF", color="green")
    plt.plot(years, pv, marker='o', label="Present Value", color="red")
    plt.fill_between(years, projected, pv, color="gray", alpha=0.2)
    plt.title("Future vs Discounted Cash Flows")
    plt.xlabel("Year")
    plt.ylabel("Value per Share")
    plt.legend()
    plt.grid(True)
    plt.show()
    

# ============================================================
#  Entry point
# ============================================================

if __name__ == "__main__":
    main()