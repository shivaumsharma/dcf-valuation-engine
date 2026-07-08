# ============================================================
#  Stock Valuation Mini DCF Engine
# ============================================================

import yfinance as yf
import numpy as np
import pandas as pd
try:
    import matplotlib.pyplot as plt
except:
    plt = None
import plotly.graph_objects as go

# ============================================================
#  SECTION 1: ASSUMPTIONS
#  All key inputs live here so they're easy to change/discuss
# ============================================================

TICKER          = "MSFT"      # Stock to analyse
TERMINAL_GROWTH = 0.025       # Long-term stable growth rate after Year 5 (2.5%)
PROJECTION_YEARS= 5           # How many years we project into the future
MARGIN_OF_SAFETY=0.15


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
    is_indian = ticker.endswith(".NS") or ticker.endswith(".BO")
    

    # --- Current market price ---
    info = stock.info
    beta = info.get("beta", 1.0)
    current_price = info.get("currentPrice") or info.get("regularMarketPrice")
    if current_price is None:
        raise ValueError(f"Could not fetch price for {ticker}")

    # --- Free Cash Flow ---
    # yfinance sometimes exposes FCF directly; otherwise we compute it
    cash_flow = stock.cashflow  # DataFrame: rows = line items, cols = dates
    # --- Latest Free Cash Flow (for valuation) ---

    cash = info.get("totalCash", 0)
    debt = info.get("totalDebt", 0)
    try:
        if "Free Cash Flow" in cash_flow.index:
         fcf = float(cash_flow.loc["Free Cash Flow"].iloc[0])
        else:
            operating_cf = float(cash_flow.loc["Operating Cash Flow"].iloc[0])
            capex = float(cash_flow.loc["Capital Expenditure"].iloc[0])
            fcf = operating_cf + capex
    except KeyError:
        try:

            net_income = float(
                stock.financials.loc["Net Income"].iloc[0]
            )

            fcf = net_income * 0.7

        except Exception:

            raise ValueError(
                "Unable to compute FCF or fallback FCF."
            )
            
        

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
        if start_fcf <= 0 or end_fcf <= 0:
            raise ValueError(
                "Cannot calculate CAGR from negative FCF"
            )

        growth_rate = (end_fcf / start_fcf) ** (1 / years) - 1

    except Exception:
        raise ValueError("Failed to compute historical FCF")

    # Shares outstanding — needed to convert total FCF → per-share value
    shares = info.get("sharesOutstanding")
    if shares is None or shares == 0:
        raise ValueError("Shares outstanding data unavailable.")

    revenue_growth = info.get("revenueGrowth")
    analyst_growth = (info.get("revenueGrowth") or info.get("earningsGrowth"))

    return {
        "ticker": ticker,
        "current_price": current_price,
        "fcf_total": fcf,
        "fcf_per_share": fcf / shares,
        "shares": shares,
        "growth_rate": growth_rate,
        "analyst_growth": analyst_growth,
        "revenue_growth":revenue_growth,
        "is_indian": is_indian,
        "cash":cash,
        "debt":debt,
        "beta": beta,
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

def calculate_dcf(projected_fcfs:  list[float],discount_rate:   float,terminal_growth: float,
years:           int) -> dict:
    """
    Discounts each projected FCF back to present value (PV).
    Then calculates Terminal Value using the Gordon Growth Model.
    Sums everything to get total intrinsic value.

    PV formula:   PV = CF / (1 + r)^t
    Terminal Val: TV = FCF_last * (1 + g) / (r - g)
    """
    if discount_rate <= terminal_growth:
        raise ValueError(
            "Discount rate must exceed terminal growth"
        )
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
#  FUNCTION 4: sensitivity_analysis_2d()
#  Shows how both terminal and discount values fluctuated works 
# ============================================================
def sensitivity_analysis_2d(projected_fcfs,base_discount,base_terminal,years):
    discount_rates = [
    base_discount - 0.02,
    base_discount - 0.01,
    base_discount,
    base_discount + 0.01,
    base_discount + 0.02
    ]
    terminal_rates = [
    base_terminal - 0.01,
    base_terminal,
    base_terminal + 0.01
    ]
    data = []
    for discount in discount_rates:

        row = []

        for terminal in terminal_rates:

            result = calculate_dcf(projected_fcfs,discount,terminal,years)

            row.append(
                result["intrinsic_value"]
            )

        data.append(row)
    
    df = pd.DataFrame(data,

    index=[
        f"{r:.1%}"
        for r in discount_rates
    ],

    columns=[
        f"{r:.1%}"
        for r in terminal_rates
    ]
    )
    return df
    
# ============================================================
#  FUNCTION 6: Monte Carlo
#  Runs Monte Carlo simulations for best output
# ============================================================

def run_monte_carlo(iterations,historical_growth,wacc,terminal_g,fcf_per_share,years):
    """
    Monte Carlo DCF simulation.

    Returns:
        numpy array of intrinsic values
    """
    # -----------------------------
    # Sample uncertain assumptions
    # -----------------------------
    growth_samples=np.random.normal(loc=historical_growth,scale=abs(historical_growth)*0.30,size=iterations)
    discount_samples=np.random.normal(loc=wacc,scale=0.015,size=iterations)
    terminal_samples=np.random.normal(loc=terminal_g,scale=0.005,size=iterations)
    intrinsic_values=[]


    # -----------------------------
    # Clip to realistic ranges
    # -----------------------------

    growth_samples=np.clip(growth_samples,-0.10,0.30)
    discount_samples=np.clip(discount_samples,0.05,0.20)
    terminal_samples=np.clip(terminal_samples,0.00,0.05)

    # -----------------------------
    # Remove invalid combinations
    # Discount rate must exceed
    # terminal growth
    # -----------------------------

    valid_mask=(discount_samples>terminal_samples)

    growth_samples=growth_samples[valid_mask]
    discount_samples=discount_samples[valid_mask]
    terminal_samples=terminal_samples[valid_mask]

    for growth,discount,terminal in zip(growth_samples,discount_samples,terminal_samples):
        projected=project_cash_flows(fcf_per_share,growth,years)
        result=calculate_dcf(projected,discount,terminal,years)
        intrinsic_values.append(result["intrinsic_value"])

    return np.array(intrinsic_values)

# ============================================================
#  FUNCTION 7: def Monte_carlo_statistics()
#  Random SCenarios Geneartions gives better predictibility values 
# ============================================================
def monte_carlo_statistics(mc_values,current_price):
    """
    Computes statistics from Monte Carlo outputs.
    """

    return {
        "mean": np.mean(mc_values),
        "median": np.median(mc_values),
        "std_dev": np.std(mc_values),

        "p25": np.percentile(mc_values,25),
        "p75": np.percentile(mc_values,75),

        "ci_lower": np.percentile(mc_values,5),
        "ci_upper": np.percentile(mc_values,95),

        "prob_undervalued":np.mean(mc_values>current_price)
    }

# ============================================================
#  FUNCTION 7: def monte_carlo_statistics()
#  Random SCenarios Geneartions gives better predictibility values 
# ============================================================

def classify_valuation(
    current_price,
    mean_iv,
    margin_of_safety=0.15
):
    """
    Classifies valuation using Monte Carlo mean IV.
    """

    undervalued_threshold = mean_iv * (
        1 - margin_of_safety
    )

    overvalued_threshold = mean_iv * (
        1 + margin_of_safety * 0.5
    )

    if current_price < undervalued_threshold:

        verdict = "UNDERVALUED"

    elif current_price > overvalued_threshold:

        verdict = "OVERVALUED"

    else:

        verdict = "FAIRLY VALUED"

    return {
        "verdict": verdict,
        "undervalued_threshold":
            undervalued_threshold,
        "overvalued_threshold":
            overvalued_threshold
    }
# ============================================================
#  FUNCTION 7: main()
#  Orchestrates everything and prints clean output
# ============================================================
def plot_projection_chart(
    projected_fcfs,
    pv_cash_flows
):
    """
    Interactive Plotly chart for projected FCF vs discounted PV.
    """

    years = list(
        range(
            1,
            len(projected_fcfs) + 1
        )
    )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=years,
            y=projected_fcfs,
            mode="lines+markers",
            name="Projected FCF"
        )
    )

    fig.add_trace(
        go.Scatter(
            x=years,
            y=pv_cash_flows,
            mode="lines+markers",
            name="Present Value"
        )
    )

    fig.update_layout(
        title="Future vs Discounted Cash Flows",
        xaxis_title="Year",
        yaxis_title="Value per Share",
        template="plotly_dark"
    )

    return fig
# ============================================================
#  FUNCTION 7: main()
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
    historical_growth = data["growth_rate"]
    if historical_growth < 0:

        st.warning(
            f"Historical CAGR is "
            f"{historical_growth:.2%}. "
            f"Negative growth may indicate "
            f"deteriorating business performance "
            f"or unreliable data."
        )

    elif historical_growth > 0.40:

        st.warning(
            f"Historical CAGR is "
            f"{historical_growth:.2%}. "
            f"Growth above 40% is often "
            f"unsustainable and should be "
            f"reviewed manually."
        )
    analyst_growth = data.get("analyst_growth")

    # -----------------------------------------
    # choose growth source
    # -----------------------------------------

    if growth_mode == "Historical CAGR":
        growth_rate = historical_growth

    elif growth_mode == "Analyst Estimate":

        if analyst_growth is not None:
            growth_rate = analyst_growth

        else:
            st.warning(
                "Analyst estimate unavailable. "
                "Falling back to historical CAGR."
            )

            growth_rate = historical_growth

    else:
        growth_rate = manual_growth / 100

    print(f"    Estimated Growth Rate: {growth_rate:.2%}")

    projected = project_cash_flows(fcf_per_share, growth_rate, PROJECTION_YEARS)

    for i, fcf in enumerate(projected, start=1):
        print(f"    Year {i}: ${fcf:,.4f}")

    beta = max(0.8, min(data["beta"], 2.0))

    discount_rate = 0.04 + beta * 0.03
    # Step 3 — DCF calculation
    print("\n[3] Discounting cash flows...")
    result = calculate_dcf(projected, discount_rate, TERMINAL_GROWTH, PROJECTION_YEARS)
    net_cash_per_share = ((data["cash"] - data["debt"])/ data["shares"])

    result["intrinsic_value"] += net_cash_per_share

    mc_values = run_monte_carlo(iterations=10000,historical_growth=growth_rate,wacc=discount_rate,terminal_g=TERMINAL_GROWTH,fcf_per_share=fcf_per_share,years=PROJECTION_YEARS)

    if len(mc_values) == 0:
        raise ValueError(
            "No valid Monte Carlo simulations generated."
        )
    stats = monte_carlo_statistics(mc_values,current_price)

    mean_iv = stats["mean"]
    median_iv = stats["median"]
    std_iv = stats["std_dev"]

    p25 = stats["p25"]
    p75 = stats["p75"]

    ci_lower = stats["ci_lower"]
    ci_upper = stats["ci_upper"]


    print("\nMonte Carlo Statistics")
    print("=" * 55)
    print(f"    Mean IV      : ${mean_iv:.2f}")
    print(f"    Median IV    : ${median_iv:.2f}")
    print(f"    Std Dev      : ${std_iv:.2f}")
    print(f"    P25          : ${p25:.2f}")
    print(f"    P75          : ${p75:.2f}")
    print(f"    90% CI Lower : ${ci_lower:.2f}")
    print(f"    90% CI Upper : ${ci_upper:.2f}")
    print(f"    Probability Undervalued:"f"{stats["prob_undervalued"]:.1%}")
    print(f"    MC Samples: {len(mc_values)}")
    print(f"    MC Mean IV: {mean_iv:.2f}")

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
    
    valuation = classify_valuation(current_price=current_price,mean_iv=mean_iv,margin_of_safety=MARGIN_OF_SAFETY)
    print("\n Margin of Safety Analysis")
    print("-" * 40)

    print(f"Margin of Safety : {MARGIN_OF_SAFETY:.0%}")
    print(f"Classification : "f"{valuation['verdict']}")
    

    sensitivity_2d = sensitivity_analysis_2d(
    projected,
    discount_rate,
    TERMINAL_GROWTH,
    PROJECTION_YEARS
    )

    print("\n2D Sensitivity Grid")

    print(sensitivity_2d.round(2))

    # Step 5 — Sensitivity table
    print("\n[4] Sensitivity Analysis — Discount Rate vs Intrinsic Value")
    sensitivity = sensitivity_analysis(
        projected, TERMINAL_GROWTH, PROJECTION_YEARS, discount_rate
    )
    print(sensitivity.to_string(index=False))
    print()
    fig = plot_projection_chart(
    projected,
    result["pv_cash_flows"]
    )

    fig.show()
        

# ============================================================
#  Entry point
# ============================================================

# if __name__ == "__main__":
#     main()
    
    