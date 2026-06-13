import streamlit as st

from dcf_engine import(get_stock_data,project_cash_flows,calculate_dcf,run_monte_carlo,monte_carlo_statistics,classify_valuation,sensitivity_analysis_2d,plot_projection_chart,DISCOUNT_RATE,TERMINAL_GROWTH,PROJECTION_YEARS)

@st.cache_data(ttl=3600)
def cached_stock_data(ticker):
    return get_stock_data(ticker)


st.set_page_config(page_title="DCF Valuation", layout="centered")

st.title("📊 DCF Valuation Engine")
st.write("Analyze whether a stock is undervalued or overvalued.")

# Input
ticker = st.text_input("Enter Stock Ticker", "MSFT").upper()

# ---------------------------------------------------
# Growth assumption selection
# ---------------------------------------------------

growth_mode = st.sidebar.radio(
    "Growth Assumption",
    [
        "Historical CAGR",
        "Analyst Estimate",
        "Manual Input"
    ]
)

manual_growth = st.sidebar.slider(
    "Manual Growth Rate (%)",
    min_value=0,
    max_value=50,
    value=10,
    step=1
)
if st.button("Analyze"):

    try:
        with st.spinner("Running DCF Analysis..."):
        # Fetch data
            data = cached_stock_data(ticker)

            current_price = data["current_price"]
            fcf_per_share = data["fcf_per_share"]
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
            
            with st.expander("Growth Assumption Details"):
                    st.write(f"Historical CAGR: "f"{historical_growth:.2%}")

                    if analyst_growth is not None:
                        st.write(f"Analyst Estimate: "f"{analyst_growth:.2%}")
                    else:
                        st.write( "Analyst Estimate: Not Available")

            # Run model
            projected = project_cash_flows(fcf_per_share, growth_rate, PROJECTION_YEARS)
            result = calculate_dcf(projected,DISCOUNT_RATE,TERMINAL_GROWTH,PROJECTION_YEARS)
            mc_values = run_monte_carlo(
            iterations=10000,
            historical_growth=growth_rate,
            wacc=DISCOUNT_RATE,
            terminal_g=TERMINAL_GROWTH,
            fcf_per_share=fcf_per_share,
            years=PROJECTION_YEARS)

            if len(mc_values) == 0:
                raise ValueError(
                    "No valid Monte Carlo simulations generated."
                )

            stats = monte_carlo_statistics(mc_values,current_price)

            valuation = classify_valuation(
                current_price=current_price,
                mean_iv=stats["mean"],
                margin_of_safety=0.15)

            intrinsic = result["intrinsic_value"]
            
            # Output
            st.subheader("📈 Results")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "Current Price",
                    f"${current_price:.2f}"
                )

            with col2:
                st.metric(
                    "Base DCF",
                    f"${intrinsic:.2f}"
                )

            with col3:
                st.metric(
                    "MC Mean IV",
                    f"${stats['mean']:.2f}"
                )
            
            st.subheader("🎲 Monte Carlo Statistics")

            st.write(f"Mean IV: ${stats['mean']:.2f}")
            st.write(f"Median IV: ${stats['median']:.2f}")
            st.write(f"Std Dev: ${stats['std_dev']:.2f}")
            st.write(f"P25: ${stats['p25']:.2f}")
            st.write(f"P75: ${stats['p75']:.2f}")
            st.write(
                f"90% Confidence Interval: "
                f"${stats['ci_lower']:.2f}"
                f" - "
                f"${stats['ci_upper']:.2f}"
            )
            st.write(
                f"Probability Undervalued: "
                f"{stats['prob_undervalued']:.1%}"
            )

            st.write(f"Growth Assumption Used: "f"{growth_rate:.2%}")
            st.write(f"Source: {growth_mode}")
                    

            if valuation["verdict"] == "UNDERVALUED":
                st.success("UNDERVALUED")

            elif valuation["verdict"] == "OVERVALUED":
                st.error("OVERVALUED")

            else:
                st.warning("FAIRLY VALUED")

            # Sensitivity
            st.subheader("📊 2D Sensitivity Grid")

            grid = sensitivity_analysis_2d(
                projected,
                DISCOUNT_RATE,
                TERMINAL_GROWTH,
                PROJECTION_YEARS
            )

            st.dataframe(
                grid,
                width="stretch"
            )
            
            # Chart
            st.subheader("📉 Cash Flow Projection")

            fig = plot_projection_chart(projected,result["pv_cash_flows"])

            st.plotly_chart(fig,width="stretch")

    except Exception as e:
        st.error(f"Error: {str(e)}")


        