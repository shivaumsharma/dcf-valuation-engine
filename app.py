import streamlit as st
import matplotlib.pyplot as plt
from DCF import get_stock_data, project_cash_flows, calculate_dcf

st.set_page_config(page_title="DCF Valuation", layout="centered")

st.title("📊 DCF Valuation Engine")
st.write("Analyze whether a stock is undervalued or overvalued.")

# Input
ticker = st.text_input("Enter Stock Ticker", "MSFT")

if st.button("Analyze"):

    try:
        # Fetch data
        data = get_stock_data(ticker)

        current_price = data["current_price"]
        fcf_per_share = data["fcf_per_share"]
        growth_rate = data["growth_rate"]

        # Run model
        projected = project_cash_flows(fcf_per_share, growth_rate, 5)
        result = calculate_dcf(projected, 0.10, 0.025, 5)

        intrinsic = result["intrinsic_value"]
        margin = ((intrinsic - current_price) / current_price) * 100

        # Output
        st.subheader("📈 Results")

        st.write(f"**Current Price:** ${current_price:.2f}")
        st.write(f"**Intrinsic Value:** ${intrinsic:.2f}")
        st.write(f"**Estimated Growth Rate:** {growth_rate:.2%}")

        if intrinsic > current_price:
            st.success(f"UNDERVALUED (+{margin:.2f}%)")
        else:
            st.error(f"OVERVALUED ({margin:.2f}%)")

        # Sensitivity
        st.subheader("📊 Sensitivity Analysis")

        for rate in [0.08, 0.09, 0.10, 0.11, 0.12]:
            res = calculate_dcf(projected, rate, 0.025, 5)
            st.write(f"{int(rate*100)}% → ${res['intrinsic_value']:.2f}")

    except Exception as e:
        st.error(f"Error: {str(e)}")


# Chart
st.subheader("📉 Cash Flow Projection")

years = list(range(1, len(projected) + 1))

fig, ax = plt.subplots()
ax.plot(years, projected, marker='o', label="Future FCF")
ax.plot(years, result["pv_cash_flows"], marker='o', label="Present Value")
ax.fill_between(years, projected, result["pv_cash_flows"], alpha=0.2)

ax.set_xlabel("Year")
ax.set_ylabel("Value per Share")
ax.set_title("Future vs Discounted Cash Flows")
ax.legend()

st.pyplot(fig)