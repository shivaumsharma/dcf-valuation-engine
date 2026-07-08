import os
import streamlit as st
import pandas as pd
import numpy as np 
from dcf_engine import(get_stock_data,project_cash_flows,calculate_dcf,run_monte_carlo,monte_carlo_statistics,classify_valuation,sensitivity_analysis_2d,plot_projection_chart,TERMINAL_GROWTH,PROJECTION_YEARS)
from feature_engineering import compute_features_for_ticker
from ml_classifier import predict_verdict, MODEL_PATH

@st.cache_data(ttl=3600)
def cached_stock_data(ticker):
    return get_stock_data(ticker)


st.set_page_config(page_title="DCF Valuation", layout="centered")

st.title("📊 DCF Valuation Engine")
st.write("Analyze whether a stock is undervalued or overvalued.")

tab1, tab2 = st.tabs(
    [
        "📈 Single Stock",
        "📋 Batch Screener"
    ]
)
with tab1:
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
                print(f"{ticker} | "f"FCF CAGR={data['growth_rate']:.2%} | "f"Analyst={data.get('analyst_growth')} | "f"Revenue={data.get('revenue_growth')}")
                currency_symbol = ("₹"if data.get("is_indian", False)else "$")

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
                revenue_growth = data.get("revenue_growth")
                # -----------------------------------------
                # choose growth source
                # -----------------------------------------

                if growth_mode == "Historical CAGR":
                        growth_inputs = []

                        if historical_growth is not None:
                            growth_inputs.append(historical_growth)

                        if analyst_growth is not None:
                            growth_inputs.append(analyst_growth)

                        if revenue_growth is not None:
                            growth_inputs.append(revenue_growth)

                        growth_rate = np.mean(growth_inputs)

                        growth_rate = np.clip(growth_rate,-0.10,0.20)

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
                beta=data["beta"]
                discount_rate=0.04+beta*0.03
                result = calculate_dcf(projected,discount_rate,TERMINAL_GROWTH,PROJECTION_YEARS)
                net_cash_per_share = ((data["cash"] - data["debt"])/ data["shares"])

                result["intrinsic_value"] += net_cash_per_share
                mc_values = run_monte_carlo(
                iterations=10000,
                historical_growth=growth_rate,
                wacc=discount_rate,
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
                        f"{currency_symbol}{current_price:.2f}"
                    )

                with col2:
                    st.metric(
                        "Base DCF",
                        f"{currency_symbol}{intrinsic:.2f}"
                    )

                with col3:
                    st.metric(
                        "MC Mean IV",
                        f"{currency_symbol}{stats['mean']:.2f}"
                    )
                
                st.subheader("🎲 Monte Carlo Statistics")

                st.write(f"Mean IV: {currency_symbol}{stats['mean']:.2f}")
                st.write(f"Median IV: {currency_symbol}{stats['median']:.2f}")
                st.write(f"Std Dev: {currency_symbol}{stats['std_dev']:.2f}")
                st.write(f"P25: {currency_symbol}{stats['p25']:.2f}")
                st.write(f"P75: {currency_symbol}{stats['p75']:.2f}")
                st.write(
                    f"90% Confidence Interval: "
                    f"{currency_symbol}{stats['ci_lower']:.2f}"
                    f" - "
                    f"{currency_symbol}{stats['ci_upper']:.2f}"
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

                # ---------------------------------------------------
                # ML classifier verdict (optional, shown alongside the
                # threshold verdict once a model has been trained via
                # `python backtest_engine.py generate/evaluate` then
                # `python train_classifier.py`). Backtest report at
                # backtest_report.txt documents how this was validated.
                # ---------------------------------------------------
                st.subheader("🤖 ML Classifier Verdict")

                if os.path.exists(MODEL_PATH):
                    feature_row = compute_features_for_ticker(ticker, mc_iterations=3000)
                    if feature_row is not None:
                        ml_result = predict_verdict(feature_row)
                        if ml_result is not None:
                            ml_col1, ml_col2 = st.columns(2)
                            with ml_col1:
                                st.metric("ML Verdict", ml_result["verdict"])
                            with ml_col2:
                                conf = ml_result["probabilities"].get(ml_result["verdict"], 0)
                                st.metric("Model Confidence", f"{conf:.0%}")
                            st.caption(
                                f"Model: {ml_result['model_name']} — trained on realized "
                                f"forward-return outcomes, not analyst consensus. "
                                f"See backtest_report.txt for held-out precision/recall/F1."
                            )
                            if ml_result["verdict"] != valuation["verdict"]:
                                st.info(
                                    "Note: the ML classifier and the threshold rule "
                                    "disagree here — worth digging into why before "
                                    "trusting either one blindly."
                                )
                    else:
                        st.caption("Could not compute ML features for this ticker.")
                else:
                    st.caption(
                        "No trained ML classifier found yet. Run "
                        "`python backtest_engine.py generate`, wait for calls to "
                        "mature (or use `demo` mode for an immediate approximate "
                        "run), then `python backtest_engine.py evaluate` and "
                        "`python train_classifier.py` to enable this panel."
                    )

                # Sensitivity
                st.subheader("📊 2D Sensitivity Grid")

                grid = sensitivity_analysis_2d(
                    projected,
                    discount_rate,
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
        
with tab2:
    st.subheader("Batch Stock Screener")

    tickers_text=st.text_area("Enter tickers seperated by commas","AAPL,MSFT,GOOGL,META,INFY.NS,TCS.NS,RELIANCE.NS")

    run_screen=st.button("Run Batch Analysis")

    if run_screen:
        tickers=[t.strip().upper()for t in tickers_text.split(",")if t.strip()]

        progress=st.progress(0)

        results=[]
        for idx, ticker in enumerate(tickers):

            try:
                data = cached_stock_data(ticker)
                print("Beta:", data["beta"])
                current_price = data["current_price"]
                historical_growth = data["growth_rate"]
                analyst_growth = data.get("analyst_growth")
                revenue_growth = data.get("revenue_growth")

                growth_inputs = []

                if historical_growth is not None:
                    growth_inputs.append(historical_growth)

                if analyst_growth is not None:
                    growth_inputs.append(analyst_growth)

                if revenue_growth is not None:
                    growth_inputs.append(revenue_growth)

                if len(growth_inputs) > 0:
                    growth_rate = sum(growth_inputs) / len(growth_inputs)
                else:
                    growth_rate = 0.05

                growth_rate = max(-0.10, min(growth_rate, 0.20))
            
                beta = max(0.8, min(data["beta"], 2.0))

                discount_rate = 0.04 + beta * 0.03
                fcf_per_share = data["fcf_per_share"]
                mc_values = run_monte_carlo(iterations=3000,historical_growth=growth_rate,wacc=discount_rate,terminal_g=TERMINAL_GROWTH,fcf_per_share=fcf_per_share,years=PROJECTION_YEARS)

                if len(mc_values) == 0:
                        raise ValueError("No simulations generated")

                stats = monte_carlo_statistics(mc_values,current_price)

                valuation = classify_valuation(current_price=current_price,mean_iv=stats["mean"],margin_of_safety=0.15)

                upside = ((stats["mean"] - current_price)/ current_price) * 100

                row = {"Ticker":ticker,"Current Price":round(current_price, 2),"MC Mean IV":round(stats["mean"], 2),"Upside %":round(upside, 2),"Underval Prob %":round(stats["prob_undervalued"] * 100,1),"90% CI":f"{stats['ci_lower']:.0f}"f" - "f"{stats['ci_upper']:.0f}","Verdict":valuation["verdict"]}

                if os.path.exists(MODEL_PATH):
                    feature_row = compute_features_for_ticker(ticker, mc_iterations=1500)
                    ml_result = predict_verdict(feature_row) if feature_row is not None else None
                    row["ML Verdict"] = ml_result["verdict"] if ml_result else "N/A"

                results.append(row)

            except Exception as e:
                print(f"{ticker}: {e}")
                results.append({
                            "Ticker":
                                ticker,
                            "Current Price":
                                None,
                            "MC Mean IV":
                                None,
                            "Upside %":
                                None,
                            "Underval Prob %":
                                None,
                            "90% CI":
                               None,
                            "Verdict":
                                "Data unavailable"
                        })
                
                progress.progress((idx + 1)/ len(tickers))
                df = pd.DataFrame(results)
                df["sort_col"] = pd.to_numeric(df["Upside %"],errors="coerce")
                df = df.sort_values("sort_col",ascending=False)
                df = df.drop(columns=["sort_col"])

                st.subheader("📊 Screening Results")

                st.dataframe(df,width="stretch",hide_index=True)