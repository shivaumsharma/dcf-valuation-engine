# ============================================================
#  feature_engineering.py
#  Turns the existing DCF / Monte Carlo pipeline into a flat
#  feature vector per ticker, so the same numbers that used to
#  be thrown away after one threshold check become ML inputs.
#  No new data sources — this wraps dcf_engine.py.
# ============================================================

from __future__ import annotations

import numpy as np
import pandas as pd

from dcf_engine import (
    get_stock_data,
    project_cash_flows,
    calculate_dcf,
    run_monte_carlo,
    monte_carlo_statistics,
    classify_valuation,
    TERMINAL_GROWTH,
    PROJECTION_YEARS,
)

# Columns actually fed into the ML models (bookkeeping columns like
# 'ticker' / 'current_price' / 'threshold_verdict' are kept in the frame
# for reference but excluded from X at training time).
FEATURE_COLUMNS = [
    "growth_rate",
    "discount_rate",
    "beta",
    "revenue_growth",
    "fcf_yield",
    "dcf_over_price",
    "mc_mean_over_price",
    "mc_std_over_price",
    "prob_undervalued",
    "net_cash_per_share_over_price",
    "margin_vs_undervalued_threshold",
]


def compute_features_for_ticker(
    ticker: str,
    mc_iterations: int = 5000,
    price_override: float | None = None,
) -> dict | None:
    """
    Runs the full valuation pipeline for one ticker (fetch -> project ->
    DCF -> Monte Carlo -> threshold verdict) and flattens the outputs
    into one feature row.

    price_override: if given, use this as "current_price" instead of the
    live price from yfinance. This lets backtest_engine.py simulate what
    the call would have looked like against a historical price, using
    current fundamentals as an approximation (documented limitation --
    yfinance doesn't expose point-in-time historical fundamentals on the
    free tier, so this is an approximation, not a true point-in-time
    backtest).

    Returns None (rather than raising) on any failure so a batch run
    over 80-100 tickers doesn't die on one bad/missing ticker.
    """
    try:
        data = get_stock_data(ticker)
        current_price = price_override if price_override is not None else data["current_price"]
        fcf_per_share = data["fcf_per_share"]

        growth_rate = data.get("growth_rate")
        if growth_rate is None or not np.isfinite(growth_rate):
            growth_rate = data.get("analyst_growth") or 0.05
        growth_rate = float(np.clip(growth_rate, -0.10, 0.30))

        beta = max(0.8, min(float(data.get("beta") or 1.0), 2.0))
        discount_rate = 0.04 + beta * 0.03

        projected = project_cash_flows(fcf_per_share, growth_rate, PROJECTION_YEARS)
        dcf_result = calculate_dcf(projected, discount_rate, TERMINAL_GROWTH, PROJECTION_YEARS)

        net_cash_per_share = (data["cash"] - data["debt"]) / data["shares"]
        base_intrinsic = dcf_result["intrinsic_value"] + net_cash_per_share

        mc_values = run_monte_carlo(
            iterations=mc_iterations,
            historical_growth=growth_rate,
            wacc=discount_rate,
            terminal_g=TERMINAL_GROWTH,
            fcf_per_share=fcf_per_share,
            years=PROJECTION_YEARS,
        )
        if len(mc_values) == 0:
            return None

        stats = monte_carlo_statistics(mc_values, current_price)
        valuation = classify_valuation(current_price, stats["mean"], margin_of_safety=0.15)

        row = {
            "ticker": ticker,
            "current_price": current_price,
            "base_dcf_value": base_intrinsic,
            "mc_mean_iv": stats["mean"],
            # --- ML features ---
            "growth_rate": growth_rate,
            "discount_rate": discount_rate,
            "beta": beta,
            "revenue_growth": float(data.get("revenue_growth") or 0.0),
            "fcf_yield": fcf_per_share / current_price,
            "dcf_over_price": base_intrinsic / current_price,
            "mc_mean_over_price": stats["mean"] / current_price,
            "mc_std_over_price": stats["std_dev"] / current_price,
            "prob_undervalued": stats["prob_undervalued"],
            "net_cash_per_share_over_price": net_cash_per_share / current_price,
            "margin_vs_undervalued_threshold": (stats["mean"] - current_price) / current_price,
            # --- bookkeeping ---
            "threshold_verdict": valuation["verdict"],
        }
        return row
    except Exception as exc:
        print(f"  [skip] {ticker}: {exc}")
        return None


def build_feature_frame(tickers: list[str], mc_iterations: int = 5000) -> pd.DataFrame:
    """Runs compute_features_for_ticker across a list of tickers."""
    rows = []
    for i, ticker in enumerate(tickers, start=1):
        print(f"[{i}/{len(tickers)}] {ticker}")
        row = compute_features_for_ticker(ticker, mc_iterations=mc_iterations)
        if row is not None:
            rows.append(row)
    return pd.DataFrame(rows)
