# ============================================================
#  backtest_engine.py
#
#  Fixes the two validation problems in the original project:
#    1. n=20  -> runs against the ~100-ticker universe.py list
#    2. "agrees with analyst consensus" -> ground truth is now
#       REALIZED forward price movement, not another human's opinion.
#
#  Two ways to get ground truth, both included:
#
#  A) WALK-FORWARD (the methodologically correct one):
#     `generate_calls()` logs today's verdict + price for every ticker
#     in the universe to calls_log.csv. Run this once now.
#     `evaluate_calls()` is run again >= 6 months later: it looks up
#     each ticker's CURRENT price, computes the realized return since
#     the call, and grades the original verdict against it. This is a
#     genuine, non-fabricated backtest -- it just takes 6 months to
#     mature, same as any real forward-testing process would.
#
#  B) HISTORICAL DEMO (for an immediate report to look at today):
#     `historical_demo_backtest()` uses the price from N months ago as
#     the "call price" and today's price as the outcome, but reuses
#     TODAY's fundamentals to generate the verdict (yfinance's free
#     tier doesn't expose point-in-time historical fundamentals). This
#     is clearly labeled as an approximation -- useful to sanity-check
#     the pipeline and produce a report now, but the walk-forward
#     result (A) is the one that belongs on a resume/interview claim.
#
#  Ground truth label definition (both modes):
#     realized_return = (price_now / price_then) - 1
#     UNDERVALUED   if realized_return >  +REALIZED_MARGIN
#     OVERVALUED    if realized_return <  -REALIZED_MARGIN
#     FAIRLY VALUED otherwise
# ============================================================

from __future__ import annotations

import argparse
import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.metrics import classification_report, confusion_matrix

from feature_engineering import compute_features_for_ticker, FEATURE_COLUMNS
from universe import FULL_UNIVERSE

CALLS_LOG_PATH = "calls_log.csv"
EVALUATED_PATH = "evaluated_calls.csv"
REPORT_PATH = "backtest_report.txt"

REALIZED_MARGIN = 0.05     # +/-5% forward move defines "right" direction
LABELS = ["UNDERVALUED", "FAIRLY VALUED", "OVERVALUED"]


# ------------------------------------------------------------
# A) Walk-forward: step 1 -- log calls today
# ------------------------------------------------------------

def generate_calls(tickers: list[str] = None, log_path: str = CALLS_LOG_PATH,
                    mc_iterations: int = 5000) -> pd.DataFrame:
    """
    Runs the valuation pipeline on every ticker today and appends the
    call (verdict + all features + call date) to calls_log.csv.
    Re-run this periodically (e.g. monthly) to build up an
    ever-growing, honestly-dated set of calls to evaluate later.
    """
    tickers = tickers or FULL_UNIVERSE
    call_date = datetime.today().strftime("%Y-%m-%d")

    rows = []
    for i, ticker in enumerate(tickers, start=1):
        print(f"[{i}/{len(tickers)}] logging call for {ticker}")
        row = compute_features_for_ticker(ticker, mc_iterations=mc_iterations)
        if row is None:
            continue
        row["call_date"] = call_date
        rows.append(row)

    new_df = pd.DataFrame(rows)
    if os.path.exists(log_path):
        existing = pd.read_csv(log_path)
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df

    combined.to_csv(log_path, index=False)
    print(f"\nLogged {len(new_df)} calls -> {log_path} "
          f"({len(combined)} total rows on file)")
    return new_df


# ------------------------------------------------------------
# A) Walk-forward: step 2 -- evaluate calls that are old enough
# ------------------------------------------------------------

def _latest_price(ticker: str) -> float | None:
    try:
        hist = yf.Ticker(ticker).history(period="5d")
        if hist.empty:
            return None
        return float(hist["Close"].iloc[-1])
    except Exception as exc:
        print(f"  [skip] {ticker}: could not fetch latest price ({exc})")
        return None


def _realized_label(realized_return: float, margin: float = REALIZED_MARGIN) -> str:
    if realized_return > margin:
        return "UNDERVALUED"
    elif realized_return < -margin:
        return "OVERVALUED"
    return "FAIRLY VALUED"


def evaluate_calls(log_path: str = CALLS_LOG_PATH, min_age_days: int = 182,
                    realized_margin: float = REALIZED_MARGIN) -> pd.DataFrame | None:
    """
    Loads calls_log.csv, keeps only calls old enough to have a
    meaningful forward return, fetches today's price for each, and
    grades the original threshold_verdict against the REALIZED label.

    Writes evaluated_calls.csv and backtest_report.txt (confusion
    matrix + per-class precision/recall/F1).
    """
    if not os.path.exists(log_path):
        print(f"No {log_path} found. Run generate_calls() first.")
        return None

    log = pd.read_csv(log_path)
    log["call_date"] = pd.to_datetime(log["call_date"])
    cutoff = datetime.today() - timedelta(days=min_age_days)
    eligible = log[log["call_date"] <= cutoff].copy()

    if eligible.empty:
        print(f"No calls in {log_path} are >= {min_age_days} days old yet. "
              f"Oldest call on file: {log['call_date'].min().date() if len(log) else 'n/a'}. "
              f"Come back after {cutoff.date()}.")
        return None

    rows = []
    for _, r in eligible.iterrows():
        price_now = _latest_price(r["ticker"])
        if price_now is None:
            continue
        price_then = r["current_price"]
        realized_return = (price_now / price_then) - 1
        realized_label = _realized_label(realized_return, realized_margin)
        rows.append({
            **r.to_dict(),
            "price_now": price_now,
            "realized_return": realized_return,
            "realized_label": realized_label,
        })

    evaluated = pd.DataFrame(rows)
    evaluated.to_csv(EVALUATED_PATH, index=False)
    _write_report(evaluated)
    print(f"\nEvaluated {len(evaluated)} calls -> {EVALUATED_PATH}, {REPORT_PATH}")
    return evaluated


# ------------------------------------------------------------
# B) Historical demo backtest (produces a report immediately)
# ------------------------------------------------------------

def historical_demo_backtest(tickers: list[str] = None, months_back: int = 6,
                              realized_margin: float = REALIZED_MARGIN,
                              mc_iterations: int = 5000) -> pd.DataFrame:
    """
    APPROXIMATION, clearly labeled as such in the report: uses the
    closing price from ~months_back months ago as the "call price",
    computes the verdict using TODAY's fundamentals (not point-in-time
    fundamentals -- a documented limitation of yfinance's free tier),
    and grades it against today's price. Useful for an immediate demo
    report; the walk-forward result (generate_calls/evaluate_calls) is
    the rigorous one to cite as a resume metric.
    """
    tickers = tickers or FULL_UNIVERSE
    rows = []
    for i, ticker in enumerate(tickers, start=1):
        print(f"[{i}/{len(tickers)}] {ticker}")
        try:
            hist = yf.Ticker(ticker).history(period=f"{months_back * 31 + 15}d")
            if hist.empty or len(hist) < 5:
                continue
            price_then = float(hist["Close"].iloc[0])
            price_now = float(hist["Close"].iloc[-1])

            row = compute_features_for_ticker(
                ticker, mc_iterations=mc_iterations, price_override=price_then
            )
            if row is None:
                continue

            realized_return = (price_now / price_then) - 1
            row["price_now"] = price_now
            row["realized_return"] = realized_return
            row["realized_label"] = _realized_label(realized_return, realized_margin)
            row["call_date"] = hist.index[0].strftime("%Y-%m-%d")
            rows.append(row)
        except Exception as exc:
            print(f"  [skip] {ticker}: {exc}")

    evaluated = pd.DataFrame(rows)
    evaluated.to_csv("evaluated_calls_demo.csv", index=False)
    _write_report(evaluated, report_path="backtest_report_demo.txt",
                   header="HISTORICAL DEMO BACKTEST (approximation -- see docstring; "
                          "do not cite this number, use the walk-forward result instead)")
    print(f"\nDemo-evaluated {len(evaluated)} calls -> evaluated_calls_demo.csv, "
          f"backtest_report_demo.txt")
    return evaluated


# ------------------------------------------------------------
# Shared reporting: confusion matrix + precision/recall/F1
# ------------------------------------------------------------

def _write_report(evaluated: pd.DataFrame, report_path: str = REPORT_PATH,
                   header: str = "WALK-FORWARD BACKTEST (realized forward return "
                                  "vs. logged verdict)") -> None:
    if evaluated.empty:
        print("Nothing to report -- evaluated set is empty.")
        return

    y_true = evaluated["realized_label"]
    y_pred = evaluated["threshold_verdict"]

    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    cm_df = pd.DataFrame(cm, index=[f"true_{l}" for l in LABELS],
                          columns=[f"pred_{l}" for l in LABELS])

    report = classification_report(y_true, y_pred, labels=LABELS, zero_division=0)
    accuracy = (y_true == y_pred).mean()

    lines = [
        "=" * 60,
        header,
        "=" * 60,
        f"Stocks evaluated : {len(evaluated)}",
        f"Overall accuracy : {accuracy:.1%}  (a single blended number like this "
        f"is exactly what hides class-level weaknesses -- see confusion "
        f"matrix and per-class report below for what it's masking)",
        "",
        "Confusion matrix (rows = realized/true, cols = predicted):",
        cm_df.to_string(),
        "",
        "Per-class precision / recall / F1:",
        report,
    ]
    text = "\n".join(lines)
    with open(report_path, "w") as f:
        f.write(text)
    print(text)


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DCF backtest engine")
    parser.add_argument(
        "mode", choices=["generate", "evaluate", "demo"],
        help="generate: log today's calls. evaluate: grade calls >=6mo old "
             "against realized returns. demo: immediate approximate backtest."
    )
    parser.add_argument("--min-age-days", type=int, default=182)
    parser.add_argument("--months-back", type=int, default=6)
    parser.add_argument("--mc-iterations", type=int, default=5000)
    args = parser.parse_args()

    if args.mode == "generate":
        generate_calls(mc_iterations=args.mc_iterations)
    elif args.mode == "evaluate":
        evaluate_calls(min_age_days=args.min_age_days)
    elif args.mode == "demo":
        historical_demo_backtest(months_back=args.months_back, mc_iterations=args.mc_iterations)
