# dcf-valuation-engine

A Python-based Discounted Cash Flow (DCF) valuation engine that computes intrinsic stock value using real financial data. Includes automated data fetching, cash flow projection, sensitivity analysis, and visualization for investment decision support.

Example Output

Ticker: MSFT  
Current Price: $420.15  
Intrinsic Value: $381.72  
Verdict: Overvalued (-9.1%)

Sensitivity Analysis

Discount Rate → Intrinsic Value

8% → $452.30  
9% → $412.85  
10% → $381.72  
11% → $354.10  
12% → $329.55

## Validation methodology

Earlier versions of this project validated on 20 hand-picked stocks and
reported "% agreement with analyst price targets" as an accuracy number.
Two problems with that: n=20 is too small to support a directional-accuracy
claim, and agreement-with-analysts measures agreement with other humans,
not correctness — it says nothing about whether the valuation call was
actually right.

The validation pipeline now:

- Runs against a **100-stock universe** (`universe.py`) spanning US and
  Indian large caps across sectors, instead of 20 tickers.
- Grades every verdict against **realized forward price movement**
  (`backtest_engine.py`), not analyst consensus. A call is graded
  UNDERVALUED/OVERVALUED/FAIRLY VALUED based on what the stock actually
  did over the following ~6 months, using a ±5% band to define "moved".
- Reports a **confusion matrix and per-class precision/recall/F1**
  (`ml_classifier_metrics.json`, `backtest_report.txt`) instead of one
  blended accuracy number, so it's visible which class the model is
  actually good at calling.

Two ways to produce that realized-return ground truth are included:

1. **Walk-forward (rigorous):** `python backtest_engine.py generate` logs
   today's verdict + price for every ticker to `calls_log.csv`. Run
   `python backtest_engine.py evaluate` again ≥6 months later to grade
   those calls against what actually happened. This is a genuine backtest;
   it just takes 6 months to mature, like any real forward-test would.
2. **Historical demo (approximation, for an immediate report):**
   `python backtest_engine.py demo` uses a price from ~6 months ago as the
   "call price" and today's price as the outcome, but reuses *today's*
   fundamentals to generate the verdict, since yfinance's free tier doesn't
   expose point-in-time historical financials. Useful to sanity-check the
   pipeline now; the walk-forward result is the one to cite as a metric.

## Where the ML is

The original pipeline was DCF math + Monte Carlo sampling + an if/else
threshold — solid quant finance engineering, but no learned model
anywhere. The threshold step has been replaced with a trained classifier:

- **Features** (`feature_engineering.py`): growth rate, discount rate,
  beta, revenue growth, FCF yield, DCF/price and Monte-Carlo-mean/price
  ratios, MC dispersion, probability-undervalued, net-cash/price, and
  margin vs. the old threshold — all things the pipeline already computed
  and previously threw away after one comparison.
- **Models** (`ml_classifier.py`): a Logistic Regression baseline compared
  against XGBoost, evaluated with stratified k-fold cross-validation plus
  a held-out test set.
- **Labels:** the realized forward-return outcome from the backtest above
  — so the backtest isn't just a report, it's the training set.
- **Training:** `python train_classifier.py` (reads `evaluated_calls.csv`,
  writes `valuation_classifier.joblib` + `ml_classifier_metrics.json`).
- The Streamlit app (`app.py`) shows the ML verdict + model confidence
  alongside the original threshold verdict once a model has been trained,
  and flags when the two disagree.

Resume bullet this supports:

> Replaced threshold-based valuation classification with a gradient-boosted
> classifier (XGBoost) trained on fundamentals + DCF-derived features,
> validated via walk-forward backtesting against realized 6-month forward
> returns (not analyst consensus) across a 100-stock universe, with
> per-class precision/recall/F1 reported alongside overall accuracy.

Note the metrics need `evaluate_calls()` to actually mature (or the demo
run) before they're real numbers — don't put a specific F1/accuracy figure
on the resume until `ml_classifier_metrics.json` exists and n ≥ ~40-50.
