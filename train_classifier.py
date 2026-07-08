# ============================================================
#  train_classifier.py
#  Entry point for training the ML valuation classifier.
#
#  Usage:
#      python backtest_engine.py generate          # log calls today
#      # ... wait >= 6 months, or use `demo` mode for an immediate,
#      # clearly-labeled approximate report ...
#      python backtest_engine.py evaluate           # grade vs realized returns
#      python train_classifier.py                   # train on evaluated_calls.csv
#
#  Kept as a separate file (rather than `python ml_classifier.py`) so
#  that ml_classifier.LabelEncodedXGB is always imported as
#  "ml_classifier.LabelEncodedXGB" rather than pickled under
#  "__main__" -- otherwise app.py's later `import ml_classifier` can't
#  unpickle the saved model.
# ============================================================

from ml_classifier import main

if __name__ == "__main__":
    main()