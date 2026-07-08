# ============================================================
#  ml_classifier.py
#
#  Replaces the if/else threshold in dcf_engine.classify_valuation()
#  with an actual trained, evaluated ML classifier -- the "where's
#  the ML" fix.
#
#  Training data = evaluated_calls.csv (or evaluated_calls_demo.csv),
#  produced by backtest_engine.py: features captured at call time,
#  labeled with the REALIZED forward-return outcome (not analyst
#  consensus). So the two feedback items are wired together on
#  purpose -- the backtest isn't just a report, it's the training set.
#
#  Models compared:
#    - Logistic Regression  (baseline, interpretable coefficients)
#    - XGBoost               (gradient-boosted trees; falls back to
#                             sklearn's GradientBoostingClassifier if
#                             xgboost isn't installed)
#
#  Evaluated with stratified k-fold cross-validation (small-n safe)
#  plus a held-out test split for a final confusion matrix.
# ============================================================

from __future__ import annotations

import argparse
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_validate
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, ClassifierMixin

from feature_engineering import FEATURE_COLUMNS

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    from sklearn.ensemble import GradientBoostingClassifier
    HAS_XGBOOST = False


class LabelEncodedXGB(BaseEstimator, ClassifierMixin):
    """
    Thin wrapper so XGBClassifier (which requires numeric 0..K-1 class
    labels) plugs into the same sklearn-style fit/predict/predict_proba
    API as everything else here, using string labels like
    'UNDERVALUED' directly.
    """

    def __init__(self, **xgb_kwargs):
        self.xgb_kwargs = xgb_kwargs
        self._encoder = LabelEncoder()
        self._model = XGBClassifier(**xgb_kwargs)

    def fit(self, X, y):
        y_enc = self._encoder.fit_transform(y)
        self._model.fit(X, y_enc)
        self.classes_ = self._encoder.classes_
        return self

    def predict(self, X):
        preds = self._model.predict(X)
        return self._encoder.inverse_transform(preds)

    def predict_proba(self, X):
        return self._model.predict_proba(X)

    @property
    def feature_importances_(self):
        return self._model.feature_importances_

LABELS = ["UNDERVALUED", "FAIRLY VALUED", "OVERVALUED"]
MODEL_PATH = "valuation_classifier.joblib"
METRICS_PATH = "ml_classifier_metrics.json"

MIN_ROWS_FOR_TRAINING = 40  # below this, cross-val folds get unstable


def _build_xgb():
    if HAS_XGBOOST:
        return LabelEncodedXGB(
            n_estimators=200,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="mlogloss",
            random_state=42,
        )
    print("[note] xgboost not installed -- falling back to "
          "sklearn.GradientBoostingClassifier (same idea, slower, "
          "`pip install xgboost` for the real thing).")
    return GradientBoostingClassifier(
        n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42
    )


def load_training_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Training file is missing feature columns: {missing}")
    df = df.dropna(subset=FEATURE_COLUMNS + ["realized_label"])
    return df


def cross_validate_models(df: pd.DataFrame, n_splits: int = 5) -> dict:
    X = df[FEATURE_COLUMNS].astype("float64").to_numpy()
    y = df["realized_label"].astype(str).to_numpy()

    n_splits = min(n_splits, df["realized_label"].value_counts().min())
    n_splits = max(n_splits, 2)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    models = {
        "logistic_regression": Pipeline([
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000)),
        ]),
        "xgboost" if HAS_XGBOOST else "gradient_boosting": _build_xgb(),
    }

    results = {}
    for name, model in models.items():
        scores = cross_validate(
            model, X, y, cv=cv,
            scoring=["accuracy", "f1_macro"],
            return_train_score=False,
        )
        results[name] = {
            "cv_accuracy_mean": float(np.mean(scores["test_accuracy"])),
            "cv_accuracy_std": float(np.std(scores["test_accuracy"])),
            "cv_f1_macro_mean": float(np.mean(scores["test_f1_macro"])),
            "cv_f1_macro_std": float(np.std(scores["test_f1_macro"])),
            "n_splits": n_splits,
        }
    return results


def train_and_evaluate(df: pd.DataFrame, test_size: float = 0.25) -> dict:
    X = df[FEATURE_COLUMNS].astype("float64").to_numpy()
    y = df["realized_label"].astype(str).to_numpy()

    stratify = y if min(np.unique(y, return_counts=True)[1]) >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=stratify
    )

    report = {}
    fitted_models = {}

    for name, model in [
        ("logistic_regression", Pipeline([
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000)),
        ])),
        ("xgboost" if HAS_XGBOOST else "gradient_boosting", _build_xgb()),
    ]:
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        cm = confusion_matrix(y_test, preds, labels=LABELS)
        cm_df = pd.DataFrame(cm, index=[f"true_{l}" for l in LABELS],
                              columns=[f"pred_{l}" for l in LABELS])
        cls_report = classification_report(y_test, preds, labels=LABELS,
                                            zero_division=0, output_dict=True)

        report[name] = {
            "test_f1_macro": float(f1_score(y_test, preds, average="macro", zero_division=0)),
            "confusion_matrix": cm_df.to_dict(),
            "classification_report": cls_report,
        }
        fitted_models[name] = model

        print(f"\n--- {name} (held-out test set, n={len(y_test)}) ---")
        print(cm_df.to_string())
        print(classification_report(y_test, preds, labels=LABELS, zero_division=0))

    return report, fitted_models


def feature_importance_table(model, model_name: str) -> pd.DataFrame:
    """Best-effort feature importance/coefficients for whichever model won."""
    try:
        if model_name == "logistic_regression":
            clf = model.named_steps["clf"]
            # multinomial: one coef row per class -> take mean abs across classes
            importances = np.abs(clf.coef_).mean(axis=0)
        else:
            importances = model.feature_importances_
        return pd.DataFrame({
            "feature": FEATURE_COLUMNS,
            "importance": importances,
        }).sort_values("importance", ascending=False).reset_index(drop=True)
    except Exception as exc:
        print(f"Could not extract feature importances: {exc}")
        return pd.DataFrame()


def train_final_model_on_all_data(df: pd.DataFrame, best_model_name: str):
    """Refits the winning model on 100% of available data for deployment."""
    X, y = df[FEATURE_COLUMNS], df["realized_label"]
    if best_model_name == "logistic_regression":
        model = Pipeline([
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000)),
        ])
    else:
        model = _build_xgb()
    model.fit(X, y)
    joblib.dump({"model": model, "model_name": best_model_name,
                 "feature_columns": FEATURE_COLUMNS}, MODEL_PATH)
    print(f"\nSaved deployable model ({best_model_name}) -> {MODEL_PATH}")
    return model


def predict_verdict(feature_row: dict) -> dict | None:
    """
    Loads the saved model and predicts a verdict + class probabilities
    for a single feature row (as produced by
    feature_engineering.compute_features_for_ticker). Used by app.py.
    Returns None if no trained model file exists yet.
    """
    import os
    if not os.path.exists(MODEL_PATH):
        return None
    bundle = joblib.load(MODEL_PATH)
    model = bundle["model"]
    cols = bundle["feature_columns"]

    X = pd.DataFrame([{c: feature_row[c] for c in cols}])
    pred = model.predict(X)[0]
    proba = model.predict_proba(X)[0]
    classes = model.classes_ if hasattr(model, "classes_") else model.named_steps["clf"].classes_
    proba_map = {cls: float(p) for cls, p in zip(classes, proba)}
    return {"verdict": pred, "probabilities": proba_map, "model_name": bundle["model_name"]}


def main():
    parser = argparse.ArgumentParser(description="Train/evaluate the ML valuation classifier")
    parser.add_argument("--data", default="evaluated_calls.csv",
                         help="Path to evaluated calls CSV from backtest_engine.py "
                              "(use evaluated_calls_demo.csv for the historical demo set)")
    args = parser.parse_args()

    df = load_training_data(args.data)
    print(f"Loaded {len(df)} labeled rows from {args.data}")
    print(df["realized_label"].value_counts())

    if len(df) < MIN_ROWS_FOR_TRAINING:
        print(f"\n[warning] Only {len(df)} labeled rows -- below the "
              f"{MIN_ROWS_FOR_TRAINING}-row floor for stable cross-validation. "
              f"Metrics below will be noisy; grow the universe / wait for more "
              f"walk-forward calls to mature before citing these numbers.")

    print("\n=== Cross-validated comparison (5-fold, stratified) ===")
    cv_results = cross_validate_models(df)
    print(json.dumps(cv_results, indent=2))

    print("\n=== Held-out test set evaluation ===")
    test_report, fitted_models = train_and_evaluate(df)

    best_name = max(test_report, key=lambda k: test_report[k]["test_f1_macro"])
    print(f"\nBest model on held-out F1-macro: {best_name}")

    importances = feature_importance_table(fitted_models[best_name], best_name)
    print("\nFeature importances:")
    print(importances.to_string(index=False))

    final_model = train_final_model_on_all_data(df, best_name)

    with open(METRICS_PATH, "w") as f:
        json.dump({
            "n_rows": len(df),
            "cross_validation": cv_results,
            "held_out_test": {k: {
                "test_f1_macro": v["test_f1_macro"],
                "classification_report": v["classification_report"],
            } for k, v in test_report.items()},
            "best_model": best_name,
            "feature_importances": importances.to_dict(orient="records"),
        }, f, indent=2)
    print(f"\nSaved full metrics -> {METRICS_PATH}")

# NOTE: intentionally no `if __name__ == "__main__": main()` here.
# Run training via `python train_classifier.py` instead (see that file).
# Reason: if this module is ever executed directly, LabelEncodedXGB gets
# pickled with __module__ == "__main__", and later `import ml_classifier`
# (e.g. from app.py) can't find that class to unpickle the saved model.
# Keeping the CLI entry point in a separate file avoids that trap.
