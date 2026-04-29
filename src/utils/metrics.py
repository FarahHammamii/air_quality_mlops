"""
src/utils/metrics.py
====================
Evaluation metrics for air quality prediction.
Matches the metrics used in the HEART paper.

FIXES vs v1
-----------
1. MAPE: guarded against near-zero y_true values (PM2.5 can be 0–5 µg/m³
   in clean-air periods) — clamp denominator to max(|y|, 1.0) so we never
   divide by near-zero, which previously inflated MAPE to 1000%+.
2. R²: uses sklearn.metrics.r2_score which is numerically stable and
   matches the formula the paper references (1 - SS_res/SS_tot).
   When the model just predicts the mean, R² = 0.  When it's perfect, R² = 1.
   Negative R² means the model is worse than predicting the mean — a clear
   signal that target scaling is needed (handled in window_builder.py).
3. Within-25%: same safe denominator as MAPE.
"""

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Calculate all relevant metrics.  All inputs must already be in raw µg/m³
    (i.e. inverse-transformed if the model was trained on scaled targets).

    Parameters
    ----------
    y_true : array of actual PM2.5 values (µg/m³)
    y_pred : array of predicted PM2.5 values (µg/m³)

    Returns
    -------
    dict of metric name → float
    """
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()

    mse  = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    mae  = float(mean_absolute_error(y_true, y_pred))
    r2   = float(r2_score(y_true, y_pred))

    # Safe MAPE: clamp denominator to avoid div-by-zero on near-zero readings
    # Using max(|y_true|, 1.0) µg/m³ as a floor — 1 µg/m³ is essentially zero
    # for air quality purposes but prevents numerical explosion.
    denom = np.maximum(np.abs(y_true), 1.0)
    mape  = float(np.mean(np.abs((y_true - y_pred) / denom)) * 100)

    # Fraction of predictions within 25% of true value (same safe denominator)
    within_25pct = float(
        np.mean(np.abs(y_true - y_pred) / denom <= 0.25) * 100
    )

    # Fraction within 50 µg/m³ absolute error (WHO/EPA threshold context)
    within_50 = float(np.mean(np.abs(y_true - y_pred) <= 50) * 100)

    # Systematic bias: positive = over-prediction, negative = under-prediction
    bias = float(np.mean(y_pred - y_true))

    # Willmott Index of Agreement (0 = no agreement, 1 = perfect)
    numerator   = float(np.sum((y_pred - y_true) ** 2))
    denominator = float(np.sum(
        (np.abs(y_pred - np.mean(y_true)) + np.abs(y_true - np.mean(y_true))) ** 2
    ))
    index_of_agreement = 1.0 - (numerator / (denominator + 1e-8))

    return {
        "MSE":                  mse,
        "RMSE":                 rmse,
        "MAE":                  mae,
        "R²":                   r2,
        "MAPE (%)":             mape,
        "Within 25% (%)":       within_25pct,
        "Within 50 µg/m³ (%)":  within_50,
        "Bias":                 bias,
        "Index of Agreement":   index_of_agreement,
    }


def compare_models(baseline_pred: np.ndarray,
                   heart_pred:    np.ndarray,
                   y_true:        np.ndarray) -> dict:
    """
    Compare baseline and HEART models.

    All arrays must be in raw µg/m³ (inverse-transformed).

    Returns
    -------
    dict with keys:
      'baseline'          : metric dict for baseline
      'heart'             : metric dict for HEART
      'improvements'      : % improvement HEART vs baseline (positive = better)
      'meets_paper_claim' : bool, True if MSE improvement ≥ 7.5%
    """
    baseline_metrics = calculate_metrics(y_true, baseline_pred)
    heart_metrics    = calculate_metrics(y_true, heart_pred)

    mse_improvement  = (baseline_metrics["MSE"]  - heart_metrics["MSE"])  / (baseline_metrics["MSE"]  + 1e-8) * 100
    mae_improvement  = (baseline_metrics["MAE"]  - heart_metrics["MAE"])  / (baseline_metrics["MAE"]  + 1e-8) * 100
    rmse_improvement = (baseline_metrics["RMSE"] - heart_metrics["RMSE"]) / (baseline_metrics["RMSE"] + 1e-8) * 100

    return {
        "baseline":    baseline_metrics,
        "heart":       heart_metrics,
        "improvements": {
            "MSE (%)":  mse_improvement,
            "MAE (%)":  mae_improvement,
            "RMSE (%)": rmse_improvement,
        },
        "meets_paper_claim": mse_improvement >= 7.5,
    }


def print_comparison(comparison: dict) -> None:
    """Pretty-print model comparison results."""
    print("\n" + "=" * 70)
    print("MODEL COMPARISON RESULTS")
    print("=" * 70)

    print("\n📊 Baseline Model:")
    for metric, value in comparison["baseline"].items():
        print(f"   {metric:25s}: {value:.4f}")

    print("\n🚀 HEART Model (with Attention):")
    for metric, value in comparison["heart"].items():
        print(f"   {metric:25s}: {value:.4f}")

    print("\n📈 Improvements (HEART vs Baseline):")
    for metric, value in comparison["improvements"].items():
        sign = "✅" if value > 0 else "❌"
        print(f"   {sign} {metric:20s}: {value:+.2f}%")

    print("\n" + "=" * 70)
    if comparison["meets_paper_claim"]:
        print("✅ SUCCESS: Achieved HEART paper's improvement target (≥7.5% MSE reduction)")
    else:
        print("⚠️  NOTE: Below paper's claimed improvement. "
              "Consider more data or a longer rolling window.")
    print("=" * 70)


if __name__ == "__main__":
    np.random.seed(42)
    y_true        = np.abs(np.random.randn(200) * 40 + 80)  # realistic PM2.5
    baseline_pred = y_true + np.random.randn(200) * 15
    heart_pred    = y_true + np.random.randn(200) * 12

    comparison = compare_models(baseline_pred, heart_pred, y_true)
    print_comparison(comparison)
