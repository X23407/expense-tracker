"""
insights.py - Manual Anomaly Detection and 3-Month Weighted Moving Average Forecast

Contains transparent, manually implemented algorithms for:
1. Z-Score Anomaly Detection: Identifies category-level spending outliers
   with manual mean, variance, and standard deviation calculations.
2. Weighted Moving Average (WMA) Forecast: Predicts next month's total
   and category-level expenses using fixed weights [0.2, 0.3, 0.5].
"""

from datetime import datetime
import math
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

# Ensure root in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.analytics import format_inr


# ============================================================================
# 1. MANUAL Z-SCORE ANOMALY DETECTION
# ============================================================================

def detect_anomalies(
    df: pd.DataFrame,
    threshold: float = 2.5,
    min_category_samples: int = 5,
) -> pd.DataFrame:
    """
    Manually calculate Z-scores for expenses grouped by category and flag outliers.

    Algorithm Logic:
    1. For each category, compute:
       - Sample Mean: mu = sum(x) / N
       - Sample Variance: s^2 = sum((x - mu)^2) / (N - 1)  [Bessel's correction]
       - Sample Standard Deviation: sigma = sqrt(s^2)
    2. For each transaction amount x, compute its standardized Z-score:
       - z = (x - mu) / sigma
    3. An expense is flagged if z >= threshold (default 2.5).
       A Z-score of 2.5 means the transaction is 2.5 standard deviations above
       the user's typical expenditure in that specific category (top ~1% of values).
    4. Construct human-readable reason string:
       "₹8,400 is 3.1σ above your average Shopping expense of ₹1,200"

    Design Notes:
    - Pure NumPy / pandas implementation without heavy external dependencies.
    - Handles small categories gracefully by falling back to overall dataset distribution.
    """
    if df.empty:
        return pd.DataFrame(
            columns=[
                "id", "date", "merchant", "category", "amount",
                "payment_method", "z_score", "category_mean", "category_std", "reason"
            ]
        )

    # Calculate overall dataset statistics as fallback for small categories
    all_amounts = df["amount"].tolist()
    n_all = len(all_amounts)
    mean_all = sum(all_amounts) / n_all if n_all > 0 else 0.0

    if n_all > 1:
        variance_all = sum((x - mean_all) ** 2 for x in all_amounts) / (n_all - 1)
        std_all = math.sqrt(variance_all)
    else:
        std_all = 1.0

    flagged_rows: List[Dict[str, Any]] = []

    # Group transactions by category to calculate category-specific baselines
    categories = df["category"].unique()

    for cat in categories:
        cat_df = df[df["category"] == cat].copy()
        cat_amounts = cat_df["amount"].tolist()
        n_cat = len(cat_amounts)

        # If category has sufficient samples, use category-specific parameters
        if n_cat >= min_category_samples:
            cat_mean = sum(cat_amounts) / n_cat
            # Manual sample variance with Bessel's correction (N - 1)
            cat_variance = sum((x - cat_mean) ** 2 for x in cat_amounts) / (n_cat - 1)
            cat_std = math.sqrt(cat_variance)
        else:
            # Fallback to global statistics if category has too few samples
            cat_mean = mean_all
            cat_std = std_all

        # Prevent division by zero if standard deviation is zero (all amounts identical)
        if cat_std < 1e-4:
            continue

        for _, row in cat_df.iterrows():
            amt = float(row["amount"])
            # Standardized Z-Score: distance from mean divided by standard deviation
            z = (amt - cat_mean) / cat_std

            # Outlier check: only flag unusually HIGH expenses (positive Z-score)
            if z >= threshold:
                sigma_str = f"{z:.1f}σ"
                reason = (
                    f"{format_inr(amt)} is {sigma_str} above your "
                    f"average {cat} expense of {format_inr(cat_mean)}"
                )

                flagged_rows.append({
                    "id": row.get("id", ""),
                    "date": row["date"],
                    "merchant": row["merchant"],
                    "category": cat,
                    "amount": amt,
                    "payment_method": row.get("payment_method", "UPI"),
                    "z_score": round(z, 2),
                    "category_mean": round(cat_mean, 2),
                    "category_std": round(cat_std, 2),
                    "reason": reason,
                })

    if not flagged_rows:
        return pd.DataFrame(
            columns=[
                "id", "date", "merchant", "category", "amount",
                "payment_method", "z_score", "category_mean", "category_std", "reason"
            ]
        )

    result_df = pd.DataFrame(flagged_rows)
    return result_df.sort_values(by="z_score", ascending=False).reset_index(drop=True)


# ============================================================================
# 2. MANUAL WEIGHTED MOVING AVERAGE FORECAST
# ============================================================================

def compute_next_month_str(current_month_str: str) -> str:
    """Given 'YYYY-MM', returns the next chronological month string 'YYYY-MM'."""
    year, month = map(int, current_month_str.split("-"))
    if month == 12:
        return f"{year + 1}-01"
    else:
        return f"{year}-{month + 1:02d}"


def calculate_wma_forecast(
    df: pd.DataFrame,
    weights: Tuple[float, float, float] = (0.2, 0.3, 0.5),
) -> Dict[str, Any]:
    """
    Manually calculate next month's spending forecast using a 3-month Weighted Moving Average (WMA).

    Mathematical Formulation:
      Given spending in the last 3 consecutive months [M_{-3}, M_{-2}, M_{-1}],
      Predicted Spending M_{next} = (w1 * M_{-3}) + (w2 * M_{-2}) + (w3 * M_{-1})
      where weights = [0.2, 0.3, 0.5] and sum(weights) = 1.0.

    Weighting Scheme Rationale:
    1. Consumer spending exhibits recency bias: lifestyle habits, recent inflation,
       and ongoing commitments are most heavily reflected in the most recent month
       (w3 = 50%).
    2. Intermediate month provides stability against one-off spikes (w2 = 30%).
    3. Three months ago provides a baseline anchor (w1 = 20%).
    4. Compared to simple moving average, WMA reacts faster to real lifestyle changes
       without suffering the severe extrapolation volatility of high-degree polynomial regressions.

    Returns:
        Dictionary with:
          - 'forecast_month': str ('YYYY-MM')
          - 'historical_months': List[str] of the 3 anchor months used
          - 'historical_amounts': List[float] of spends in those 3 months
          - 'weights': Tuple[float, float, float]
          - 'total_forecast': float (predicted total INR spend)
          - 'category_forecasts': pd.DataFrame (category, predicted_spend, last_month_spend, trend)
    """
    if df.empty:
        return {
            "forecast_month": "N/A",
            "historical_months": [],
            "historical_amounts": [],
            "weights": weights,
            "total_forecast": 0.0,
            "category_forecasts": pd.DataFrame(),
        }

    temp = df.copy()
    temp["month"] = pd.to_datetime(temp["date"]).dt.strftime("%Y-%m")

    # Group by month to get total monthly spend
    monthly_series = temp.groupby("month")["amount"].sum().sort_index()

    if len(monthly_series) == 0:
        return {
            "forecast_month": "N/A",
            "historical_months": [],
            "historical_amounts": [],
            "weights": weights,
            "total_forecast": 0.0,
            "category_forecasts": pd.DataFrame(),
        }

    # Normalize weights so they strictly sum to 1.0
    w_sum = sum(weights)
    norm_w = [w / w_sum for w in weights]

    # Handle datasets with fewer than 3 months of history
    all_months = list(monthly_series.index)
    if len(all_months) < 3:
        # Fallback to simple average of available months
        anchor_months = all_months
        anchor_amounts = [float(monthly_series[m]) for m in anchor_months]
        total_pred = float(np.mean(anchor_amounts))
        next_month = compute_next_month_str(all_months[-1])
        used_weights = [1.0 / len(anchor_months)] * len(anchor_months)
    else:
        # Use the last 3 chronological months
        anchor_months = all_months[-3:]
        anchor_amounts = [float(monthly_series[m]) for m in anchor_months]
        # WMA formula: w1*M1 + w2*M2 + w3*M3
        total_pred = (
            norm_w[0] * anchor_amounts[0]
            + norm_w[1] * anchor_amounts[1]
            + norm_w[2] * anchor_amounts[2]
        )
        next_month = compute_next_month_str(anchor_months[-1])
        used_weights = norm_w

    # Compute Per-Category WMA Forecasts
    cat_pivot = (
        temp.pivot_table(index="category", columns="month", values="amount", aggfunc="sum")
        .fillna(0.0)
    )

    cat_rows: List[Dict[str, Any]] = []
    for cat in cat_pivot.index:
        cat_series = cat_pivot.loc[cat]
        if len(anchor_months) == 3:
            m1, m2, m3 = anchor_months
            val1 = float(cat_series.get(m1, 0.0))
            val2 = float(cat_series.get(m2, 0.0))
            val3 = float(cat_series.get(m3, 0.0))
            pred = (norm_w[0] * val1) + (norm_w[1] * val2) + (norm_w[2] * val3)
            last_actual = val3
        else:
            vals = [float(cat_series.get(m, 0.0)) for m in anchor_months]
            pred = float(np.mean(vals)) if vals else 0.0
            last_actual = vals[-1] if vals else 0.0

        pct_shift = (
            ((pred - last_actual) / last_actual * 100.0)
            if last_actual > 0
            else 0.0
        )

        cat_rows.append({
            "category": cat,
            "forecast_amount": round(pred, 2),
            "last_month_amount": round(last_actual, 2),
            "change_pct": round(pct_shift, 1),
        })

    cat_forecast_df = pd.DataFrame(cat_rows).sort_values(
        by="forecast_amount", ascending=False
    ).reset_index(drop=True)

    return {
        "forecast_month": next_month,
        "historical_months": anchor_months,
        "historical_amounts": anchor_amounts,
        "weights": used_weights,
        "total_forecast": round(total_pred, 2),
        "category_forecasts": cat_forecast_df,
    }


def compare_forecast_to_budget(
    forecast_df: pd.DataFrame,
    budgets_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compare forecasted category expenditure against monthly budget limits to flag
    categories anticipated to exceed budget next month.
    """
    if forecast_df.empty or budgets_df.empty:
        return pd.DataFrame()

    merged = pd.merge(forecast_df, budgets_df, on="category", how="left")
    merged["monthly_limit"] = merged["monthly_limit"].fillna(0.0)
    merged["projected_variance"] = merged["forecast_amount"] - merged["monthly_limit"]
    merged["projected_pct"] = (
        (merged["forecast_amount"] / merged["monthly_limit"].replace(0, np.nan)) * 100.0
    ).fillna(0.0).round(1)

    merged["warning"] = merged["forecast_amount"] > merged["monthly_limit"]
    merged["status"] = merged["warning"].apply(
        lambda x: "⚠️ Expected Over Budget" if x else "✅ Within Budget"
    )

    return merged.sort_values(by="projected_variance", ascending=False).reset_index(drop=True)


def backtest_forecast(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Backtesting evaluation for forecasting accuracy:
    Hides the most recent month, trains WMA on the 3 preceding months,
    predicts the hidden month, and computes the Mean Absolute Error (MAE) and MAPE.
    """
    temp = df.copy()
    temp["month"] = pd.to_datetime(temp["date"]).dt.strftime("%Y-%m")
    monthly = temp.groupby("month")["amount"].sum().sort_index()

    months = list(monthly.index)
    if len(months) < 4:
        return {"error": "Need at least 4 months of historical data for backtesting."}

    # Hide the latest month and use preceding 3 months as anchors
    actual_test_month = months[-1]
    actual_value = float(monthly[actual_test_month])

    train_anchors = months[-4:-1]
    m1, m2, m3 = [float(monthly[m]) for m in train_anchors]
    predicted_value = (0.2 * m1) + (0.3 * m2) + (0.5 * m3)

    abs_error = abs(predicted_value - actual_value)
    mape = (abs_error / actual_value) * 100.0 if actual_value > 0 else 0.0

    return {
        "test_month": actual_test_month,
        "anchor_months": train_anchors,
        "actual_amount": round(actual_value, 2),
        "predicted_amount": round(predicted_value, 2),
        "abs_error": round(abs_error, 2),
        "mape_pct": round(mape, 2),
    }

