"""
test_anomaly.py - Unit tests for manual Z-Score anomaly detection
"""

import math
import pandas as pd
import pytest

from src.insights import detect_anomalies


def test_manual_z_score_calculation():
    """
    Test manual z-score calculation on controlled hand-crafted numbers.
    Consider a category with 5 transactions:
    Values: [100, 100, 100, 100, 500]
    Mean = (100*4 + 500) / 5 = 180
    Variance = (4 * (100 - 180)^2 + (500 - 180)^2) / (5 - 1)
             = (4 * 6400 + 102400) / 4 = (25600 + 102400) / 4 = 128000 / 4 = 32000
    Std = sqrt(32000) ~= 178.885
    For 500: z = (500 - 180) / 178.885 = 320 / 178.885 ~= 1.7888
    """
    df = pd.DataFrame([
        {"id": 1, "date": "2026-05-01", "merchant": "Cafe A", "category": "Food & Dining", "amount": 100.0, "payment_method": "UPI"},
        {"id": 2, "date": "2026-05-02", "merchant": "Cafe B", "category": "Food & Dining", "amount": 100.0, "payment_method": "UPI"},
        {"id": 3, "date": "2026-05-03", "merchant": "Cafe C", "category": "Food & Dining", "amount": 100.0, "payment_method": "UPI"},
        {"id": 4, "date": "2026-05-04", "merchant": "Cafe D", "category": "Food & Dining", "amount": 100.0, "payment_method": "UPI"},
        {"id": 5, "date": "2026-05-05", "merchant": "Luxury Dining", "category": "Food & Dining", "amount": 500.0, "payment_method": "Credit Card"},
    ])

    # With threshold 1.5, 500 (z ~= 1.79) must be flagged
    anomalies_1_5 = detect_anomalies(df, threshold=1.5, min_category_samples=5)
    assert len(anomalies_1_5) == 1
    assert anomalies_1_5.iloc[0]["amount"] == 500.0
    assert abs(anomalies_1_5.iloc[0]["z_score"] - 1.79) < 0.05
    assert "Luxury Dining" in anomalies_1_5.iloc[0]["merchant"]

    # With threshold 2.0, 500 (z ~= 1.79) must NOT be flagged
    anomalies_2_0 = detect_anomalies(df, threshold=2.0, min_category_samples=5)
    assert len(anomalies_2_0) == 0


def test_anomaly_reason_string():
    """Verify reason string format matches requirement ('₹X is Y.Zσ above your average...')."""
    # Create 10 normal shopping items around 1000 and 1 massive outlier 15000
    rows = [{"id": i, "date": "2026-05-01", "merchant": "Store", "category": "Shopping", "amount": 1000.0, "payment_method": "UPI"} for i in range(10)]
    rows.append({"id": 99, "date": "2026-05-15", "merchant": "Croma Electronics", "category": "Shopping", "amount": 15000.0, "payment_method": "Credit Card"})
    df = pd.DataFrame(rows)

    anomalies = detect_anomalies(df, threshold=2.5)
    assert len(anomalies) == 1
    flagged = anomalies.iloc[0]
    assert flagged["amount"] == 15000.0
    assert "σ above your average Shopping expense" in flagged["reason"]


def test_empty_dataframe_anomaly():
    """Empty dataframe returns empty anomalies dataframe without crashing."""
    empty_df = pd.DataFrame()
    res = detect_anomalies(empty_df)
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 0

