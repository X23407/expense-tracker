"""
test_forecast.py - Unit tests for manual 3-Month Weighted Moving Average (WMA)
"""

import pandas as pd
import pytest

from src.insights import calculate_wma_forecast, compare_forecast_to_budget


def test_manual_wma_exact_calculation():
    """
    Test exact weighted moving average on controlled numbers:
    Month 1: 10,000
    Month 2: 20,000
    Month 3: 30,000
    Weights: [0.2, 0.3, 0.5]
    Expected: (0.2 * 10,000) + (0.3 * 20,000) + (0.5 * 30,000)
            = 2,000 + 6,000 + 15,000 = 23,000.
    """
    df = pd.DataFrame([
        {"date": "2026-01-15", "amount": 10000.0, "category": "General", "merchant": "Shop A"},
        {"date": "2026-02-15", "amount": 20000.0, "category": "General", "merchant": "Shop B"},
        {"date": "2026-03-15", "amount": 30000.0, "category": "General", "merchant": "Shop C"},
    ])

    res = calculate_wma_forecast(df, weights=(0.2, 0.3, 0.5))

    assert res["forecast_month"] == "2026-04"
    assert res["historical_months"] == ["2026-01", "2026-02", "2026-03"]
    assert res["total_forecast"] == 23000.0


def test_wma_per_category_forecast():
    """Verify category-level forecasts match WMA formula."""
    df = pd.DataFrame([
        # Month 1
        {"date": "2026-01-10", "amount": 1000.0, "category": "Food & Dining", "merchant": "Swiggy"},
        {"date": "2026-01-12", "amount": 5000.0, "category": "Rent & Utilities", "merchant": "Rent"},
        # Month 2
        {"date": "2026-02-10", "amount": 2000.0, "category": "Food & Dining", "merchant": "Swiggy"},
        {"date": "2026-02-12", "amount": 5000.0, "category": "Rent & Utilities", "merchant": "Rent"},
        # Month 3
        {"date": "2026-03-10", "amount": 3000.0, "category": "Food & Dining", "merchant": "Swiggy"},
        {"date": "2026-03-12", "amount": 5000.0, "category": "Rent & Utilities", "merchant": "Rent"},
    ])

    res = calculate_wma_forecast(df, weights=(0.2, 0.3, 0.5))
    cat_df = res["category_forecasts"]

    # Food: 0.2*1000 + 0.3*2000 + 0.5*3000 = 200 + 600 + 1500 = 2300
    food_row = cat_df[cat_df["category"] == "Food & Dining"].iloc[0]
    assert food_row["forecast_amount"] == 2300.0

    # Rent: 0.2*5000 + 0.3*5000 + 0.5*5000 = 5000
    rent_row = cat_df[cat_df["category"] == "Rent & Utilities"].iloc[0]
    assert rent_row["forecast_amount"] == 5000.0


def test_compare_forecast_to_budget():
    """Verify budget comparison flags anticipated over-budget categories."""
    fc_df = pd.DataFrame([
        {"category": "Food & Dining", "forecast_amount": 12000.0},
        {"category": "Transport", "forecast_amount": 3000.0},
    ])
    budgets_df = pd.DataFrame([
        {"category": "Food & Dining", "monthly_limit": 10000.0},
        {"category": "Transport", "monthly_limit": 5000.0},
    ])

    comp = compare_forecast_to_budget(fc_df, budgets_df)
    assert len(comp) == 2

    food = comp[comp["category"] == "Food & Dining"].iloc[0]
    assert bool(food["warning"]) is True
    assert "Expected Over Budget" in food["status"]

    trans = comp[comp["category"] == "Transport"].iloc[0]
    assert bool(trans["warning"]) is False
    assert "Within Budget" in trans["status"]
