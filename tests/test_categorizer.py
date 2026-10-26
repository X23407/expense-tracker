"""
test_categorizer.py - Unit tests for rule-based and ML categorization
"""

import pandas as pd
import pytest

from src.categorizer import ExpenseClassifier, categorize, categorize_by_rules, clean_text


def test_clean_text():
    """Verify text cleaning removes punctuation and normalizes spaces."""
    assert clean_text("UPI/SWIGGY/12345@YBL") == "upi swiggy 12345 ybl"
    assert clean_text("POS 4321 AMAZON RETAIL BLR") == "pos 4321 amazon retail blr"
    assert clean_text("  Domino's   Pizza  ") == "domino s pizza"


def test_keyword_rules_known_merchants():
    """Verify known merchants are correctly classified by keyword rules."""
    assert categorize_by_rules("Swiggy Bangalore") == "Food & Dining"
    assert categorize_by_rules("Zomato Delivery") == "Food & Dining"
    assert categorize_by_rules("Domino's Pizza Order") == "Food & Dining"
    assert categorize_by_rules("BigBasket Weekly") == "Groceries"
    assert categorize_by_rules("Zepto 10 min grocery") == "Groceries"
    assert categorize_by_rules("Uber Ride to Airport") == "Transport"
    assert categorize_by_rules("Namma Metro card recharge") == "Transport"
    assert categorize_by_rules("BESCOM Electricity Bill") == "Rent & Utilities"
    assert categorize_by_rules("Amazon Retail India") == "Shopping"
    assert categorize_by_rules("Apollo Pharmacy Medicines") == "Health"
    assert categorize_by_rules("Netflix India Subscription") == "Subscriptions"
    assert categorize_by_rules("MakeMyTrip Flight Booking") == "Travel"


def test_keyword_rules_case_insensitive():
    """Rules must be strictly case-insensitive."""
    assert categorize_by_rules("swiggy") == "Food & Dining"
    assert categorize_by_rules("SWIGGY") == "Food & Dining"
    assert categorize_by_rules("SwIgGy") == "Food & Dining"


def test_unknown_merchant_fallback():
    """Unknown or unmapped merchants should fall back to Miscellaneous."""
    cat, source = categorize("Some Unknown Arbitrary Vendor XYZ 99")
    assert cat == "Miscellaneous"
    assert source == "default"


def test_ml_classifier_training_and_prediction():
    """Verify TF-IDF + Logistic Regression pipeline trains and classifies novel inputs."""
    training_data = pd.DataFrame([
        {"merchant": "Swiggy Order 1", "category": "Food & Dining"},
        {"merchant": "Swiggy Order 2", "category": "Food & Dining"},
        {"merchant": "Zomato Meals", "category": "Food & Dining"},
        {"merchant": "Zomato Food Express", "category": "Food & Dining"},
        {"merchant": "Cafe Coffee Bistro", "category": "Food & Dining"},
        {"merchant": "Uber Trip Whitefield", "category": "Transport"},
        {"merchant": "Uber Trip Indiranagar", "category": "Transport"},
        {"merchant": "Ola Cab Ride", "category": "Transport"},
        {"merchant": "Ola Outstation", "category": "Transport"},
        {"merchant": "Rapido Bike Taxi", "category": "Transport"},
        {"merchant": "BigBasket Grocery Box", "category": "Groceries"},
        {"merchant": "BigBasket Fresh Veggies", "category": "Groceries"},
        {"merchant": "Zepto Quick Delivery", "category": "Groceries"},
        {"merchant": "Blinkit Instant Groceries", "category": "Groceries"},
        {"merchant": "DMart Supermarket Goods", "category": "Groceries"},
    ])

    classifier = ExpenseClassifier()
    success = classifier.train(training_data)
    assert success is True
    assert classifier.is_trained is True

    # Test prediction on unseen variations
    pred = classifier.predict("Zomato Late Night")
    assert pred == "Food & Dining"

    pred_uber = classifier.predict("Uber Ride Koramangala")
    assert pred_uber == "Transport"

