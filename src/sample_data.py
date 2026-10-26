"""
sample_data.py - Realistic Indian Financial Dataset Generator

Calibrated to mirror genuine Indian personal UPI / PhonePe spending patterns:
- Realistic student / young professional monthly spend: ~₹25,000 to ₹38,000/mo.
- Daily average outflow: ~₹300 to ₹700/day.
- High frequency of small micro-payments (₹10 - ₹90 for tea, canteen, snacks, autos).
- Modest grocery and meal orders (₹120 - ₹450).
- Fixed recurring monthly rent (₹10,000/mo) on the 1st ("Campus Hostel & Rent").
- Digital subscriptions: Spotify Premium (₹119) and YouTube Premium (₹149).
- Realistic anomalies: ₹3,000 - ₹5,000 for smartphone repair, exam fees, dental care.
- Messy statement export matching PhonePe CSV schema with Credit filtering and duplicate rows.
"""

from datetime import date, datetime, timedelta
from pathlib import Path
import random
import sys
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd

# Ensure repository root is on sys.path for direct script execution
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.db import bulk_insert_expenses, get_connection, init_db, set_budget, DEFAULT_BUDGETS

# Data directory paths
DATA_DIR = BASE_DIR / "data"
SAMPLE_EXPENSES_CSV = DATA_DIR / "sample_expenses.csv"
SAMPLE_BANK_CSV = DATA_DIR / "sample_bank_statement.csv"

# Realistic category distributions aligned with PhonePe user habits
CATEGORY_SPECS = {
    "Food & Dining": {
        "merchants": [
            "Chai Point", "Local Tea Stall", "The Green Cafe", "Campus Canteen",
            "Campus Tea Stall", "Local Juice & Snacks", "Swiggy", "Zomato", "City Food Corner", "Local Dhaba"
        ],
        "micro_amounts": [10.0, 15.0, 20.0, 25.0, 30.0, 40.0, 50.0, 60.0, 90.0],
        "regular_mean": 180,
        "regular_std": 80,
        "min_amount": 10,
        "max_amount": 450,
        "default_payment": ["UPI", "UPI", "UPI", "Cash"],
    },
    "Groceries": {
        "merchants": ["City Supermarket", "Zepto", "Blinkit", "Local Kirana Store", "Nature's Basket", "Vegetable Vendor"],
        "micro_amounts": [45.0, 65.0, 90.0, 120.0],
        "regular_mean": 240,
        "regular_std": 110,
        "min_amount": 35,
        "max_amount": 650,
        "default_payment": ["UPI", "UPI", "Debit Card"],
    },
    "Transport": {
        "merchants": ["Rapido Bike Taxi", "Namma Metro Recharge", "City Auto Rickshaw", "HP Fuel Station", "Uber Auto"],
        "micro_amounts": [20.0, 25.0, 40.0, 50.0, 70.0],
        "regular_mean": 90,
        "regular_std": 45,
        "min_amount": 20,
        "max_amount": 250,
        "default_payment": ["UPI", "UPI", "Cash"],
    },
    "Rent & Utilities": {
        "merchants": ["Jio Mobile Recharge", "Airtel Wi-Fi", "BESCOM Electricity", "Water Bill"],
        "micro_amounts": [99.0, 149.0, 239.0],
        "regular_mean": 280,
        "regular_std": 120,
        "min_amount": 80,
        "max_amount": 650,
        "default_payment": ["UPI", "Credit Card"],
    },
    "Entertainment": {
        "merchants": ["PVR Cinemas", "BookMyShow", "Steam Games", "Tea & Snacks with Friends", "Gaming Cafe"],
        "micro_amounts": [50.0, 80.0, 120.0],
        "regular_mean": 220,
        "regular_std": 90,
        "min_amount": 50,
        "max_amount": 500,
        "default_payment": ["UPI", "Credit Card"],
    },
    "Shopping": {
        "merchants": ["Amazon India", "Flipkart", "Myntra", "Local Market Goods", "Decathlon Sports"],
        "micro_amounts": [150.0, 220.0, 350.0],
        "regular_mean": 480,
        "regular_std": 220,
        "min_amount": 100,
        "max_amount": 1200,
        "default_payment": ["UPI", "Credit Card", "Debit Card"],
    },
    "Health": {
        "merchants": ["Apollo Pharmacy", "Local Chemist", "Tata 1mg", "Clinic Consultation", "Dr. Lal PathLabs"],
        "micro_amounts": [35.0, 60.0, 110.0],
        "regular_mean": 190,
        "regular_std": 90,
        "min_amount": 25,
        "max_amount": 550,
        "default_payment": ["UPI", "Cash"],
    },
    "Education": {
        "merchants": ["Stationery & Xerox", "College Bookstore", "Udemy Course", "Blossom Book House"],
        "micro_amounts": [20.0, 35.0, 60.0, 90.0],
        "regular_mean": 180,
        "regular_std": 95,
        "min_amount": 15,
        "max_amount": 600,
        "default_payment": ["UPI", "Credit Card"],
    },
    "Miscellaneous": {
        "merchants": ["Rohan Sharma (Transfer)", "Amit Verma (Shared Bill)", "Vikram Singh (Transfer)", "Campus Stationery", "Courier & Prints", "Local Hardware"],
        "micro_amounts": [10.0, 20.0, 50.0, 90.0, 200.0],
        "regular_mean": 320,
        "regular_std": 180,
        "min_amount": 10,
        "max_amount": 900,
        "default_payment": ["UPI", "UPI", "Cash"],
    },
}

# Fixed monthly subscriptions
RECURRING_SUBSCRIPTIONS = [
    {"merchant": "Spotify Premium", "category": "Subscriptions", "amount": 119.0, "day": 12, "payment_method": "UPI", "note": "Monthly student audio streaming subscription"},
    {"merchant": "YouTube Premium", "category": "Subscriptions", "amount": 149.0, "day": 18, "payment_method": "UPI", "note": "Monthly ad-free video streaming"},
]

# Realistic statistical anomalies scaled to student budget (~₹2,500 to ₹5,500)
INJECTED_ANOMALIES = [
    {"merchant": "Mobile Repair Care", "category": "Miscellaneous", "amount": 4800.0, "days_ago": 28, "payment_method": "UPI", "note": "Smartphone screen replacement and battery overhaul"},
    {"merchant": "Dr. Smile Dental Clinic", "category": "Health", "amount": 3500.0, "days_ago": 65, "payment_method": "UPI", "note": "Emergency root canal treatment and medication"},
    {"merchant": "University Exam Portal", "category": "Education", "amount": 5200.0, "days_ago": 88, "payment_method": "Credit Card", "note": "Semester examination and professional certification fees"},
    {"merchant": "Zara Retail India", "category": "Shopping", "amount": 4200.0, "days_ago": 110, "payment_method": "Credit Card", "note": "Winter jacket and formal interview outfit"},
    {"merchant": "Motorcycle Service Point", "category": "Transport", "amount": 2800.0, "days_ago": 135, "payment_method": "UPI", "note": "Two-wheeler full servicing, brake pads, and oil replacement"},
    {"merchant": "Grand Celebration Feast", "category": "Food & Dining", "amount": 3800.0, "days_ago": 160, "payment_method": "Credit Card", "note": "College graduation and project completion team party"},
]


def generate_dataset(seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate realistic Indian expenses dataset (~320 transactions) over 180 days.
    Mirrors genuine PhonePe daily patterns:
    - Monthly rent of ₹10,000 on the 1st
    - 65% micro-transactions between ₹10 and ₹90
    - Monthly total ~₹28,000 - ₹35,000
    """
    random.seed(seed)
    np.random.seed(seed)

    today = date.today()
    start_date = today - timedelta(days=180)

    records: List[Dict[str, Any]] = []

    # 1. Monthly Fixed Rent on the 1st of each month (₹10,000 to Campus Hostel & Rent)
    cur_month = date(start_date.year, start_date.month, 1)
    while cur_month <= today:
        if cur_month >= start_date:
            records.append({
                "date": cur_month.isoformat(),
                "amount": 10000.0,
                "category": "Rent & Utilities",
                "merchant": "Campus Hostel & Rent",
                "payment_method": "UPI",
                "note": f"Monthly accommodation rent for {cur_month.strftime('%B %Y')}",
            })
        if cur_month.month == 12:
            cur_month = date(cur_month.year + 1, 1, 1)
        else:
            cur_month = date(cur_month.year, cur_month.month + 1, 1)

    # 2. Monthly Fixed Recurring Subscriptions (Spotify & YouTube)
    for sub in RECURRING_SUBSCRIPTIONS:
        sub_month = date(start_date.year, start_date.month, 1)
        while sub_month <= today:
            try:
                sub_date = date(sub_month.year, sub_month.month, min(sub["day"], 28))
                if start_date <= sub_date <= today:
                    records.append({
                        "date": sub_date.isoformat(),
                        "amount": float(sub["amount"]),
                        "category": sub["category"],
                        "merchant": sub["merchant"],
                        "payment_method": sub["payment_method"],
                        "note": sub["note"],
                    })
            except ValueError:
                pass
            if sub_month.month == 12:
                sub_month = date(sub_month.year + 1, 1, 1)
            else:
                sub_month = date(sub_month.year, sub_month.month + 1, 1)

    # 3. Daily Transactions (~1.6 transactions per day, matching genuine PhonePe cadence)
    current_day = start_date
    while current_day <= today:
        day_of_week = current_day.weekday()
        is_weekend = day_of_week in [4, 5, 6]

        # 0 to 3 transactions per day
        lambda_val = 1.8 if is_weekend else 1.4
        num_txns = np.random.poisson(lam=lambda_val)

        categories = ["Food & Dining", "Groceries", "Transport", "Miscellaneous", "Shopping", "Health", "Education", "Entertainment"]
        weights = [0.42, 0.20, 0.16, 0.10, 0.04, 0.03, 0.03, 0.02]

        if num_txns > 0:
            chosen_cats = np.random.choice(categories, size=num_txns, p=weights)
            for cat in chosen_cats:
                spec = CATEGORY_SPECS[cat]
                merchant = random.choice(spec["merchants"])
                payment = random.choice(spec["default_payment"])

                # 65% probability of a realistic micro-amount (₹10 - ₹90)
                if random.random() < 0.65 and "micro_amounts" in spec:
                    amt = float(random.choice(spec["micro_amounts"]))
                else:
                    amt = float(np.random.normal(loc=spec["regular_mean"], scale=spec["regular_std"]))
                    amt = round(float(np.clip(amt, spec["min_amount"], spec["max_amount"])), 2)

                records.append({
                    "date": current_day.isoformat(),
                    "amount": amt,
                    "category": cat,
                    "merchant": merchant,
                    "payment_method": payment,
                    "note": f"Paid via {payment}",
                })

        current_day += timedelta(days=1)

    # 4. Inject 6 Realistic Statistical Anomalies
    for anom in INJECTED_ANOMALIES:
        target_date = today - timedelta(days=anom["days_ago"])
        records.append({
            "date": target_date.isoformat(),
            "amount": float(anom["amount"]),
            "category": anom["category"],
            "merchant": anom["merchant"],
            "payment_method": anom["payment_method"],
            "note": f"[ANOMALY INJECTED] {anom['note']}",
        })

    clean_df = pd.DataFrame(records)
    clean_df["date"] = pd.to_datetime(clean_df["date"]).dt.strftime("%Y-%m-%d")
    clean_df = clean_df.sort_values(by=["date", "amount"], ascending=[True, False]).reset_index(drop=True)

    # 5. Build PhonePe Statement CSV format for testing the importer
    # Schema matches user's PhonePe export:
    # Date,Time,Transaction Details,Transaction ID,UTR,Transaction Type,Credit/debit instrument,Amount
    phonepe_rows: List[Dict[str, Any]] = []

    # Sample ~120 recent transactions
    sample_for_statement = clean_df.tail(120).copy()

    for idx, row in sample_for_statement.iterrows():
        dt_obj = datetime.strptime(row["date"], "%Y-%m-%d")
        date_str = dt_obj.strftime("%b %d, %Y")
        time_str = f"{random.randint(8, 22):02d}:{random.randint(10, 59):02d} {'pm' if random.random() > 0.4 else 'am'}"

        merchant = row["merchant"]
        amt = float(row["amount"])
        txn_id = f"T{dt_obj.strftime('%y%m%d')}{random.randint(1000000000000, 9999999999999)}"
        utr = f"{random.randint(100000000000, 999999999999)}"

        phonepe_rows.append({
            "Date": date_str,
            "Time": time_str,
            "Transaction Details": f"Paid to {merchant.upper()}",
            "Transaction ID": txn_id,
            "UTR": utr,
            "Transaction Type": "DEBIT",
            "Credit/debit instrument": f"Paid by XXXXXX{random.randint(1000, 9999)}",
            "Amount": f"{int(amt) if amt.is_integer() else amt}",
        })

        # Inject realistic CREDIT deposits (refunds or incoming peer transfers)
        if idx % 18 == 0:
            cred_amt = random.choice([20.0, 50.0, 200.0, 2000.0])
            phonepe_rows.append({
                "Date": date_str,
                "Time": "08:16 am",
                "Transaction Details": f"Received from {random.choice(['COLLEGE SCHOLARSHIP', 'PEER REIMBURSEMENT', 'PROJECT STIPEND', 'UPI CASHBACK'])}",
                "Transaction ID": f"T{dt_obj.strftime('%y%m%d')}{random.randint(1000000000000, 9999999999999)}",
                "UTR": f"{random.randint(100000000000, 999999999999)}",
                "Transaction Type": "CREDIT",
                "Credit/debit instrument": "Credited to XXXXXX4127",
                "Amount": f"{int(cred_amt)}",
            })

    # Deliberately inject 8 exact duplicate rows to test duplicate rejection
    dups_to_inject = random.sample(phonepe_rows[:40], 8)
    phonepe_rows.extend(dups_to_inject)

    messy_bank_df = pd.DataFrame(phonepe_rows)

    return clean_df, messy_bank_df


def save_sample_datasets() -> Tuple[Path, Path]:
    """Generate and write calibrated sample CSV files to the data/ directory."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    clean_df, messy_bank_df = generate_dataset(seed=42)

    clean_df.to_csv(SAMPLE_EXPENSES_CSV, index=False)

    # Write messy bank statement with realistic PhonePe header metadata and disclaimer footer
    header_str = (
        'Transaction Statement for 9876543210\n'
        'Duration,"01 May, 2026 - 26 Oct, 2026"\n\n'
    )
    footer_str = (
        '\nThis is an automatically generated statement. Customer(s) are requested to immediately notify PhonePe in case\n'
        'of any errors in the statement at https://support.phonepe.com/statement and visit https://www.phonepe.com/\n'
        'terms-conditions/ for PhonePe Terms & Conditions and Privacy Policy.\n\n'
        'Disclaimer : Do not fall prey to fictitious offers of winning prizes, money circulation schemes and cheap funds,\n'
        'etc. through SMS, emails and calls. The contents of this email and document are confidential and intended for\n'
        'the recipient specified in this document. If you received this message by mistake, please inform PhonePe at\n'
        'https://support.phonepe.com/statement so that we can ensure the recipient\'s details are corrected.\n'
    )
    raw_csv = messy_bank_df.to_csv(index=False)
    SAMPLE_BANK_CSV.write_text(header_str + raw_csv + footer_str, encoding="utf-8")

    return SAMPLE_EXPENSES_CSV, SAMPLE_BANK_CSV


def load_sample_data_if_empty(db_path: Any = None) -> bool:
    """
    Check if the expenses database table is empty; if so, populate it with
    the calibrated realistic dataset.
    """
    init_db(db_path)
    conn = get_connection(db_path)
    try:
        cursor = conn.execute("SELECT COUNT(*) AS count FROM expenses;")
        count = cursor.fetchone()["count"]
        if count > 0:
            return False
    finally:
        conn.close()

    clean_df, _ = generate_dataset(seed=42)
    clean_df.to_csv(SAMPLE_EXPENSES_CSV, index=False)

    # Bulk insert into SQLite database
    bulk_insert_expenses(clean_df, db_path=db_path)

    # Ensure updated budgets are recorded
    for cat, limit in DEFAULT_BUDGETS.items():
        set_budget(cat, limit, db_path=db_path)

    return True


if __name__ == "__main__":
    clean_path, bank_path = save_sample_datasets()
    print("Calibrated realistic sample data generated successfully:")
    print(f"  Clean expenses: {clean_path}")
    print(f"  PhonePe bank statement: {bank_path}")
