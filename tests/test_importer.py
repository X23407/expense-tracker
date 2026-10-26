"""
test_importer.py - Unit tests for date parsing, merchant extraction, and duplicate detection
"""

import io
from pathlib import Path
import tempfile
import pandas as pd
import pytest

from src.db import init_db, add_expense
from src.importer import (
    extract_merchant_from_narration,
    parse_and_process_statement,
    parse_flexible_date,
)


def test_parse_flexible_date():
    """Verify various bank date formats parse to ISO YYYY-MM-DD."""
    assert parse_flexible_date("2026-04-29") == "2026-04-29"
    assert parse_flexible_date("29/04/2026") == "2026-04-29"
    assert parse_flexible_date("29-04-2026") == "2026-04-29"
    assert parse_flexible_date("29-Apr-2026") == "2026-04-29"
    assert parse_flexible_date("29 April 2026") == "2026-04-29"
    assert parse_flexible_date("") is None
    assert parse_flexible_date(None) is None


def test_extract_merchant_from_narration():
    """Verify extraction cleans UPI IDs, POS codes, and infers payment methods."""
    m1, p1 = extract_merchant_from_narration("UPI/SWIGGY/12345@ybl/Order")
    assert m1 == "Swiggy"
    assert p1 == "UPI"

    m2, p2 = extract_merchant_from_narration("POS 4321 AMAZON RETAIL BLR IN")
    assert "Amazon" in m2
    assert p2 == "Credit Card"

    m3, p3 = extract_merchant_from_narration("ECOM/PUR/86809773/IRCTC")
    assert "Irctc" in m3 or "IRCTC" in m3
    assert p3 == "Debit Card"

    m4, p4 = extract_merchant_from_narration("ATM-CASH-WDL/STARBUCKS/9821")
    assert "Starbucks" in m4
    assert p4 == "Cash"


def test_importer_duplicate_detection_and_credit_filtering():
    """Verify importer ignores credit rows and detects both internal and DB duplicates."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp_db:
        db_path = tmp_db.name

    init_db(db_path)

    # Pre-insert an expense into DB to test DB duplicate detection
    add_expense(
        date="2026-05-10",
        amount=500.0,
        category="Food & Dining",
        merchant="Swiggy",
        payment_method="UPI",
        db_path=db_path,
    )

    # Construct synthetic bank CSV content with:
    # 1. Row matching the pre-inserted DB expense (DB duplicate)
    # 2. A new unique valid expense
    # 3. An exact copy of the new valid expense (In-file duplicate)
    # 4. A credit salary row (should be skipped)
    csv_content = """Txn Date,Narration,Debit,Credit,Balance
10/05/2026,UPI/SWIGGY/12345@ybl,500.00,,50000.00
11/05/2026,POS 9982 ZEPTO BLR,350.00,,49650.00
11/05/2026,POS 9982 ZEPTO BLR,350.00,,49300.00
12/05/2026,ACH/SALARY/TECHCORP,,75000.00,124300.00
"""
    csv_file = io.StringIO(csv_content)

    results = parse_and_process_statement(csv_file, existing_db_path=db_path)

    assert results["read"] == 4
    assert results["credits_skipped"] == 1
    # 1 DB duplicate (Swiggy) + 1 In-file duplicate (Zepto repeated) = 2 duplicates skipped
    assert results["duplicates_skipped"] == 2
    # Exactly 1 valid new expense imported (first Zepto row)
    assert results["imported"] == 1

    cleaned = results["cleaned_df"]
    assert len(cleaned) == 1
    assert cleaned.iloc[0]["date"] == "2026-05-11"
    assert "Zepto" in cleaned.iloc[0]["merchant"]
    assert float(cleaned.iloc[0]["amount"]) == 350.0

    # Cleanup temp db
    Path(db_path).unlink(missing_ok=True)


def test_importer_handles_phonepe_header_and_disclaimer_footer():
    """Verify importer ignores PhonePe statement header metadata and disclaimer footer blocks."""
    statement_with_header_and_footer = '''Transaction Statement for 9876543210
Duration,"26 Sept, 2026 - 26 Oct, 2026"

Date,Time,Transaction Details,Transaction ID,UTR,Transaction Type,Credit/debit instrument,Amount
"Oct 24, 2026","02:41 pm","Paid to Campus Tea Stall","T2610241440597128587736","729854585662","DEBIT","Paid by XXXXXX0240","20"
"Oct 22, 2026","02:07 pm","Paid to Campus Stationery","T2610221407039156009851","507730787710","DEBIT","Paid by XXXXXX0240","50"
"Oct 19, 2026","08:16 am","Received from Scholarship Cell","T2610190816081238226819","081608057331","CREDIT","Credited to XXXXXX4127","2500"
"Oct 18, 2026","05:41 pm","Paid to Local Grocery Mart","T2610181741130713154224","707748753529","DEBIT","Paid by XXXXXX0240","245"

This is an automatically generated statement. Customer(s) are requested to immediately notify PhonePe in case
of any errors in the statement at https://support.phonepe.com/statement and visit https://www.phonepe.com/
terms-conditions/ for PhonePe Terms & Conditions and Privacy Policy.

Disclaimer : Do not fall prey to fictitious offers of winning prizes, money circulation schemes and cheap funds,
etc. through SMS, emails and calls. The contents of this email and document are confidential and intended for
the recipient specified in this document. If you received this message by mistake, please inform PhonePe at
https://support.phonepe.com/statement so that we can ensure the recipient's details are corrected.
'''
    csv_stream = io.StringIO(statement_with_header_and_footer)
    res = parse_and_process_statement(csv_stream)

    assert res["error"] is None
    # 4 transaction rows read from table
    assert res["read"] == 4
    # 1 credit deposit skipped (Scholarship Cell)
    assert res["credits_skipped"] == 1
    # 3 debits imported
    assert res["imported"] == 3
    assert len(res["cleaned_df"]) == 3
    # Verify dates parsed properly
    dates = list(res["cleaned_df"]["date"])
    assert "2026-10-24" in dates
    assert "2026-10-22" in dates
    assert "2026-10-18" in dates

