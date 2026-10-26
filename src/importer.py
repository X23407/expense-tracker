"""
importer.py - Bank & UPI Statement CSV Parser and Duplicate Detector

Handles real-world messy bank statement CSV exports:
1. Normalizes varied date formats (DD/MM/YYYY, YYYY-MM-DD, DD-Mon-YYYY).
2. Filters out credit deposits (Salary, refunds) and keeps only debit expenses.
3. Cleans noisy transaction narrations (UPI IDs, POS terminal codes, ATM prefixes).
4. Auto-assigns categories using categorizer.py.
5. Performs duplicate detection against existing DB records AND within the batch
   (matching on Date + Amount + Merchant).
"""

from datetime import datetime
import io
from pathlib import Path
import re
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd

# Ensure root is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.categorizer import ExpenseClassifier, categorize
from src.db import is_duplicate


def parse_flexible_date(date_raw: Any) -> Optional[str]:
    """
    Parse messy date strings across multiple common formats into ISO 'YYYY-MM-DD'.
    Handles formats like:
      - '29/04/2026', '29-04-2026'
      - '2026-04-29', '2026/04/29'
      - '29-Apr-2026', '29-April-2026'
      - '29 Apr 2026'
    """
    if pd.isna(date_raw):
        return None

    raw_str = str(date_raw).strip()
    if not raw_str:
        return None

    # Handle PhonePe and Indian calendar abbreviations e.g. 'Sept' -> 'Sep'
    raw_str = re.sub(r"\bSept\b", "Sep", raw_str, flags=re.IGNORECASE)

    # List of common date formats seen in Indian bank and PhonePe statements
    formats_to_try = [
        "%Y-%m-%d",
        "%d/%m/%Y",
        "%d-%m-%Y",
        "%d-%b-%Y",
        "%d-%B-%Y",
        "%d %b %Y",
        "%d %B %Y",
        "%b %d, %Y",
        "%b %d %Y",
        "%B %d, %Y",
        "%B %d %Y",
        "%d %b, %Y",
        "%Y/%m/%d",
        "%m/%d/%Y",
    ]

    for fmt in formats_to_try:
        try:
            parsed = datetime.strptime(raw_str, fmt)
            return parsed.strftime("%Y-%m-%d")
        except ValueError:
            continue

    # Fallback to pandas to_datetime
    try:
        parsed = pd.to_datetime(raw_str, dayfirst=True)
        return parsed.strftime("%Y-%m-%d")
    except Exception:
        return None


def extract_merchant_from_narration(narration: str) -> Tuple[str, str]:
    """
    Extract a clean merchant name and inferred payment method from raw bank narration.

    Examples:
      'UPI/SWIGGY/12345@ybl/PayTo' -> ('Swiggy', 'UPI')
      'POS 4321 AMAZON RETAIL BLR IN' -> ('Amazon Retail', 'Credit Card')
      'ECOM/PUR/9982/ZEPTO' -> ('Zepto', 'Debit Card')
      'ATM-CASH-WDL/STARBUCKS/123' -> ('Starbucks', 'Cash')
      'ACH/NETFLIX.COM/REC' -> ('Netflix', 'Credit Card')
    """
    if not isinstance(narration, str) or not narration.strip():
        return "Unknown Merchant", "UPI"

    text = narration.strip()

    # Determine payment method heuristic
    upper = text.upper()
    if "UPI" in upper:
        pay_method = "UPI"
    elif "POS" in upper or "ACH" in upper or "CREDIT" in upper:
        pay_method = "Credit Card"
    elif "ECOM" in upper or "DEBIT" in upper:
        pay_method = "Debit Card"
    elif "ATM" in upper or "CASH" in upper:
        pay_method = "Cash"
    else:
        pay_method = "UPI"

    # Banking noise tokens that should NOT be treated as merchant names
    BANKING_NOISE_TOKENS = {
        "UPI", "POS", "ECOM", "ACH", "ATM", "PUR", "BIL", "INB", "NEFT", "IMPS", "RTGS",
        "PAYTO", "PAYMENT", "ORDER", "TXN", "REF", "CARD", "WDL", "ATM-CASH-WDL", "CR", "DR",
        "NETBANKING", "TRANSFER", "SALARY", "DEBIT", "CREDIT"
    }

    # Step 1: Strip standard prefixes (including PhonePe and GPay styles)
    cleaned = re.sub(
        r"^(Paid to|Received from|Transfer to|UPI|POS|ECOM|ACH|ATM-CASH-WDL|IMPS|NEFT|RTGS|CARD|PUR|BIL)[/\s\-_]+",
        "",
        text,
        flags=re.IGNORECASE,
    )

    # Step 2: Handle slash or hyphen segments: e.g. SWIGGY/12345@ybl/PayTo or PUR/86809773/IRCTC
    if "/" in cleaned or "-" in cleaned:
        raw_segments = re.split(r"[/\-_]", cleaned)
        segments = [s.strip() for s in raw_segments if s.strip()]
        # Find the first segment that is NOT a numeric ID, NOT an email/handle, and NOT a banking noise token
        candidate = None
        for seg in segments:
            seg_upper = seg.upper()
            if not seg.isdigit() and "@" not in seg and seg_upper not in BANKING_NOISE_TOKENS and len(seg) > 1:
                candidate = seg
                break
        if candidate:
            cleaned = candidate
        elif segments:
            cleaned = segments[-1]  # fallback to last segment if all are ambiguous

    # Step 3: Strip trailing UPI handles or numbers, e.g. '@okaxis', '@ybl', '12345'
    cleaned = re.sub(r"@[a-zA-Z0-9_\-]+", "", cleaned)
    cleaned = re.sub(r"\b\d{4,}\b", "", cleaned)  # remove long reference numbers
    cleaned = re.sub(r"\b(BLR|MUM|DEL|IN|PAYTO|PAYMENT|BILLPAY|PURCHASE|ORDER)\b", "", cleaned, flags=re.IGNORECASE)

    # Step 4: Clean punctuation and multiple spaces
    cleaned = re.sub(r"[^a-zA-Z0-9\s&']", " ", cleaned)
    cleaned = " ".join(cleaned.split()).strip()

    if not cleaned:
        cleaned = "Unknown Merchant"

    # Title-case for clean presentation
    cleaned = cleaned.title()

    # Common brand normalization dictionary
    brand_map = {
        "Swiggy": "Swiggy",
        "Zomato": "Zomato",
        "Dominos": "Domino's Pizza",
        "Domino'S Pizza": "Domino's Pizza",
        "Bigbasket": "BigBasket",
        "Zepto": "Zepto",
        "Blinkit": "Blinkit",
        "Dmart": "DMart",
        "Uber": "Uber",
        "Ola": "Ola Cabs",
        "Ola Cabs": "Ola Cabs",
        "Rapido": "Rapido",
        "Amazon": "Amazon India",
        "Amazon Retail": "Amazon India",
        "Flipkart": "Flipkart",
        "Myntra": "Myntra",
        "Apollo Pharmacy": "Apollo Pharmacy",
        "Apollo": "Apollo Pharmacy",
        "Netflix": "Netflix India",
        "Spotify": "Spotify Premium",
        "Bescom": "BESCOM Electricity",
        "Bwssb": "BWSSB Water Bill",
        "Indane": "Indane Gas Agency",
        "Starbucks": "Starbucks",
        "Pvr": "PVR Cinemas",
        "Bookmyshow": "BookMyShow",
    }

    for key, val in brand_map.items():
        if key.lower() in cleaned.lower():
            cleaned = val
            break

    return cleaned, pay_method


def read_statement_csv(file_or_path: Union[str, Path, Any]) -> pd.DataFrame:
    """
    Robust CSV parser capable of handling real-world bank and UPI statements
    (e.g., PhonePe, Google Pay, Paytm, HDFC, SBI, ICICI) with:
    1. Metadata header blocks (e.g. 'Transaction Statement for...',
       'Duration, "26 Sept, 2026 - 26 Oct, 2026"', account numbers, bank notices).
    2. Disclaimer, summary, and legal footers at the bottom of the export.
    3. Diverse text encodings (UTF-8, UTF-8 with BOM, Latin-1 / Windows-1252).
    4. Semicolon or tab delimiters with automatic fallback.
    """
    text = ""
    # Case 1: Stream or buffer object (e.g. Streamlit UploadedFile, BytesIO, StringIO)
    if hasattr(file_or_path, "read"):
        raw = file_or_path.read()
        if hasattr(file_or_path, "seek"):
            file_or_path.seek(0)
        if isinstance(raw, bytes):
            for enc in ["utf-8-sig", "utf-8", "latin-1"]:
                try:
                    text = raw.decode(enc)
                    break
                except UnicodeDecodeError:
                    continue
            else:
                text = raw.decode("utf-8", errors="replace")
        else:
            text = str(raw)
    # Case 2: File path on disk
    elif isinstance(file_or_path, (str, Path)):
        p = Path(file_or_path)
        if p.exists() and p.is_file():
            raw_bytes = p.read_bytes()
            for enc in ["utf-8-sig", "utf-8", "latin-1"]:
                try:
                    text = raw_bytes.decode(enc)
                    break
                except UnicodeDecodeError:
                    continue
            else:
                text = raw_bytes.decode("utf-8", errors="replace")
        else:
            text = str(file_or_path)
    else:
        text = str(file_or_path)

    text = text.strip()
    if not text:
        return pd.DataFrame()

    lines = text.splitlines()

    # Financial keywords indicating a table header row
    HEADER_COLS = [
        "date", "txn date", "txndate", "transaction date", "value date",
        "amount", "debit", "withdrawal", "spent", "dr",
        "credit", "deposit", "cr",
        "particulars", "description", "details", "narration", "transaction details", "remarks",
        "transaction type", "txn type", "type", "utr", "chq", "ref no", "balance", "instrument"
    ]

    FOOTER_PATTERNS = [
        "this is an automatically generated statement",
        "customer(s) are requested to immediately notify",
        "customer(s) are requested",
        "disclaimer",
        "terms-conditions",
        "terms & conditions",
        "support.phonepe.com",
        "fictitious offers",
        "confidential and intended",
        "end of statement",
        "grand total",
        "total withdrawal",
        "total debit",
        "total credit",
        "generated on",
        "page ",
        "***",
        "important notice",
        "note:",
    ]

    header_idx = None
    best_score = 0

    # Scan the first 50 lines to find the true transaction table header
    for idx, line in enumerate(lines[:50]):
        clean = line.strip().lower()
        if not clean:
            continue
        score = sum(1 for kw in HEADER_COLS if kw in clean)
        has_date = any(dk in clean for dk in ["date", "txn date", "value date", "txndate"])
        if has_date and score >= 2 and score > best_score:
            best_score = score
            header_idx = idx

    # If no header containing date + financial keyword was found, default to line 0
    if header_idx is None:
        header_idx = 0

    valid_lines = []
    for line in lines[header_idx:]:
        clean = line.strip().lower()
        if not clean:
            continue
        # Truncate statement as soon as a footer / disclaimer marker is reached
        if any(pat in clean for pat in FOOTER_PATTERNS):
            break
        valid_lines.append(line)

    if not valid_lines:
        return pd.DataFrame()

    clean_csv_str = "\n".join(valid_lines)

    try:
        # Standard fast parse with C engine
        return pd.read_csv(io.StringIO(clean_csv_str))
    except Exception:
        # Resilient fallback with python engine and auto-separator detection
        return pd.read_csv(io.StringIO(clean_csv_str), sep=None, engine="python", on_bad_lines="skip")


def parse_and_process_statement(
    file_or_path: Union[str, Path, Any],
    existing_db_path: Optional[Union[str, Path]] = None,
    classifier: Optional[ExpenseClassifier] = None,
    use_ml: bool = False,
) -> Dict[str, Any]:
    """
    Parse a bank or UPI statement CSV, clean fields, detect duplicates, and categorize.

    Returns a dict with:
      - 'read': total rows in statement
      - 'credits_skipped': count of non-debit rows skipped
      - 'duplicates_skipped': count of duplicate debit transactions identified
      - 'imported': count of valid new transactions ready to import
      - 'uncategorized': count assigned to 'Miscellaneous'
      - 'cleaned_df': DataFrame with columns [date, amount, category, merchant, payment_method, note]
      - 'duplicate_df': DataFrame of detected duplicate rows
      - 'error': error message string if parsing failed
    """
    result: Dict[str, Any] = {
        "read": 0,
        "credits_skipped": 0,
        "duplicates_skipped": 0,
        "imported": 0,
        "uncategorized": 0,
        "cleaned_df": pd.DataFrame(),
        "duplicate_df": pd.DataFrame(),
        "error": None,
    }

    try:
        raw_df = read_statement_csv(file_or_path)
    except Exception as e:
        result["error"] = f"Failed to read CSV file: {str(e)}"
        return result

    result["read"] = len(raw_df)
    if raw_df.empty:
        result["error"] = "CSV file is empty."
        return result

    # Standardize column headers: lowercase & strip
    raw_df.columns = [str(c).strip().lower() for c in raw_df.columns]

    # 1. Identify Date column
    date_col = None
    for c in raw_df.columns:
        if any(term in c for term in ["date", "txn date", "txndate", "transaction date", "value date"]):
            date_col = c
            break
    if not date_col:
        date_col = raw_df.columns[0]  # default fallback

    # 2. Identify Narration / Description column
    desc_col = None
    for c in raw_df.columns:
        if any(term in c for term in ["narration", "description", "details", "remarks", "particulars", "merchant"]):
            desc_col = c
            break
    if not desc_col:
        for c in raw_df.columns:
            if c != date_col and raw_df[c].dtype == object:
                desc_col = c
                break

    # 3. Identify Debit / Amount column
    debit_col = None
    for c in raw_df.columns:
        if any(term in c for term in ["debit", "withdrawal", "spent", "dr"]):
            debit_col = c
            break

    # If no explicit Debit column, check for general 'amount' column
    if not debit_col:
        for c in raw_df.columns:
            if any(term in c for term in ["amount", "txn amount"]):
                debit_col = c
                break

    if not debit_col:
        result["error"] = "Could not find a Debit or Amount column in the CSV."
        return result

    # Check for separate Credit column to identify non-expenses
    credit_col = None
    for c in raw_df.columns:
        if any(term in c for term in ["credit", "deposit", "cr"]):
            credit_col = c
            break

    # Check for Transaction Type column (e.g. PhonePe DEBIT vs CREDIT)
    txn_type_col = None
    for c in raw_df.columns:
        if any(term in c for term in ["transaction type", "txn type"]):
            txn_type_col = c
            break

    cleaned_records: List[Dict[str, Any]] = []
    duplicate_records: List[Dict[str, Any]] = []
    # Track (date, round(amount, 2), merchant_lower) within current batch
    seen_in_batch = set()

    for _, row in raw_df.iterrows():
        # Check transaction type column if present (e.g. PhonePe CREDIT row)
        if txn_type_col:
            ttype = str(row.get(txn_type_col, "")).strip().upper()
            if "CREDIT" in ttype or ttype == "CR":
                result["credits_skipped"] += 1
                continue

        # Check credit: if debit is empty/zero and credit is present, skip (it's income/deposit)
        debit_val = row.get(debit_col, "")
        if pd.isna(debit_val) or str(debit_val).strip() == "":
            result["credits_skipped"] += 1
            continue

        # Parse numerical amount
        try:
            # Remove currency symbols and commas e.g. "₹ 1,234.50"
            clean_amt_str = re.sub(r"[^\d.]", "", str(debit_val))
            amount = float(clean_amt_str)
            if amount <= 0:
                result["credits_skipped"] += 1
                continue
        except (ValueError, TypeError):
            result["credits_skipped"] += 1
            continue

        # Parse date
        parsed_date = parse_flexible_date(row.get(date_col, ""))
        if not parsed_date:
            continue

        # Parse narration & merchant
        raw_narration = str(row.get(desc_col, "") or "")
        if "received from" in raw_narration.lower():
            result["credits_skipped"] += 1
            continue
        merchant, pay_method = extract_merchant_from_narration(raw_narration)

        # Categorize
        category, source = categorize(merchant, classifier=classifier, use_ml=use_ml)

        record = {
            "date": parsed_date,
            "amount": round(amount, 2),
            "category": category,
            "merchant": merchant,
            "payment_method": pay_method,
            "note": f"Imported from statement: {raw_narration[:50].strip()}",
        }

        # Duplicate Check:
        # Check 1: In-batch duplicate (same date + amount + merchant within this CSV)
        batch_key = (parsed_date, round(amount, 2), merchant.lower())
        is_batch_dup = batch_key in seen_in_batch

        # Check 2: Database duplicate (already exists in SQLite)
        is_db_dup = is_duplicate(parsed_date, round(amount, 2), merchant, db_path=existing_db_path)

        if is_batch_dup or is_db_dup:
            result["duplicates_skipped"] += 1
            record["duplicate_reason"] = "Already in DB" if is_db_dup else "Repeated in File"
            duplicate_records.append(record)
        else:
            seen_in_batch.add(batch_key)
            cleaned_records.append(record)
            if category == "Miscellaneous":
                result["uncategorized"] += 1

    result["imported"] = len(cleaned_records)
    result["cleaned_df"] = pd.DataFrame(cleaned_records)
    result["duplicate_df"] = pd.DataFrame(duplicate_records)

    return result
