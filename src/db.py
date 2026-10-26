"""
db.py - SQLite Database Management Layer

This module is the ONLY module in the application that directly accesses
the SQLite database. Keeping all SQL queries and schema management here ensures
that the backend storage can be swapped to PostgreSQL, Supabase, or MySQL in
the future simply by modifying this single file.
"""

from datetime import datetime
from pathlib import Path
import sqlite3
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd

# Default categories and recommended monthly budgets (in INR)
# Calibrated for realistic student / young professional monthly spending (~₹32,000/month)
DEFAULT_BUDGETS: Dict[str, float] = {
    "Food & Dining": 6000.0,
    "Groceries": 3500.0,
    "Transport": 1500.0,
    "Rent & Utilities": 12000.0,
    "Entertainment": 1000.0,
    "Shopping": 2500.0,
    "Health": 1000.0,
    "Education": 1500.0,
    "Subscriptions": 300.0,
    "Travel": 1500.0,
    "Miscellaneous": 1500.0,
}

PAYMENT_METHODS: List[str] = ["UPI", "Credit Card", "Debit Card", "Cash"]


def get_db_path() -> Path:
    """Return the resolved absolute path to the SQLite database file."""
    base_dir = Path(__file__).resolve().parent.parent
    data_dir = base_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir / "expenses.db"


def get_connection(db_path: Optional[Union[str, Path]] = None) -> sqlite3.Connection:
    """
    Establish and return a connection to the SQLite database.

    Enables foreign keys and sets row_factory to sqlite3.Row for dict-like row access.
    """
    resolved_path = Path(db_path) if db_path else get_db_path()
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(resolved_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def init_db(db_path: Optional[Union[str, Path]] = None) -> None:
    """
    Initialize SQLite schema for expenses and budgets tables with indexes.

    Also populates the budgets table with standard default limits if it is empty.
    """
    conn = get_connection(db_path)
    try:
        with conn:
            # 1. Expenses Table
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS expenses (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT NOT NULL,
                    amount REAL NOT NULL CHECK(amount >= 0),
                    category TEXT NOT NULL,
                    merchant TEXT NOT NULL,
                    payment_method TEXT NOT NULL,
                    note TEXT DEFAULT '',
                    created_at TEXT NOT NULL
                );
                """
            )

            # Performance indexes for common filters and duplicate checks
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_expenses_date ON expenses(date);"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_expenses_category ON expenses(category);"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_expenses_dup_check ON expenses(date, amount, merchant);"
            )

            # 2. Budgets Table
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS budgets (
                    category TEXT PRIMARY KEY,
                    monthly_limit REAL NOT NULL CHECK(monthly_limit >= 0)
                );
                """
            )

            # Populate default budgets if table is empty
            cursor = conn.execute("SELECT COUNT(*) AS count FROM budgets;")
            count = cursor.fetchone()["count"]
            if count == 0:
                conn.executemany(
                    "INSERT INTO budgets (category, monthly_limit) VALUES (?, ?);",
                    list(DEFAULT_BUDGETS.items()),
                )
    finally:
        conn.close()


def add_expense(
    date: str,
    amount: float,
    category: str,
    merchant: str,
    payment_method: str,
    note: str = "",
    db_path: Optional[Union[str, Path]] = None,
) -> int:
    """
    Insert a single expense record into the database.

    Returns:
        The auto-generated integer ID of the inserted row.
    """
    created_at = datetime.now().isoformat()
    conn = get_connection(db_path)
    try:
        with conn:
            cursor = conn.execute(
                """
                INSERT INTO expenses (date, amount, category, merchant, payment_method, note, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    date.strip(),
                    float(amount),
                    category.strip(),
                    merchant.strip(),
                    payment_method.strip(),
                    note.strip(),
                    created_at,
                ),
            )
            return cursor.lastrowid
    finally:
        conn.close()


def update_expense(
    expense_id: int,
    date: str,
    amount: float,
    category: str,
    merchant: str,
    payment_method: str,
    note: str = "",
    db_path: Optional[Union[str, Path]] = None,
) -> bool:
    """
    Update an existing expense record by ID.

    Returns:
        True if a row was updated, False otherwise.
    """
    conn = get_connection(db_path)
    try:
        with conn:
            cursor = conn.execute(
                """
                UPDATE expenses
                SET date = ?, amount = ?, category = ?, merchant = ?, payment_method = ?, note = ?
                WHERE id = ?;
                """,
                (
                    date.strip(),
                    float(amount),
                    category.strip(),
                    merchant.strip(),
                    payment_method.strip(),
                    note.strip(),
                    int(expense_id),
                ),
            )
            return cursor.rowcount > 0
    finally:
        conn.close()


def delete_expense(
    expense_id: int, db_path: Optional[Union[str, Path]] = None
) -> bool:
    """
    Delete an expense record by ID.

    Returns:
        True if a row was deleted, False otherwise.
    """
    conn = get_connection(db_path)
    try:
        with conn:
            cursor = conn.execute(
                "DELETE FROM expenses WHERE id = ?;", (int(expense_id),)
            )
            return cursor.rowcount > 0
    finally:
        conn.close()


def get_expense_by_id(
    expense_id: int, db_path: Optional[Union[str, Path]] = None
) -> Optional[Dict[str, Any]]:
    """Fetch a single expense by ID as a dictionary, or None if not found."""
    conn = get_connection(db_path)
    try:
        cursor = conn.execute(
            "SELECT * FROM expenses WHERE id = ?;", (int(expense_id),)
        )
        row = cursor.fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def get_expenses(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    category: Optional[Union[str, List[str]]] = None,
    payment_method: Optional[Union[str, List[str]]] = None,
    search: Optional[str] = None,
    db_path: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """
    Retrieve expenses matching specified filter criteria as a pandas DataFrame.

    All filters are optional. If no filters are provided, returns all expenses
    ordered by date descending.
    """
    conn = get_connection(db_path)
    try:
        query = "SELECT * FROM expenses WHERE 1=1"
        params: List[Any] = []

        if start_date:
            query += " AND date >= ?"
            params.append(start_date)

        if end_date:
            query += " AND date <= ?"
            params.append(end_date)

        if category:
            if isinstance(category, list) and category:
                placeholders = ",".join("?" for _ in category)
                query += f" AND category IN ({placeholders})"
                params.extend(category)
            elif isinstance(category, str) and category != "All":
                query += " AND category = ?"
                params.append(category)

        if payment_method:
            if isinstance(payment_method, list) and payment_method:
                placeholders = ",".join("?" for _ in payment_method)
                query += f" AND payment_method IN ({placeholders})"
                params.extend(payment_method)
            elif isinstance(payment_method, str) and payment_method != "All":
                query += " AND payment_method = ?"
                params.append(payment_method)

        if search:
            search_param = f"%{search.strip().lower()}%"
            query += " AND (LOWER(merchant) LIKE ? OR LOWER(note) LIKE ?)"
            params.extend([search_param, search_param])

        query += " ORDER BY date DESC, id DESC;"

        df = pd.read_sql_query(query, conn, params=params)
        return df
    finally:
        conn.close()


def is_duplicate(
    date_str: str,
    amount: float,
    merchant: str,
    db_path: Optional[Union[str, Path]] = None,
) -> bool:
    """
    Check if an expense with the identical date, amount (rounded to 2 decimal places),
    and cleaned merchant name already exists in the database.
    """
    conn = get_connection(db_path)
    try:
        # Check within a tiny delta (0.01) to account for floating point comparisons
        cursor = conn.execute(
            """
            SELECT 1 FROM expenses
            WHERE date = ?
              AND ABS(amount - ?) < 0.01
              AND LOWER(TRIM(merchant)) = LOWER(TRIM(?))
            LIMIT 1;
            """,
            (date_str.strip(), float(amount), merchant.strip()),
        )
        return cursor.fetchone() is not None
    except sqlite3.OperationalError:
        # Table does not exist yet (e.g. uninitialized test DB)
        return False
    finally:
        conn.close()


def bulk_insert_expenses(
    records: Union[List[Dict[str, Any]], pd.DataFrame],
    db_path: Optional[Union[str, Path]] = None,
) -> int:
    """
    Insert multiple expense records in a single database transaction.

    Accepts either a list of dictionaries or a pandas DataFrame.
    """
    if isinstance(records, pd.DataFrame):
        data_to_insert = records[
            ["date", "amount", "category", "merchant", "payment_method", "note"]
        ].to_dict(orient="records")
    else:
        data_to_insert = records

    if not data_to_insert:
        return 0

    now_iso = datetime.now().isoformat()
    rows = [
        (
            r["date"],
            float(r["amount"]),
            r["category"],
            r["merchant"],
            r["payment_method"],
            r.get("note", "") or "",
            r.get("created_at", now_iso),
        )
        for r in data_to_insert
    ]

    conn = get_connection(db_path)
    try:
        with conn:
            cursor = conn.executemany(
                """
                INSERT INTO expenses (date, amount, category, merchant, payment_method, note, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?);
                """,
                rows,
            )
            return cursor.rowcount
    finally:
        conn.close()


def get_budgets(db_path: Optional[Union[str, Path]] = None) -> pd.DataFrame:
    """Retrieve all category monthly budgets as a DataFrame."""
    conn = get_connection(db_path)
    try:
        df = pd.read_sql_query(
            "SELECT category, monthly_limit FROM budgets ORDER BY category ASC;", conn
        )
        return df
    finally:
        conn.close()


def set_budget(
    category: str,
    monthly_limit: float,
    db_path: Optional[Union[str, Path]] = None,
) -> bool:
    """Insert or update the monthly spending limit for a specific category."""
    conn = get_connection(db_path)
    try:
        with conn:
            conn.execute(
                """
                INSERT INTO budgets (category, monthly_limit)
                VALUES (?, ?)
                ON CONFLICT(category) DO UPDATE SET monthly_limit = excluded.monthly_limit;
                """,
                (category.strip(), float(monthly_limit)),
            )
            return True
    finally:
        conn.close()


def clear_all_expenses(db_path: Optional[Union[str, Path]] = None) -> None:
    """Clear all records from the expenses table (for testing or resetting)."""
    conn = get_connection(db_path)
    try:
        with conn:
            conn.execute("DELETE FROM expenses;")
    finally:
        conn.close()


def get_all_categories(db_path: Optional[Union[str, Path]] = None) -> List[str]:
    """Return a sorted list of unique categories currently in the database or defaults."""
    conn = get_connection(db_path)
    try:
        cursor = conn.execute(
            "SELECT DISTINCT category FROM expenses UNION SELECT category FROM budgets ORDER BY 1 ASC;"
        )
        results = [row[0] for row in cursor.fetchall() if row[0]]
        return results if results else list(DEFAULT_BUDGETS.keys())
    finally:
        conn.close()


def get_all_merchants(db_path: Optional[Union[str, Path]] = None) -> List[str]:
    """Return a sorted list of unique merchants in the database."""
    conn = get_connection(db_path)
    try:
        cursor = conn.execute("SELECT DISTINCT merchant FROM expenses ORDER BY merchant ASC;")
        return [row[0] for row in cursor.fetchall() if row[0]]
    finally:
        conn.close()


def get_all_payment_methods(db_path: Optional[Union[str, Path]] = None) -> List[str]:
    """Return unique payment methods present in the database or defaults."""
    conn = get_connection(db_path)
    try:
        cursor = conn.execute(
            "SELECT DISTINCT payment_method FROM expenses ORDER BY payment_method ASC;"
        )
        methods = [row[0] for row in cursor.fetchall() if row[0]]
        return methods if methods else PAYMENT_METHODS
    finally:
        conn.close()
