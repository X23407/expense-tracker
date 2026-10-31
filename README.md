# 💳 Personal Expense Tracker with Analytics, Smart Insights & Spending Graph

A full-stack financial analytics application built with **Python 3.10+**, **Streamlit**, **NetworkX**, **Plotly**, **SQLite**, and **Scikit-Learn**. 

Designed for personal expense tracking, automated bank/UPI statement parsing, budget management, graph-based relationship modeling, and statistical spending intelligence.

---

## 🌟 Key Features

| Feature | Primary Technologies | Core Capabilities |
| :--- | :--- | :--- |
| **1. Expense Manager (CRUD)** | SQLite, Python `re`, Pandas | Direct transaction entry, full search & category filtering, bank/UPI statement ingestion with automatic header/disclaimer stripping, duplicate detection, and full CSV backup/restore. |
| **2. Analytics Dashboard** | Plotly Express & Graph Objects | Executive financial KPI cards, dynamic timespan filtering (3M, 6M, 12M, All), category breakdown donut, top merchants by spend, weekday vs weekend comparison, and customizable budget utilization tracking. |
| **3. Spending Graph** | NetworkX, Plotly Spring Layout | Tripartite graph (`Merchant`, `Category`, `Payment Method`), degree centrality, weighted degree (node strength), betweenness centrality (bridge channels), and Newman's greedy modularity community detection. |
| **4. Smart Insights** | Plain Python / NumPy, Scikit-Learn | Z-Score statistical anomaly detection ($z \ge 2.5\sigma$ with automated explanations), 3-month weighted moving average forecast ($w=[0.2, 0.3, 0.5]$) with backtesting, and deterministic keyword rules with TF-IDF Logistic Regression. |

---

## 📂 Project Structure

```text
expense/
├── app.py                      # Main Streamlit web application controller
├── requirements.txt            # Streamlit Community Cloud dependencies
├── README.md                   # Setup, deployment, and architecture guide
├── data/
│   ├── sample_expenses.csv     # 6-month calibrated Indian expense dataset (~312 rows)
│   ├── sample_bank_statement.csv # Realistic bank statement (header metadata, UPI narrations, duplicates)
│   └── expenses.db             # Local SQLite database (auto-created on startup)
├── src/
│   ├── __init__.py
│   ├── db.py                   # SQLite schema, CRUD operations & budget targets
│   ├── sample_data.py          # Calibrated realistic financial dataset & anomaly generator
│   ├── categorizer.py          # Case-insensitive rule engine + TF-IDF Logistic Regression classifier
│   ├── importer.py             # Statement reader (auto-header/footer detection), merchant cleaner & duplicate detector
│   ├── analytics.py            # Financial KPIs, aggregations & interactive Plotly figures
│   ├── insights.py             # Z-score outlier detector & 3-month WMA forecaster
│   └── graph.py                # NetworkX graph builder, centralities, communities & plot
└── tests/
    ├── test_categorizer.py     # Unit tests for keyword rules & ML pipeline
    ├── test_importer.py       # Unit tests for messy dates, PhonePe headers/footers & duplicates
    ├── test_anomaly.py        # Unit tests for Z-score calculations & thresholding
    ├── test_forecast.py       # Unit tests for 3-month WMA mathematical weights
    └── test_graph.py          # Unit tests for NetworkX graph construction & metrics
```

---

## 🚀 Quickstart & Local Setup

### 1. Clone the repository & enter workspace
```bash
git clone https://github.com/x23407/expense-tracker/
cd expense-tracker
```

### 2. Create and activate a Python virtual environment
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit application
```bash
python -m streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## ☁️ Deployment on Streamlit Community Cloud

This project is pre-configured for one-click deployment on **Streamlit Community Cloud**:

1. Push your repository to GitHub.
2. Sign in to [share.streamlit.io](https://share.streamlit.io).
3. Click **"New App"**, select your repository, branch (`main`), and set **Main file path** to `app.py`.
4. Click **"Deploy!"**.

### Handling Ephemeral Filesystems on Streamlit Cloud
Streamlit Community Cloud runs inside stateless ephemeral Docker containers. Files written to local disk (such as `data/expenses.db`) reset upon container sleep or rebuild. 

This project handles this constraint via a 3-part resilience architecture:
1. **Auto-Seed on Empty Database:** On app boot, `load_sample_data_if_empty()` in `src/sample_data.py` inspects the SQLite database. If empty, it populates the calibrated dataset and default budgets so the app is immediately functional out-of-the-box.
2. **One-Click Backup & Restore:** In the **Expenses** page, users can download a complete CSV backup (`expenses_backup_YYYYMMDD.csv`) and restore it on any fresh instance via the Bank/UPI CSV Importer.
3. **Isolated Database Layer:** All database operations are strictly encapsulated in `src/db.py`. To swap SQLite for a cloud database (e.g. Supabase, PostgreSQL, Neon), **only `src/db.py` needs to be modified**. Frontend and analytics code remain untouched.

---

## 🧪 Running the Automated Test Suite

A complete test suite is provided using `pytest` to guarantee mathematical and algorithmic correctness:

```bash
python -m pytest tests/ -v
```

### Test Coverage Highlights:
- **`test_categorizer.py`**: Case-insensitive rule matching, domain keyword coverage, unknown merchant fallback, and Scikit-Learn TF-IDF classifier cross-validation training.
- **`test_importer.py`**: Flexible date format parsing (`DD/MM/YYYY`, `YYYY-MM-DD`, `DD-Mon-YYYY`), PhonePe header metadata stripping, disclaimer footer truncation, and duplicate prevention.
- **`test_anomaly.py`**: Mathematical Z-score verification with Bessel's correction, threshold enforcement ($z \ge 2.5$), and reason string formatting.
- **`test_forecast.py`**: Mathematical weights evaluation ($0.2 \cdot M_{-3} + 0.3 \cdot M_{-2} + 0.5 \cdot M_{-1}$), per-category projections, and budget warning comparisons.
- **`test_graph.py`**: NetworkX graph node types (`Merchant`, `Category`, `Payment Method`), edge weights, degree centrality, betweenness centrality, and greedy modularity community partitions.