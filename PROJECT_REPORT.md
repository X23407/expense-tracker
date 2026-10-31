# IT WORKSHOP III: PROJECT REPORT
## EXPENSE TRACKER & FINANCIAL ANALYTICS SYSTEM

---

### TABLE OF CONTENTS
1. [Introduction & Project Overview](#1-introduction--project-overview)
2. [Key System Features & Capabilities](#2-key-system-features--capabilities)
3. [System Architecture & Data Flow](#3-system-architecture--data-flow)
4. [Python Concepts & Libraries Used](#4-python-concepts--libraries-used)
5. [Mathematical & Algorithmic Foundations](#5-mathematical--algorithmic-foundations)
6. [Hosting Platform & Cloud Persistence Architecture](#6-hosting-platform--cloud-persistence-architecture)
7. [Conclusion & Future Scope](#7-conclusion--future-scope)

---

### 1. Introduction & Project Overview

**Expense Tracker** is a full-stack, data-driven personal finance and analytical web application built with Python. Modern digital spending—especially across UPI platforms such as PhonePe and Google Pay, debit/credit cards, and net banking—frequently leads to fragmented financial awareness. Traditional spreadsheet logging is tedious, error-prone, and lacks proactive intelligence.

This project addresses these challenges by delivering an end-to-end financial platform that combines:
- **Relational Data Management (CRUD)**: Fast local SQLite transaction recording and editing.
- **Automated Statement Ingestion**: Intelligent parsing of raw CSV and JSON statements from Indian UPI and banking platforms.
- **Interactive Visual Analytics**: Dynamic charts covering category contributions, monthly trends, top merchants, weekday vs. weekend behaviors, and budget utilization.
- **Graph Theory & Network Analysis**: Tripartite graph modeling connecting merchants, spending categories, and payment channels.
- **Statistical Machine Intelligence**: Z-score outlier detection, 3-month weighted moving average (WMA) forecasting, and machine-learning-assisted transaction categorization.

---

### 2. Key System Features & Capabilities

#### 2.1. Executive Analytics Dashboard
- **Primary Outflow Overview**: Displays total monthly spend with Month-over-Month (MoM) percentage change indicators, daily averages, and all-time expenditure volume.
- **Status Signal Tiles**: Immediate status flags for detected outliers and projected next-month commitments.
- **Compact Metrics Strip**: Quick reference for transaction volume, dominant payment method, highest-spending category, and largest single purchase.
- **Interactive Visualizations**:
  - *Monthly Outflow Trend*: Historical spending trajectory with dynamic timespan filtering (3M, 6M, 12M, All) and forecast extensions.
  - *Category Distribution Donut*: Percentage breakdown across essential spending heads.
  - *Top 10 Merchants Horizontal Bar*: Identifies dominant vendors and store chains by cumulative spend.
  - *Weekday vs. Weekend Habits*: Comparative analysis of expenditure patterns and average transaction size.
  - *Budget vs. Actual Progress*: Tracks actual spend against customizable monthly limits per category with color-coded status indicators (*Within Budget*, *Near Limit*, *Over Budget*).

#### 2.2. Transaction Management (CRUD & Danger Zone)
- **Create (Quick Add)**: Real-time transaction entry with intelligent keyword-based category pre-selection.
- **Read & Filter**: Multi-parameter search supporting keyword text search, category dropdowns, payment method filters, and date ranges.
- **Update & Delete**: In-place editing of individual transaction details or removal of erroneous entries.
- **Two-Layer Deletion Confirmation (Danger Zone)**: A protected administrative operation that requires typing a specific confirmation phrase (`DELETE ALL MY DATA`) before clearing database records, preventing accidental data loss.

#### 2.3. Bank & UPI Statement Ingestion (CSV & JSON)
- **Header & Footer Stripping**: Automatically detects table header rows by scanning for financial keywords, discarding PhonePe/banking account metadata headers and legal disclaimer footers.
- **Payment Instrument Disqualification**: Differentiates payment accounts (e.g., `Paid by XXXXXX0240`) from true transaction values, preventing masked account numbers from being misinterpreted as expenditure amounts.
- **Credit & Deposit Filtering**: Identifies and skips salary credits, refunds, and incoming peer transfers so only genuine debits are ingested.
- **Two-Tier Duplicate Detection**: Rejects duplicates repeated within the imported file as well as records already stored in the database.
- **Multi-Format Support**: Ingests both standard bank CSVs and structured JSON statements.

#### 2.4. Spending Graph & Network Analytics
- Models financial activity as an undirected tripartite graph $G = (V, E)$ where nodes $V$ represent **Merchants**, **Categories**, and **Payment Methods**, and weighted edges $E$ denote transaction volume.
- **Degree Centrality**: Identifies high-frequency connection hubs.
- **Betweenness Centrality**: Highlights bridge nodes connecting disparate spending clusters.
- **Greedy Modularity Community Detection**: Groups related vendors and payment modes into behavioral habit clusters.
- **Interactive Force-Directed Visualizer**: 2D Spring Layout visualization with hover tooltips and plain-English summary cards.

#### 2.5. Smart Insights & Machine Intelligence
- **Statistical Anomaly Detection**: Category-specific Z-score evaluation identifying unusual transactions ($z \ge 2.5\sigma$) with human-readable diagnostic explanations.
- **3-Month Weighted Moving Average (WMA) Forecast**: Computes next-month projections using recency-weighted coefficients ($w = [0.2, 0.3, 0.5]$), coupled with backtesting accuracy metrics (Mean Absolute Error and MAPE).
- **Categorization Test Bench**: Deterministic keyword engine paired with an optional Scikit-Learn TF-IDF Vectorizer and Logistic Regression pipeline for unknown merchant classification.

---

### 3. System Architecture & Data Flow

```text
[ Raw Statements (CSV/JSON) ] ──┐
                                ▼
[ User Input (Web Form) ] ──► [ Importer & Cleaner ] ──► [ Rule / ML Categorizer ]
                                                              │
                                                              ▼
                                                   [ SQLite Database (expenses.db) ]
                                                              │
                    ┌─────────────────────────────────────────┼────────────────────────────────────────┐
                    ▼                                         ▼                                        ▼
          [ Analytics Engine ]                      [ NetworkX Graph Engine ]                [ Smart Insights Engine ]
        - KPIs & MoM Calculations                  - Tripartite Graph (V, E)                - Z-Score Anomaly Detector
        - Time-Series Aggregations                 - Centrality Metrics & Clusters          - 3-Month WMA Forecaster
                    │                                         │                                        │
                    ▼                                         ▼                                        ▼
          [ Plotly Visualizations ]                 [ Spring Layout Plotly Plot ]            [ Diagnostics & Backtesting ]
                    └─────────────────────────────────────────┬────────────────────────────────────────┘
                                                              ▼
                                           [ Streamlit Web Presentation Layer (app.py) ]
```

---

### 4. Python Concepts & Libraries Used

| Library / Module | Category | Role in the Project |
| :--- | :--- | :--- |
| **`Streamlit`** | Web Framework | Reactive single-page web architecture, session state management (`st.session_state`), dialog modals, file upload widgets, responsive column grids, and custom CSS injection. |
| **`SQLite3`** | Database Layer | Lightweight relational database management. Implements indexed tables (`expenses`, `budgets`), ACID transactions, and parameterized SQL queries to prevent SQL injection. |
| **`Pandas`** | Data Manipulation | High-performance DataFrame operations, time-series conversion (`pd.to_datetime`), period slicing, multi-column grouping, and budget-vs-actual joins. |
| **`NumPy`** | Numerical Computation | Mathematical computations including sample standard deviation with Bessel's correction ($N-1$), dot-product weighting for time-series projections, and NaN handling. |
| **`NetworkX`** | Graph Theory | Graph generation, bipartite/tripartite node bipartite projection, edge weight computation, Degree Centrality, Betweenness Centrality, and Clauset-Newman-Moore greedy modularity partitioning. |
| **`Plotly`** (`graph_objects` & `express`) | Interactive Data Visualization | Replaces static plotting libraries with dynamic web charts. Features interactive hover cards, click-and-drag panning, zoom, donut charts, grouped/overlay bar charts, and network spring plots. |
| **`Scikit-Learn`** (`sklearn`) | Machine Learning | `TfidfVectorizer` (sublinear TF scaling, n-gram tokenization) and `LogisticRegression` for probabilistic classification of unstructured statement narrations. |
| **`re` (Regular Expressions)** | Text Processing | Complex pattern matching for cleaning bank narrations, removing UPI handles (`@okaxis`, `@ybl`), stripping banking noise tokens (`UPI/`, `POS/`, `ECOM/`), and sanitizing currency formats. |
| **`datetime` & `calendar`** | Time & Date Utilities | ISO date validation, month-range calculations, leap-year handling, and chronological offsetting for rolling window evaluations. |
| **`io` & `pathlib`** | File Handling | Memory-buffered stream processing (`io.StringIO`, `io.BytesIO`) for uploaded statement files and cross-platform filesystem path navigation. |

---

### 5. Mathematical & Algorithmic Foundations

#### 5.1. Z-Score Anomaly Detection
To identify spending outliers without assuming a global spending ceiling, anomalies are evaluated within individual expense categories:

$$\mu = \frac{1}{N} \sum_{i=1}^N x_i, \quad \sigma = \sqrt{\frac{1}{N-1} \sum_{i=1}^N (x_i - \mu)^2}$$

$$z_i = \frac{x_i - \mu}{\sigma}$$

An expense $x_i$ is flagged as an anomaly if $z_i \ge \theta$ (default threshold $\theta = 2.5$). This ensures that a ₹10,000 rent payment is recognized as routine within *Rent & Utilities*, whereas a ₹3,500 cafe purchase in *Food & Dining* is flagged as an outlier.

#### 5.2. 3-Month Weighted Moving Average (WMA) Forecast
To forecast upcoming financial commitments, the model applies higher weighting to more recent months:

$$\hat{M}_{next} = w_1 \cdot M_{-3} + w_2 \cdot M_{-2} + w_3 \cdot M_{-1}$$

Where normalized weights are set to $w = [0.20, 0.30, 0.50]$ ($w_1 + w_2 + w_3 = 1.0$). 
Accuracy is backtested by evaluating historical predictions against actuals using Mean Absolute Error (MAE) and Mean Absolute Percentage Error (MAPE):

$$\text{MAPE} = \frac{1}{n} \sum_{t=1}^n \left| \frac{A_t - F_t}{A_t} \right| \times 100\%$$

#### 5.3. Graph Centrality & Community Detection
- **Degree Centrality**: Measures the direct connectedness of node $v$:
  $$C_D(v) = \frac{\text{deg}(v)}{|V| - 1}$$
- **Betweenness Centrality**: Measures the extent to which a payment channel or merchant sits on shortest paths between other nodes:
  $$C_B(v) = \sum_{s \ne v \ne t} \frac{\sigma_{st}(v)}{\sigma_{st}}$$
- **Greedy Modularity Maximization**: Identifies natural community partitions by optimizing modularity gain $\Delta Q$ when merging node pairs.

---

### 6. Hosting Platform & Cloud Persistence Architecture

#### 6.1. Deployment URL
- **Production URL**: [https://expense3.streamlit.app/](https://expense3.streamlit.app/)
- **Local Development URL**: `http://localhost:8501/`

#### 6.2. Ephemeral Storage Note (Streamlit Community Cloud)
The production application is hosted on **Streamlit Community Cloud (Free Tier)**. It is essential to understand the architectural characteristics of this hosting environment:
- **Serverless Ephemeral Containers**: Streamlit Community Cloud deploys applications inside isolated Docker containers. To optimize cloud resources, inactive applications automatically enter sleep mode.
- **Container Reboots & Filesystem Reset**: When a sleeping container re-awakens or undergoes an automated rebuild, its local file storage resets to the clean state of the underlying GitHub repository. Consequently, any local SQLite database file (`expenses.db`) created during runtime is re-initialized.
- **Resilience Engineering (Automated Sample Seeding)**:
  To prevent new users or evaluators from encountering an empty, broken interface on a fresh container boot, the system incorporates an automatic self-seeding fallback (`load_sample_data_if_empty()` in `src/sample_data.py`). If the database contains zero records, it automatically seeds realistic, calibrated transactions and default category budgets.
- **Full Data Persistence in Production / Local Setup**:
  - When executed **locally** on a personal workstation, the SQLite database resides directly on the host hard drive and is **100% persistent** across app restarts and reboots.
  - On a dedicated paid server, Virtual Private Server (VPS), container environment with a persistent storage volume (e.g., AWS EBS, DigitalOcean Volume), or when connected to an external cloud database (such as PostgreSQL, Supabase, or Neon), data remains permanently persistent.
  - Users can also preserve their transactions at any time via the **One-Click CSV Backup & Export** feature and restore their history instantly using the statement importer.

---

### 7. Conclusion & Future Scope

The **Expense Tracker** project successfully demonstrates how standard Python libraries and foundational data science principles can be combined into an intuitive, production-grade web application. Moving beyond basic manual accounting, the system provides users with automated bank statement parsing, graph-based habit mapping, and statistical anomaly detection.

**Future Scope:**
- Integration with external bank account aggregators via open banking APIs.
- Multi-currency conversion for international expenditures.
- Push notifications and email alerts for budget overruns and unusual transactions.
- Long Short-Term Memory (LSTM) recurrent neural networks for multi-quarter seasonal forecasting.

