"""
app.py - Personal Expense Tracker & Financial Intelligence Dashboard

Card-based responsive interface for financial analytics, budgeting,
transaction ledger management, and spending insights.
"""

from datetime import date, datetime, timedelta
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st

# Ensure repository root is on sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.analytics import (
    compute_kpis,
    format_inr,
    get_budget_vs_actual,
    get_category_breakdown,
    get_monthly_trend,
    get_top_merchants,
    get_weekday_vs_weekend,
    plot_budget_vs_actual,
    plot_category_donut,
    plot_monthly_trend,
    plot_top_merchants,
    plot_weekday_vs_weekend,
)
from src.categorizer import ExpenseClassifier, categorize, categorize_by_rules
from src.db import (
    DEFAULT_BUDGETS,
    PAYMENT_METHODS,
    add_expense,
    bulk_insert_expenses,
    clear_all_expenses,
    delete_expense,
    get_all_categories,
    get_all_merchants,
    get_all_payment_methods,
    get_budgets,
    get_expense_by_id,
    get_expenses,
    init_db,
    set_budget,
    update_expense,
)
from src.graph import (
    build_spending_graph,
    compute_graph_metrics,
    create_network_plotly_figure,
    generate_graph_insights,
)
from src.importer import parse_and_process_statement
from src.insights import (
    backtest_forecast,
    calculate_wma_forecast,
    compare_forecast_to_budget,
    detect_anomalies,
)
from src.sample_data import load_sample_data_if_empty

# ============================================================================
# APP CONFIGURATION & BLOCKY STYLING
# ============================================================================

st.set_page_config(
    page_title="Expense Tracker",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    /* Main Canvas Background */
    .stApp {
        background-color: #f8fafc;
        color: #0f172a;
    }

    /* Sidebar Dark Navigation */
    [data-testid="stSidebar"] {
        background-color: #0f172a !important;
        border-right: 1px solid #1e293b;
    }
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #f8fafc;
    }

    /* 1. Remove the "Go to" header/label from radio */
    [data-testid="stSidebar"] [data-testid="stRadio"] > label,
    [data-testid="stSidebar"] [data-testid="stWidgetLabel"],
    [data-testid="stSidebar"] [data-testid="stRadio"] > div:first-child > label {
        display: none !important;
    }

    /* 2. Full-width consistent Navigation Buttons in Sidebar */
    [data-testid="stSidebar"] [data-testid="stRadio"] {
        width: 100% !important;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] {
        width: 100% !important;
        display: flex !important;
        flex-direction: column !important;
        gap: 8px !important;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] > label {
        width: 100% !important;
        background-color: #1e293b !important;
        border: 1px solid #334155 !important;
        border-radius: 8px !important;
        padding: 10px 14px !important;
        margin: 0 !important;
        display: flex !important;
        align-items: center !important;
        box-sizing: border-box !important;
        cursor: pointer !important;
        transition: all 0.15s ease-in-out !important;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] > label:hover {
        background-color: #273549 !important;
        border-color: #475569 !important;
    }
    /* Active selected navigation item */
    [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] > label:has(input:checked) {
        background-color: #0284c7 !important;
        border-color: #0284c7 !important;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] > label p,
    [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] > label span,
    [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] > label div {
        color: #ffffff !important;
        font-weight: 500 !important;
        font-size: 13.5px !important;
    }

    /* 3. Sidebar Date Input - Dark Slate Box matching Sidebar Navigation */
    [data-testid="stSidebar"] [data-testid="stDateInput"] {
        width: 100% !important;
    }
    [data-testid="stSidebar"] [data-testid="stDateInput"] [data-testid="stDateInputField"],
    [data-testid="stSidebar"] [data-testid="stDateInput"] div[data-baseweb="input"],
    [data-testid="stSidebar"] [data-testid="stDateInput"] div[data-baseweb="base-input"],
    [data-testid="stSidebar"] [data-testid="stDateInput"] [data-range-field] {
        background-color: #1e293b !important;
        border: 1px solid #334155 !important;
        border-radius: 8px !important;
        width: 100% !important;
        color: #f8fafc !important;
    }

    /* All segments, spinbuttons, placeholders and literals inside Sidebar Date Input */
    [data-testid="stSidebar"] [data-testid="stDateInput"] *,
    [data-testid="stSidebar"] [data-testid="stDateInput"] [role="spinbutton"],
    [data-testid="stSidebar"] [data-testid="stDateInput"] .react-aria-DateSegment,
    [data-testid="stSidebar"] [data-testid="stDateInput"] [data-type="literal"],
    [data-testid="stSidebar"] [data-testid="stDateInput"] [data-placeholder],
    [data-testid="stSidebar"] [data-testid="stDateInput"] [data-range-field] *,
    [data-testid="stSidebar"] [data-testid="stDateInput"] input,
    [data-testid="stSidebar"] [data-testid="stDateInput"] span,
    [data-testid="stSidebar"] [data-testid="stDateInput"] div {
        color: #f8fafc !important;
        -webkit-text-fill-color: #f8fafc !important;
        font-weight: 500 !important;
        font-size: 13.5px !important;
    }

    /* Calendar icon inside date picker */
    [data-testid="stSidebar"] [data-testid="stDateInput"] svg {
        fill: #94a3b8 !important;
        color: #94a3b8 !important;
    }

    /* Date Picker Popover Calendar (White card with dark legible dates) */
    div[data-baseweb="popover"],
    div[data-testid="stDateInputCalendar"],
    div[data-testid="stDateInputCalendar"] * {
        background-color: #ffffff !important;
        color: #0f172a !important;
        -webkit-text-fill-color: #0f172a !important;
    }
    div[data-testid="stDateInputCalendar"] [data-selected] {
        background-color: #0284c7 !important;
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
    }

    /* 4. Sidebar Expander & Buttons (Full width, dark block with clear white text) */
    [data-testid="stSidebar"] [data-testid="stExpander"] {
        background-color: #1e293b !important;
        border: 1px solid #334155 !important;
        border-radius: 8px !important;
        width: 100% !important;
    }
    [data-testid="stSidebar"] [data-testid="stExpander"] summary {
        color: #ffffff !important;
        font-weight: 600 !important;
        font-size: 13px !important;
    }
    [data-testid="stSidebar"] [data-testid="stExpander"] summary svg {
        fill: #94a3b8 !important;
    }

    [data-testid="stSidebar"] .stButton {
        width: 100% !important;
    }
    [data-testid="stSidebar"] .stButton > button {
        width: 100% !important;
        background-color: #0f172a !important;
        color: #ffffff !important;
        border: 1px solid #334155 !important;
        border-radius: 8px !important;
        padding: 9px 14px !important;
        font-size: 13px !important;
        font-weight: 600 !important;
        display: flex !important;
        justify-content: center !important;
        align-items: center !important;
        box-sizing: border-box !important;
        transition: all 0.15s ease-in-out !important;
    }
    [data-testid="stSidebar"] .stButton > button:hover {
        background-color: #1e293b !important;
        border-color: #475569 !important;
        color: #ffffff !important;
    }
    [data-testid="stSidebar"] .stButton > button p,
    [data-testid="stSidebar"] .stButton > button span {
        color: #ffffff !important;
    }

    /* Top Page Header */
    .page-title {
        font-size: 26px;
        font-weight: 800;
        color: #0f172a;
        margin-bottom: 2px;
        letter-spacing: -0.5px;
    }
    .page-subtitle {
        font-size: 13.5px;
        color: #64748b;
        margin-bottom: 20px;
    }

    /* Blocky Stat Card Grid */
    .stat-card {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 16px 18px;
        margin-bottom: 12px;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.03);
        min-height: 104px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    .stat-value {
        font-size: 22px;
        font-weight: 800;
        color: #0f172a;
        line-height: 1.15;
        margin-bottom: 4px;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .stat-label {
        font-size: 11.5px;
        font-weight: 600;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.4px;
    }
    .stat-subtext {
        font-size: 11px;
        color: #94a3b8;
        margin-top: 2px;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }

    /* Stat Card Action Buttons (Clean interactive pills below cards) */
    div[class*="st-key-dash_act_btn"] {
        margin-top: -6px !important;
        margin-bottom: 8px !important;
    }
    div[class*="st-key-dash_act_btn"] button {
        background-color: #ffffff !important;
        border: 1px solid #cbd5e1 !important;
        color: #0284c7 !important;
        font-size: 11.5px !important;
        font-weight: 600 !important;
        padding: 4px 8px !important;
        border-radius: 6px !important;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.02) !important;
        transition: all 0.15s ease-in-out !important;
    }
    div[class*="st-key-dash_act_btn"] button:hover {
        background-color: #f0f9ff !important;
        border-color: #0284c7 !important;
        color: #0369a1 !important;
    }

    /* Card Title */
    .card-title {
        font-size: 16px;
        font-weight: 700;
        color: #0f172a;
        margin-top: 0;
        margin-bottom: 12px;
        display: flex;
        align-items: center;
        gap: 6px;
    }

    /* Streamlit Form & Input Styling */
    .stTextInput input, .stNumberInput input, .stSelectbox select {
        border-radius: 6px !important;
        border: 1px solid #cbd5e1 !important;
        background-color: #ffffff !important;
        color: #0f172a !important;
    }

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: transparent;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 6px;
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        padding: 8px 18px;
        font-weight: 600;
        font-size: 13px;
        color: #475569;
    }
    .stTabs [aria-selected="true"] {
        background-color: #0284c7 !important;
        color: #ffffff !important;
        border-color: #0284c7 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================================
# INITIALIZATION (Ephemeral Cloud Persistence Fallback)
# ============================================================================

init_db()
if "db_initialized" not in st.session_state and not st.session_state.get("user_cleared_data", False):
    load_sample_data_if_empty()
    st.session_state["db_initialized"] = True

if "classifier" not in st.session_state:
    st.session_state["classifier"] = ExpenseClassifier()


# ============================================================================
# DIALOGS (2-LAYER CONFIRMATION)
# ============================================================================

@st.dialog("⚠️ Delete All Expenses")
def confirm_delete_all_dialog():
    st.markdown(
        """
        <div style="font-size: 14px; color: #334155; margin-bottom: 12px; line-height: 1.5;">
            Are you sure you want to delete <b>all expenses</b>? This action will permanently remove all transaction records from your database so you can start fresh.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div style="background-color: #fef2f2; border: 1px solid #fecaca; border-radius: 6px; padding: 12px 14px; margin-bottom: 14px; color: #991b1b; font-size: 13px;">
            ⚠️ <b>Two-Layer Confirmation:</b> Please type <b>DELETE</b> in the box below to unlock the permanent deletion button:
        </div>
        """,
        unsafe_allow_html=True,
    )

    typed_confirmation = st.text_input(
        "Type DELETE to confirm",
        placeholder="Type DELETE here...",
        label_visibility="collapsed",
        key="delete_all_confirmation_phrase",
    )

    is_confirmed = (typed_confirmation.strip() == "DELETE")

    btn_col1, btn_col2 = st.columns(2)
    with btn_col1:
        if st.button("Cancel", use_container_width=True):
            st.rerun()
    with btn_col2:
        if st.button(
            "🗑️ Permanently Delete All",
            type="primary",
            disabled=not is_confirmed,
            use_container_width=True,
        ):
            clear_all_expenses()
            st.session_state["user_cleared_data"] = True
            st.session_state["db_initialized"] = True
            st.toast("✅ All expenses deleted! You now have a clean slate.", icon="🧹")
            st.rerun()


def navigate_to(page_name: str, tab_name: Optional[str] = None, search_query: Optional[str] = None) -> None:
    """Safely switch pages via on_click callback before widget instantiation."""
    st.session_state["nav_radio"] = page_name
    if tab_name:
        st.session_state["insights_tab"] = tab_name
    if search_query:
        st.session_state["exp_search"] = search_query


# ============================================================================
# SIDEBAR NAVIGATION & GLOBAL CONTROLS
# ============================================================================

st.sidebar.markdown(
    """
    <div style="padding: 10px 0 16px 0;">
        <span style="font-size: 11px; font-weight: 700; color: #94a3b8; letter-spacing: 1.2px; text-transform: uppercase;">
            NAVIGATION
        </span>
    </div>
    """,
    unsafe_allow_html=True,
)

if "nav_radio" not in st.session_state:
    st.session_state["nav_radio"] = "📊 Dashboard"

page = st.sidebar.radio(
    "Go to",
    ["📊 Dashboard", "💳 Expenses (CRUD)", "🕸️ Spending Graph", "💡 Smart Insights"],
    label_visibility="collapsed",
    key="nav_radio",
)

st.sidebar.markdown("<hr style='border-color: #334155; margin: 20px 0 16px 0;'>", unsafe_allow_html=True)
st.sidebar.markdown(
    """
    <span style="font-size: 11px; font-weight: 700; color: #94a3b8; letter-spacing: 1.2px; text-transform: uppercase;">
        DATE FILTER
    </span>
    """,
    unsafe_allow_html=True,
)

all_expenses_df = get_expenses()
if not all_expenses_df.empty:
    min_dt = pd.to_datetime(all_expenses_df["date"]).min().date()
    max_dt = pd.to_datetime(all_expenses_df["date"]).max().date()
else:
    min_dt = date.today() - timedelta(days=180)
    max_dt = date.today()

date_range = st.sidebar.date_input(
    "Filter by Date Range",
    value=(min_dt, max_dt),
    min_value=min_dt - timedelta(days=365),
    max_value=max_dt + timedelta(days=30),
    label_visibility="collapsed",
)

start_date_str = None
end_date_str = None
if isinstance(date_range, tuple) and len(date_range) == 2:
    start_date_str = date_range[0].strftime("%Y-%m-%d")
    end_date_str = date_range[1].strftime("%Y-%m-%d")

st.sidebar.markdown("<hr style='border-color: #334155; margin: 20px 0 16px 0;'>", unsafe_allow_html=True)

with st.sidebar.expander("🛠️ Database Controls"):
    if st.button("🔄 Reset to Sample Data", use_container_width=True):
        clear_all_expenses()
        st.session_state["user_cleared_data"] = False
        load_sample_data_if_empty()
        st.success("Sample dataset reloaded!")
        st.rerun()


# ============================================================================
# PAGE 1: DASHBOARD (BLOCKY METRIC CARDS + CORE ANALYTICS)
# ============================================================================

if page == "📊 Dashboard":
    st.markdown('<div class="page-title">Expense tracker</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-subtitle">Schedule-based analytics & financial overview -- track spending, budgets, and habits.</div>',
        unsafe_allow_html=True,
    )

    filtered_df = get_expenses(start_date=start_date_str, end_date=end_date_str)
    budgets_df = get_budgets()

    if filtered_df.empty:
        st.warning("No transactions found for the selected date range. Adjust the sidebar date filter.")
    else:
        kpis = compute_kpis(filtered_df)
        forecast_res = calculate_wma_forecast(filtered_df)
        anomalies_df = detect_anomalies(filtered_df, threshold=2.5)
        top_cats = get_category_breakdown(filtered_df)
        top_cat_name = top_cats.iloc[0]["category"] if not top_cats.empty else "N/A"
        top_cat_amt = format_inr(top_cats.iloc[0]["amount"]) if not top_cats.empty else "₹0"

        comp_pay = filtered_df["payment_method"].value_counts()
        dominant_pm = comp_pay.index[0] if not comp_pay.empty else "N/A"
        dominant_pct = f"{round(comp_pay.iloc[0] / len(filtered_df) * 100)}%" if not comp_pay.empty else "0%"

        # ====================================================================
        # 8 Metric Cards Grid
        # ====================================================================
        r1_c1, r1_c2, r1_c3, r1_c4 = st.columns(4)
        with r1_c1:
            st.markdown(
                f"""
                <div class="stat-card">
                    <div class="stat-value">{format_inr(kpis['total_this_month'])}</div>
                    <div class="stat-label">Total Outflow (Month)</div>
                    <div class="stat-subtext">Target: {kpis['target_month']} ({kpis['mom_change_pct']:+.1f}% MoM)</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            with st.container(key="dash_act_btn_r1c1"):
                st.button(
                    "📋 Open Ledger →",
                    key="btn_goto_crud_r1c1",
                    on_click=navigate_to,
                    args=("💳 Expenses (CRUD)",),
                    use_container_width=True,
                )

        with r1_c2:
            st.markdown(
                f"""
                <div class="stat-card">
                    <div class="stat-value">{format_inr(kpis['avg_daily_spend'])}</div>
                    <div class="stat-label">Daily Average</div>
                    <div class="stat-subtext">Calculated across active days</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            with st.container(key="dash_act_btn_r1c2"):
                st.button(
                    "📅 View Habits →",
                    key="btn_goto_habits_r1c2",
                    on_click=navigate_to,
                    args=("💳 Expenses (CRUD)",),
                    use_container_width=True,
                )

        with r1_c3:
            st.markdown(
                f"""
                <div class="stat-card">
                    <div class="stat-value">{kpis['total_transactions']}</div>
                    <div class="stat-label">Total Transactions</div>
                    <div class="stat-subtext">{format_inr(kpis['total_spend_all_time'])} total volume</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            with st.container(key="dash_act_btn_r1c3"):
                st.button(
                    "📋 View Transactions →",
                    key="btn_goto_crud_r1c3",
                    on_click=navigate_to,
                    args=("💳 Expenses (CRUD)",),
                    use_container_width=True,
                )

        with r1_c4:
            st.markdown(
                f"""
                <div class="stat-card">
                    <div class="stat-value">{dominant_pm}</div>
                    <div class="stat-label">Dominant Payment Method</div>
                    <div class="stat-subtext">Used in {dominant_pct} of all expenses</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            with st.container(key="dash_act_btn_r1c4"):
                st.button(
                    "🕸️ Centrality Graph →",
                    key="btn_goto_graph_r1c4",
                    on_click=navigate_to,
                    args=("🕸️ Spending Graph",),
                    use_container_width=True,
                )

        r2_c1, r2_c2, r2_c3, r2_c4 = st.columns(4)
        with r2_c1:
            st.markdown(
                f"""
                <div class="stat-card">
                    <div class="stat-value" title="{top_cat_name}">{top_cat_name}</div>
                    <div class="stat-label">Top Spending Category</div>
                    <div class="stat-subtext">{top_cat_amt} spent</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            with st.container(key="dash_act_btn_r2c1"):
                st.button(
                    "🕸️ Category Graph →",
                    key="btn_goto_graph_r2c1",
                    on_click=navigate_to,
                    args=("🕸️ Spending Graph",),
                    use_container_width=True,
                )

        with r2_c2:
            st.markdown(
                f"""
                <div class="stat-card">
                    <div class="stat-value">{format_inr(kpis['biggest_expense_amount'])}</div>
                    <div class="stat-label">Largest Single Expense</div>
                    <div class="stat-subtext">{kpis['biggest_expense_merchant'][:25]}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            with st.container(key="dash_act_btn_r2c2"):
                st.button(
                    "🔍 Search Big Expense →",
                    key="btn_goto_search_r2c2",
                    on_click=navigate_to,
                    args=("💳 Expenses (CRUD)", None, kpis["biggest_expense_merchant"]),
                    use_container_width=True,
                )

        with r2_c3:
            st.markdown(
                f"""
                <div class="stat-card">
                    <div class="stat-value" style="color: {'#dc2626' if len(anomalies_df) > 0 else '#16a34a'};">
                        {len(anomalies_df)} Flagged
                    </div>
                    <div class="stat-label">Anomalies Detected</div>
                    <div class="stat-subtext">Exceeding 2.5σ category baseline</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            with st.container(key="dash_act_btn_r2c3"):
                st.button(
                    f"🚨 Inspect {len(anomalies_df)} Anomalies →",
                    key="btn_goto_anom",
                    on_click=navigate_to,
                    args=("💡 Smart Insights", "anomalies"),
                    use_container_width=True,
                )

        with r2_c4:
            st.markdown(
                f"""
                <div class="stat-card">
                    <div class="stat-value" style="color: #0284c7;">{format_inr(forecast_res['total_forecast'])}</div>
                    <div class="stat-label">Next Month Forecast</div>
                    <div class="stat-subtext">Projected for {forecast_res['forecast_month']} (3M WMA)</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            with st.container(key="dash_act_btn_r2c4"):
                st.button(
                    "📈 Forecast Model Breakdown →",
                    key="btn_goto_fc",
                    on_click=navigate_to,
                    args=("💡 Smart Insights", "forecast"),
                    use_container_width=True,
                )

        # ====================================================================
        # ROW 1 OF VISUALS: Trend Line & Category Donut
        # ====================================================================
        ch_c1, ch_c2 = st.columns([5, 4])
        with ch_c1:
            with st.container(border=True):
                head_col1, head_col2 = st.columns([3, 2])
                with head_col1:
                    st.markdown('<div class="card-title">📈 Monthly Outflow Trend</div>', unsafe_allow_html=True)
                with head_col2:
                    trend_span = st.segmented_control(
                        "Trend Timespan",
                        options=["3M", "6M", "12M", "All"],
                        default="All",
                        label_visibility="collapsed",
                        key="trend_timespan_selector",
                    )

                trend_input_df = filtered_df.copy()
                if trend_span and trend_span != "All" and not trend_input_df.empty:
                    span_months = {"3M": 3, "6M": 6, "12M": 12}.get(trend_span, 6)
                    max_d = pd.to_datetime(trend_input_df["date"]).max()
                    cutoff_d = max_d - pd.DateOffset(months=span_months)
                    trend_input_df = trend_input_df[pd.to_datetime(trend_input_df["date"]) >= cutoff_d]

                trend_df = get_monthly_trend(trend_input_df)
                fig_trend = plot_monthly_trend(trend_df)
                fig_trend.update_layout(height=340, margin=dict(l=20, r=20, t=30, b=20))
                st.plotly_chart(fig_trend, use_container_width=True)

        with ch_c2:
            with st.container(border=True):
                st.markdown('<div class="card-title">🍩 Category Breakdown</div>', unsafe_allow_html=True)
                fig_donut = plot_category_donut(top_cats, title="")
                fig_donut.update_layout(height=340, margin=dict(l=10, r=10, t=20, b=20))
                st.plotly_chart(fig_donut, use_container_width=True)

        # ====================================================================
        # ROW 2 OF VISUALS: Top Merchants & Weekday vs Weekend
        # ====================================================================
        ch_c3, ch_c4 = st.columns([5, 4])
        with ch_c3:
            with st.container(border=True):
                m_head1, m_head2 = st.columns([3, 2])
                with m_head1:
                    st.markdown('<div class="card-title">🏪 Top 10 Merchants by Spend</div>', unsafe_allow_html=True)
                with m_head2:
                    merch_span = st.segmented_control(
                        "Merchant Timespan",
                        options=["3M", "6M", "All"],
                        default="All",
                        label_visibility="collapsed",
                        key="merch_timespan_selector",
                    )

                merch_input_df = filtered_df.copy()
                if merch_span and merch_span != "All" and not merch_input_df.empty:
                    m_months = {"3M": 3, "6M": 6}.get(merch_span, 6)
                    max_d = pd.to_datetime(merch_input_df["date"]).max()
                    cutoff_d = max_d - pd.DateOffset(months=m_months)
                    merch_input_df = merch_input_df[pd.to_datetime(merch_input_df["date"]) >= cutoff_d]

                top_merch_df = get_top_merchants(merch_input_df, n=10)
                fig_merchants = plot_top_merchants(top_merch_df)
                fig_merchants.update_layout(height=340, margin=dict(l=20, r=20, t=30, b=20))
                st.plotly_chart(fig_merchants, use_container_width=True)

        with ch_c4:
            with st.container(border=True):
                st.markdown('<div class="card-title">📅 Weekday vs Weekend Habits</div>', unsafe_allow_html=True)
                weekend_comp = get_weekday_vs_weekend(filtered_df)
                fig_weekend = plot_weekday_vs_weekend(weekend_comp)
                fig_weekend.update_layout(height=340, margin=dict(l=20, r=20, t=30, b=20))
                st.plotly_chart(fig_weekend, use_container_width=True)

        # ====================================================================
        # ROW 3: Budget vs Actual Utilization
        # ====================================================================
        with st.container(border=True):
            b_head1, b_head2 = st.columns([3, 2])
            with b_head1:
                st.markdown('<div class="card-title">🎯 Budget vs Actual Spending</div>', unsafe_allow_html=True)
            with b_head2:
                eval_months = sorted(list(pd.to_datetime(filtered_df["date"]).dt.strftime("%Y-%m").unique()), reverse=True)
                sel_b_month = st.selectbox("Select Evaluation Month", eval_months, key="dash_b_month", label_visibility="collapsed")

            budget_comp_df = get_budget_vs_actual(filtered_df, budgets_df, month=sel_b_month)
            if not budget_comp_df.empty:
                tot_budget = float(budget_comp_df["monthly_limit"].sum())
                tot_spent = float(budget_comp_df["spent"].sum())
                overall_pct = (tot_spent / tot_budget * 100) if tot_budget > 0 else 0
                over_count = int((budget_comp_df["status"] == "Over Budget").sum())

                bk1, bk2, bk3, bk4 = st.columns(4)
                bk1.metric("Month Target", format_inr(tot_budget))
                bk2.metric("Total Spent", format_inr(tot_spent))
                bk3.metric("Budget Utilization", f"{overall_pct:.1f}%")
                bk4.metric("Over-Budget Categories", f"{over_count} Categories", delta=f"{over_count} Exceeded" if over_count > 0 else "All On Track", delta_color="inverse")

                fig_budget = plot_budget_vs_actual(budget_comp_df)
                fig_budget.update_layout(height=360, margin=dict(l=20, r=20, t=30, b=20))
                st.plotly_chart(fig_budget, use_container_width=True)

                with st.expander("⚙️ Customize Monthly Category Budgets"):
                    st.caption("Adjust your category targets to reflect your genuine personal budget. Values are saved to your database.")
                    with st.form("custom_budget_form"):
                        cats = sorted(budgets_df["category"].tolist())
                        new_limits = {}
                        col_list = st.columns(3)
                        for idx, cat_name in enumerate(cats):
                            curr_lim = float(budgets_df.loc[budgets_df["category"] == cat_name, "monthly_limit"].iloc[0])
                            with col_list[idx % 3]:
                                new_limits[cat_name] = st.number_input(
                                    f"{cat_name} (₹)",
                                    min_value=0.0,
                                    value=curr_lim,
                                    step=500.0,
                                    key=f"b_input_{cat_name}",
                                )
                        if st.form_submit_button("💾 Save Budget Targets", type="primary", use_container_width=True):
                            for c_name, lim_val in new_limits.items():
                                set_budget(c_name, lim_val)
                            st.toast("Updated monthly category budgets successfully!", icon="🎯")
                            st.rerun()


# ============================================================================
# PAGE 2: EXPENSES (SIMPLIFIED & INTUITIVE CRUD)
# ============================================================================

elif page == "💳 Expenses (CRUD)":
    st.markdown('<div class="page-title">Expense Manager</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-subtitle">Add new expenses, edit or delete records, search history, and import bank statements.</div>',
        unsafe_allow_html=True,
    )

    all_current_df = get_expenses()

    # ========================================================================
    # TOP SPLIT: ADD EXPENSE & EDIT/DELETE IN CLEAR BLOCK CARDS
    # ========================================================================
    crud_c1, crud_c2 = st.columns([1, 1])

    # BLOCK 1: ADD EXPENSE
    with crud_c1:
        with st.container(border=True):
            st.markdown('<div class="card-title">➕ Quick Add Expense</div>', unsafe_allow_html=True)

            with st.form("quick_add_form", clear_on_submit=True):
                f_col1, f_col2 = st.columns(2)
                with f_col1:
                    add_dt = st.date_input("Date", value=date.today())
                    add_amt = st.number_input("Amount (₹)", min_value=1.0, value=250.0, step=25.0)
                    add_merchant = st.text_input("Merchant / Store", placeholder="e.g. Swiggy, Uber, DMart")
                with f_col2:
                    guessed_cat = categorize_by_rules(add_merchant) or "Food & Dining"
                    all_cats = get_all_categories()
                    cat_idx = all_cats.index(guessed_cat) if guessed_cat in all_cats else 0
                    add_category = st.selectbox("Category", all_cats, index=cat_idx)
                    add_pm = st.selectbox("Payment Method", PAYMENT_METHODS)
                    add_note = st.text_input("Optional Note", placeholder="e.g. Dinner with team")

                submit_add = st.form_submit_button("➕ Save Expense", use_container_width=True, type="primary")
                if submit_add:
                    if not add_merchant.strip():
                        st.error("Please provide a merchant name.")
                    else:
                        new_id = add_expense(
                            date=add_dt.strftime("%Y-%m-%d"),
                            amount=add_amt,
                            category=add_category,
                            merchant=add_merchant,
                            payment_method=add_pm,
                            note=add_note,
                        )
                        st.success(f"Recorded expense #{new_id} ({format_inr(add_amt)} at {add_merchant})!")
                        st.rerun()

    # BLOCK 2: EDIT OR DELETE AN EXPENSE
    with crud_c2:
        with st.container(border=True):
            st.markdown('<div class="card-title">✏️ Edit or Delete Expense</div>', unsafe_allow_html=True)

            if all_current_df.empty:
                st.info("No expenses available to edit.")
            else:
                # Build an intuitive dropdown list of recent expenses
                expense_options = {}
                for _, r in all_current_df.head(40).iterrows():
                    label = f"#{r['id']} | {r['date']} | {r['merchant']} | {format_inr(r['amount'])} ({r['category']})"
                    expense_options[label] = int(r["id"])

                selected_label = st.selectbox(
                    "Select Transaction to Edit or Remove",
                    options=list(expense_options.keys()),
                    help="Choose from the 40 most recent expenses",
                )
                selected_exp_id = expense_options[selected_label]
                row_to_edit = get_expense_by_id(selected_exp_id)

                if row_to_edit:
                    with st.form("edit_expense_form"):
                        ed_c1, ed_c2 = st.columns(2)
                        with ed_c1:
                            ed_dt = st.date_input(
                                "Date",
                                value=datetime.strptime(row_to_edit["date"], "%Y-%m-%d").date(),
                            )
                            ed_amt = st.number_input(
                                "Amount (₹)", min_value=1.0, value=float(row_to_edit["amount"]), step=25.0
                            )
                            ed_merchant = st.text_input("Merchant", value=row_to_edit["merchant"])
                        with ed_c2:
                            all_cats = get_all_categories()
                            c_idx = all_cats.index(row_to_edit["category"]) if row_to_edit["category"] in all_cats else 0
                            ed_category = st.selectbox("Category", all_cats, index=c_idx)

                            all_pms = PAYMENT_METHODS
                            p_idx = all_pms.index(row_to_edit["payment_method"]) if row_to_edit["payment_method"] in all_pms else 0
                            ed_pm = st.selectbox("Payment Method", all_pms, index=p_idx)
                            ed_note = st.text_input("Note", value=row_to_edit.get("note", ""))

                        b_col1, b_col2 = st.columns(2)
                        with b_col1:
                            save_ed = st.form_submit_button("💾 Update Expense", use_container_width=True)
                        with b_col2:
                            del_btn = st.form_submit_button("🗑️ Delete Expense", use_container_width=True)

                        if save_ed:
                            update_expense(
                                expense_id=selected_exp_id,
                                date=ed_dt.strftime("%Y-%m-%d"),
                                amount=ed_amt,
                                category=ed_category,
                                merchant=ed_merchant,
                                payment_method=ed_pm,
                                note=ed_note,
                            )
                            st.success(f"Updated expense #{selected_exp_id}!")
                            st.rerun()

                        if del_btn:
                            delete_expense(selected_exp_id)
                            st.warning(f"Deleted expense #{selected_exp_id}.")
                            st.rerun()

                # --- DANGER ZONE: 2-LAYER DELETE ALL EXPENSES ---
                st.markdown("<hr style='margin: 16px 0 12px 0; border-color: #f1f5f9;'>", unsafe_allow_html=True)
                del_all_col1, del_all_col2 = st.columns([3, 2])
                with del_all_col1:
                    st.markdown(
                        "<div style='font-size: 13px; font-weight: 600; color: #991b1b;'>🚨 Reset / Wipe All Data</div>"
                        "<div style='font-size: 11.5px; color: #64748b;'>Want a fresh start? Clear all sample records to track real expenses.</div>",
                        unsafe_allow_html=True,
                    )
                with del_all_col2:
                    if st.button(
                        "🗑️ Delete All Expenses",
                        type="secondary",
                        use_container_width=True,
                        disabled=(all_current_df.empty),
                    ):
                        confirm_delete_all_dialog()

    # ========================================================================
    # TABLE VIEW & INSTANT FILTERING
    # ========================================================================
    with st.container(border=True):
        st.markdown('<div class="card-title">📋 All Expenses & Search</div>', unsafe_allow_html=True)

        srch_c1, srch_c2, srch_c3 = st.columns([3, 2, 2])
        with srch_c1:
            tbl_search = st.text_input("🔍 Search Merchant or Note", placeholder="Type to search...", key="exp_search")
        with srch_c2:
            tbl_cat = st.selectbox("Filter by Category", ["All Categories"] + get_all_categories(), key="exp_cat_filter")
        with srch_c3:
            tbl_pm = st.selectbox("Filter by Payment Method", ["All Methods"] + get_all_payment_methods(), key="exp_pm_filter")

        c_filt = None if tbl_cat == "All Categories" else tbl_cat
        p_filt = None if tbl_pm == "All Methods" else tbl_pm
        s_filt = tbl_search.strip() if tbl_search.strip() else None

        table_expenses = get_expenses(
            start_date=start_date_str,
            end_date=end_date_str,
            category=c_filt,
            payment_method=p_filt,
            search=s_filt,
        )

        st.markdown(
            f"**Found {len(table_expenses)} expenses** | Total Volume: **{format_inr(table_expenses['amount'].sum() if not table_expenses.empty else 0)}**"
        )

        if not table_expenses.empty:
            display_tbl = table_expenses.copy()
            display_tbl["Amount (₹)"] = display_tbl["amount"].apply(format_inr)
            st.dataframe(
                display_tbl[["id", "date", "merchant", "category", "Amount (₹)", "payment_method", "note"]].rename(
                    columns={
                        "id": "ID",
                        "date": "Date",
                        "merchant": "Merchant",
                        "category": "Category",
                        "payment_method": "Payment Method",
                        "note": "Note",
                    }
                ),
                use_container_width=True,
                height=340,
            )
        else:
            st.info("No expenses found matching the current search criteria.")

    # ========================================================================
    # BOTTOM SPLIT: BANK STATEMENT CSV IMPORT & BACKUP EXPORT
    # ========================================================================
    io_c1, io_c2 = st.columns([3, 2])

    with io_c1:
        with st.container(border=True):
            st.markdown('<div class="card-title">📥 Bank & UPI Statement Import</div>', unsafe_allow_html=True)
            st.write("Upload PhonePe, GPay, HDFC, SBI, ICICI or UPI statements (CSV or JSON). Skips credit deposits and duplicates automatically.")

            up_file = st.file_uploader("Upload Statement (CSV or JSON)", type=["csv", "json"], label_visibility="collapsed")
            use_demo_csv = st.button("Use Sample Bank Statement for Demo", type="secondary")

            file_to_parse = None
            if up_file is not None:
                file_to_parse = up_file
            elif use_demo_csv:
                demo_path = BASE_DIR / "data" / "sample_bank_statement.csv"
                if demo_path.exists():
                    file_to_parse = str(demo_path)

            if file_to_parse:
                with st.spinner("Cleaning narrations & detecting duplicates..."):
                    parse_res = parse_and_process_statement(
                        file_or_path=file_to_parse,
                        classifier=st.session_state["classifier"],
                    )

                if parse_res.get("error"):
                    st.error(parse_res["error"])
                else:
                    sm_c1, sm_c2, sm_c3 = st.columns(3)
                    sm_c1.metric("Rows Read", parse_res["read"])
                    sm_c2.metric("Duplicates Skipped", parse_res["duplicates_skipped"])
                    sm_c3.metric("Valid to Import", parse_res["imported"])

                    cleaned_df = parse_res["cleaned_df"]
                    if not cleaned_df.empty:
                        st.dataframe(cleaned_df[["date", "merchant", "category", "amount"]].head(6), use_container_width=True)
                        if st.button("🚀 Confirm & Import into Database", type="primary", use_container_width=True):
                            ins_cnt = bulk_insert_expenses(cleaned_df)
                            st.success(f"Imported {ins_cnt} new expenses successfully!")
                            st.rerun()
                    else:
                        st.warning("All debit entries in this file are duplicates or credits.")

    with io_c2:
        with st.container(border=True):
            st.markdown('<div class="card-title">📤 Backup & Export</div>', unsafe_allow_html=True)
            st.write("Export your entire database to a clean, standardized CSV backup file anytime.")
            full_backup = get_expenses()
            if not full_backup.empty:
                backup_csv = full_backup.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="📥 Download Expenses CSV Backup",
                    data=backup_csv,
                    file_name=f"expenses_backup_{date.today().strftime('%Y%m%d')}.csv",
                    mime="text/csv",
                    use_container_width=True,
                    type="primary",
                )
            else:
                st.info("No records to export.")


# ============================================================================
# PAGE 3: SPENDING GRAPH (NETWORKX ANALYSIS)
# ============================================================================

elif page == "🕸️ Spending Graph":
    st.markdown('<div class="page-title">Spending Graph & Network Analytics</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-subtitle">NetworkX tripartite graph analyzing how Merchants, Categories, and Payment Channels interact.</div>',
        unsafe_allow_html=True,
    )

    graph_expenses = get_expenses(start_date=start_date_str, end_date=end_date_str)

    if graph_expenses.empty:
        st.warning("No expenses available for the current date range.")
    else:
        # Controls Card
        with st.container(border=True):
            gc_col1, gc_col2 = st.columns([1, 1])
            with gc_col1:
                all_m = ["All Months"] + sorted(list(pd.to_datetime(graph_expenses["date"]).dt.strftime("%Y-%m").unique()), reverse=True)
                sel_graph_m = st.selectbox("Filter Graph by Month", all_m)
            with gc_col2:
                color_mode = st.radio("Node Coloring Scheme", ["Node Type", "Community Cluster (Habits)"], horizontal=True)
                color_param = "community" if "Community" in color_mode else "node_type"

        if sel_graph_m != "All Months":
            graph_expenses["m_str"] = pd.to_datetime(graph_expenses["date"]).dt.strftime("%Y-%m")
            graph_expenses = graph_expenses[graph_expenses["m_str"] == sel_graph_m]

        with st.spinner("Computing centralities & community partitions..."):
            G = build_spending_graph(graph_expenses)
            metrics, communities = compute_graph_metrics(G)

        # Plain-English Insights Grid
        insights = generate_graph_insights(G, metrics, communities, graph_expenses)
        ins_cols = st.columns(len(insights))
        for col, ins in zip(ins_cols, insights):
            with col:
                st.markdown(
                    f"""
                    <div class="stat-card">
                        <div class="stat-label">{ins['title']}</div>
                        <div style="font-size: 13.5px; color: #1e293b; margin-top: 4px;">{ins['text']}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        # Interactive Graph Plot
        with st.container(border=True):
            fig_network = create_network_plotly_figure(G, metrics, color_by=color_param)
            fig_network.update_layout(margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(fig_network, use_container_width=True)

        # Metrics Breakdown
        with st.expander("📊 View Centrality & Modularity Metrics Table"):
            metrics_df = pd.DataFrame(list(metrics.values()))
            if not metrics_df.empty:
                metrics_df["Volume (₹)"] = metrics_df["total_spend"].apply(format_inr)
                st.dataframe(
                    metrics_df[["node", "node_type", "Volume (₹)", "degree", "degree_centrality", "betweenness_centrality", "community_name"]].rename(
                        columns={
                            "node": "Node Name",
                            "node_type": "Type",
                            "degree": "Degree",
                            "degree_centrality": "Degree Centrality",
                            "betweenness_centrality": "Betweenness Centrality",
                            "community_name": "Cluster",
                        }
                    ).sort_values(by="Betweenness Centrality", ascending=False),
                    use_container_width=True,
                )


# ============================================================================
# PAGE 4: SMART INSIGHTS (ANOMALY DETECTION & FORECASTING)
# ============================================================================

elif page == "💡 Smart Insights":
    st.markdown('<div class="page-title">Smart Financial Insights</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-subtitle">Z-Score anomaly detection, 3-Month Moving Average forecasting, and auto-categorization.</div>',
        unsafe_allow_html=True,
    )

    all_exp_data = get_expenses()
    budgets_data = get_budgets()

    tab_titles = ["🚨 Anomaly Detection", "📈 3-Month WMA Forecast", "🤖 Categorization Lab"]
    default_tab = None
    if st.session_state.get("insights_tab") == "forecast":
        default_tab = "📈 3-Month WMA Forecast"
    elif st.session_state.get("insights_tab") == "anomalies":
        default_tab = "🚨 Anomaly Detection"

    if default_tab:
        tab_anom, tab_fc, tab_ml = st.tabs(tab_titles, default=default_tab)
        st.session_state["insights_tab"] = None
    else:
        tab_anom, tab_fc, tab_ml = st.tabs(tab_titles)

    # TAB 1: ANOMALY DETECTION
    with tab_anom:
        with st.container(border=True):
            st.markdown('<div class="card-title">🚨 Z-Score Outlier Detection</div>', unsafe_allow_html=True)
            st.write("Identifies unusually large transactions in each category by computing standardized distance: $z = (x - \\mu) / \\sigma$.")

            thresh_val = st.slider("Anomaly Sensitivity (σ Threshold)", min_value=1.5, max_value=4.0, value=2.5, step=0.1)
            anomalies_found = detect_anomalies(all_exp_data, threshold=thresh_val)

            if not anomalies_found.empty:
                st.warning(f"Discovered **{len(anomalies_found)} anomalous transactions** exceeding {thresh_val}σ.")
                anom_disp = anomalies_found.copy()
                anom_disp["Amount (₹)"] = anom_disp["amount"].apply(format_inr)
                anom_disp["Category Avg"] = anom_disp["category_mean"].apply(format_inr)
                st.dataframe(
                    anom_disp[["date", "merchant", "category", "Amount (₹)", "Category Avg", "z_score", "reason"]].rename(
                        columns={"date": "Date", "merchant": "Merchant", "category": "Category", "z_score": "Z-Score", "reason": "Explanation"}
                    ),
                    use_container_width=True,
                )
            else:
                st.success(f"No statistical anomalies found at threshold {thresh_val}σ.")

    # TAB 2: FORECASTING
    with tab_fc:
        with st.container(border=True):
            st.markdown('<div class="card-title">📈 3-Month Weighted Moving Average Forecast</div>', unsafe_allow_html=True)
            st.write("Predicts upcoming monthly spending with recency weights: $\\hat{M}_{next} = 0.2 \\cdot M_{-3} + 0.3 \\cdot M_{-2} + 0.5 \\cdot M_{-1}$.")

            fc_data = calculate_wma_forecast(all_exp_data)
            f1, f2, f3 = st.columns(3)
            f1.metric("Forecast Target Month", fc_data["forecast_month"])
            f2.metric("Projected Total Spend", format_inr(fc_data["total_forecast"]))
            f3.metric("Weights Applied", "20% (M-3), 30% (M-2), 50% (M-1)")

            tr_df = get_monthly_trend(all_exp_data)
            fc_ext_df = pd.DataFrame([{"month": fc_data["forecast_month"], "forecast_amount": fc_data["total_forecast"]}])
            fig_trend_fc = plot_monthly_trend(tr_df, forecast_df=fc_ext_df)
            st.plotly_chart(fig_trend_fc, use_container_width=True)

            st.markdown("#### Projected Category Spend vs Monthly Budgets")
            cat_fc_df = fc_data["category_forecasts"]
            budget_proj = compare_forecast_to_budget(cat_fc_df, budgets_data)
            if not budget_proj.empty:
                proj_disp = budget_proj.copy()
                proj_disp["Forecast Spend"] = proj_disp["forecast_amount"].apply(format_inr)
                proj_disp["Budget Limit"] = proj_disp["monthly_limit"].apply(format_inr)
                proj_disp["Variance"] = proj_disp["projected_variance"].apply(format_inr)
                st.dataframe(
                    proj_disp[["category", "Forecast Spend", "Budget Limit", "Variance", "status"]].rename(
                        columns={"category": "Category", "status": "Projected Status"}
                    ),
                    use_container_width=True,
                )

            with st.expander("🧪 Backtesting Accuracy Evaluation"):
                bt = backtest_forecast(all_exp_data)
                if "error" not in bt:
                    st.write(
                        f"Backtested on month **{bt['test_month']}**: Actual was **{format_inr(bt['actual_amount'])}** vs Model **{format_inr(bt['predicted_amount'])}**."
                    )
                    b1, b2 = st.columns(2)
                    b1.metric("Mean Absolute Error (MAE)", format_inr(bt["abs_error"]))
                    b2.metric("MAPE Percentage Error", f"{bt['mape_pct']}%")

    # TAB 3: CATEGORIZATION LAB
    with tab_ml:
        with st.container(border=True):
            st.markdown('<div class="card-title">🤖 Rule-Based & ML Categorizer Test Bench</div>', unsafe_allow_html=True)
            st.write("Test keyword rules and train an optional Scikit-Learn TF-IDF + Logistic Regression pipeline.")

            test_query = st.text_input("Enter narration to categorize", value="UPI/SWIGGY/ORDER9981@ybl")
            enable_ml = st.toggle("Enable Machine Learning Fallback (TF-IDF + Logistic Regression)", value=False)

            clf = st.session_state["classifier"]
            if enable_ml and not clf.is_trained:
                with st.spinner("Training TF-IDF + Logistic Regression model..."):
                    ok = clf.train(all_exp_data)
                    if ok:
                        st.success(f"Classifier trained with {clf.accuracy_score * 100:.1f}% cross-validation accuracy!")

            rule_out = categorize_by_rules(test_query)
            final_cat, src = categorize(test_query, classifier=clf, use_ml=enable_ml)

            t1, t2, t3 = st.columns(3)
            t1.metric("Rule Result", rule_out or "None")
            t2.metric("ML Result", clf.predict(test_query) if clf.is_trained else "Not Trained")
            t3.metric("Resolved Category", final_cat, help=f"Resolved via {src}")
