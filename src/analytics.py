"""
analytics.py - Aggregations, KPIs, and Plotly Visualizations

Provides data processing functions and interactive Plotly figures for:
1. Executive KPI Cards (Total spend, MoM % change, daily average, top expense).
2. Monthly Spending Trends (interactive line chart with optional forecast extension).
3. Category Distribution (interactive donut chart).
4. Top 10 Merchants by Total Spend (horizontal bar chart).
5. Weekday vs Weekend Spending Behavior (totals and daily averages).
6. Budget vs Actual Comparison (progress bars and color-coded status charts).
"""

from calendar import monthrange
from datetime import datetime
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Ensure root in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))


def format_inr(number: Optional[Union[float, int]], decimals: int = 2) -> str:
    """
    Format numerical amounts according to the Indian numbering system.
    Example: 1234567.89 -> '₹12,34,567.89'
    """
    if number is None or pd.isna(number):
        return "₹0.00" if decimals > 0 else "₹0"

    is_neg = number < 0
    num_abs = abs(number)
    parts = f"{num_abs:.{decimals}f}".split(".")
    int_str = parts[0]
    dec_str = f".{parts[1]}" if decimals > 0 else ""

    if len(int_str) > 3:
        last_three = int_str[-3:]
        remaining = int_str[:-3]
        chunks = []
        while len(remaining) > 2:
            chunks.insert(0, remaining[-2:])
            remaining = remaining[:-2]
        if remaining:
            chunks.insert(0, remaining)
        formatted_int = ",".join(chunks) + "," + last_three
    else:
        formatted_int = int_str

    prefix = "-₹" if is_neg else "₹"
    return f"{prefix}{formatted_int}{dec_str}"


def compute_kpis(df: pd.DataFrame, target_month: Optional[str] = None) -> Dict[str, Any]:
    """
    Calculate top-level financial KPIs for the dashboard.

    Args:
        df: Filtered expenses DataFrame.
        target_month: Target month in 'YYYY-MM' format (defaults to the latest month in df).

    Returns:
        Dictionary of KPI metrics with current spend, last month spend, % delta,
        daily average spend, and largest single transaction.
    """
    kpis = {
        "target_month": target_month or "All Time",
        "total_this_month": 0.0,
        "total_last_month": 0.0,
        "mom_change_pct": 0.0,
        "avg_daily_spend": 0.0,
        "biggest_expense_amount": 0.0,
        "biggest_expense_merchant": "None",
        "biggest_expense_date": "",
        "total_transactions": len(df),
        "total_spend_all_time": float(df["amount"].sum()) if not df.empty else 0.0,
    }

    if df.empty:
        return kpis

    # Ensure datetime format for month grouping
    temp_df = df.copy()
    temp_df["dt"] = pd.to_datetime(temp_df["date"])
    temp_df["month_str"] = temp_df["dt"].dt.strftime("%Y-%m")

    # If target_month not supplied, default to the latest available month
    if not target_month or target_month == "All Time":
        sorted_months = sorted(temp_df["month_str"].unique())
        cur_month = sorted_months[-1] if sorted_months else None
    else:
        cur_month = target_month

    kpis["target_month"] = cur_month

    if not cur_month:
        return kpis

    # Compute previous month string ('YYYY-MM')
    cur_year, cur_m = map(int, cur_month.split("-"))
    if cur_m == 1:
        prev_month = f"{cur_year - 1}-12"
    else:
        prev_month = f"{cur_year}-{cur_m - 1:02d}"

    this_month_df = temp_df[temp_df["month_str"] == cur_month]
    prev_month_df = temp_df[temp_df["month_str"] == prev_month]

    this_month_sum = float(this_month_df["amount"].sum())
    prev_month_sum = float(prev_month_df["amount"].sum())

    kpis["total_this_month"] = this_month_sum
    kpis["total_last_month"] = prev_month_sum

    if prev_month_sum > 0:
        kpis["mom_change_pct"] = round(
            ((this_month_sum - prev_month_sum) / prev_month_sum) * 100.0, 1
        )
    else:
        kpis["mom_change_pct"] = 0.0

    # Average daily spend
    now = datetime.now()
    if cur_month == now.strftime("%Y-%m"):
        days_in_period = max(1, now.day)
    else:
        days_in_period = monthrange(cur_year, cur_m)[1]

    kpis["avg_daily_spend"] = round(this_month_sum / days_in_period, 2)

    # Biggest single expense in target month
    if not this_month_df.empty:
        max_row = this_month_df.loc[this_month_df["amount"].idxmax()]
        kpis["biggest_expense_amount"] = float(max_row["amount"])
        kpis["biggest_expense_merchant"] = str(max_row["merchant"])
        kpis["biggest_expense_date"] = str(max_row["date"])

    return kpis


def get_monthly_trend(df: pd.DataFrame) -> pd.DataFrame:
    """Group spending by month (YYYY-MM) with total sum and transaction count."""
    if df.empty:
        return pd.DataFrame(columns=["month", "total_amount", "transaction_count", "avg_amount"])

    temp = df.copy()
    temp["month"] = pd.to_datetime(temp["date"]).dt.strftime("%Y-%m")
    trend = (
        temp.groupby("month")
        .agg(
            total_amount=("amount", "sum"),
            transaction_count=("amount", "count"),
            avg_amount=("amount", "mean"),
        )
        .reset_index()
        .sort_values("month")
    )
    return trend


def plot_monthly_trend(
    trend_df: pd.DataFrame,
    forecast_df: Optional[pd.DataFrame] = None,
) -> go.Figure:
    """
    Generate an interactive Plotly line chart of monthly expenses.

    If forecast_df is provided, renders the next month's predicted spend
    as a distinctive dashed orange line with an open marker.
    """
    fig = go.Figure()

    if trend_df.empty:
        fig.add_annotation(
            text="No transaction data available for trend chart",
            showarrow=False,
            font=dict(size=14, color="gray"),
        )
        fig.update_layout(template="plotly_white", height=380)
        return fig

    # 1. Historical Actual Spending Line
    hover_actual = [
        f"<b>{m}</b><br>Actual Spend: {format_inr(amt)}<br>Transactions: {cnt}"
        for m, amt, cnt in zip(
            trend_df["month"], trend_df["total_amount"], trend_df["transaction_count"]
        )
    ]

    fig.add_trace(
        go.Scatter(
            x=trend_df["month"],
            y=trend_df["total_amount"],
            mode="lines+markers",
            name="Actual Spend",
            line=dict(color="#2563eb", width=2.5),
            marker=dict(size=7, color="#ffffff", line=dict(color="#2563eb", width=2)),
            fill="tozeroy",
            fillcolor="rgba(37, 99, 235, 0.04)",
            hoverinfo="text",
            text=hover_actual,
        )
    )

    # 2. Forecast Extension (if provided)
    if forecast_df is not None and not forecast_df.empty:
        # Connect last actual month to the forecast month
        last_actual_month = trend_df.iloc[-1]["month"]
        last_actual_amt = trend_df.iloc[-1]["total_amount"]

        forecast_month = forecast_df.iloc[0]["month"]
        forecast_amt = forecast_df.iloc[0]["forecast_amount"]

        fig.add_trace(
            go.Scatter(
                x=[last_actual_month, forecast_month],
                y=[last_actual_amt, forecast_amt],
                mode="lines+markers",
                name="3-Month WMA Forecast",
                line=dict(color="#64748b", width=2, dash="dot"),
                marker=dict(size=7, symbol="diamond", color="#64748b"),
                hoverinfo="text",
                text=[
                    f"<b>Anchor:</b> {last_actual_month}",
                    f"<b>Projected Spend:</b> {format_inr(forecast_amt)}<br><b>Model:</b> 3-Month WMA",
                ],
            )
        )

    fig.update_layout(
        font=dict(family="Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif", size=12, color="#64748b"),
        xaxis=dict(gridcolor="#f1f5f9", zerolinecolor="#e2e8f0", showline=False, title_text=""),
        yaxis=dict(gridcolor="#f1f5f9", zerolinecolor="#e2e8f0", showline=False, title_text="Spend (₹)"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=340,
        hovermode="closest",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=11, color="#64748b")),
        margin=dict(l=10, r=10, t=25, b=10),
    )
    return fig


def get_category_breakdown(
    df: pd.DataFrame, month: Optional[str] = None
) -> pd.DataFrame:
    """Calculate total spend and percentage contribution per category."""
    if df.empty:
        return pd.DataFrame(columns=["category", "amount", "percentage"])

    temp = df.copy()
    if month and month != "All Months":
        temp["m"] = pd.to_datetime(temp["date"]).dt.strftime("%Y-%m")
        temp = temp[temp["m"] == month]

    if temp.empty:
        return pd.DataFrame(columns=["category", "amount", "percentage"])

    breakdown = (
        temp.groupby("category")["amount"]
        .sum()
        .reset_index()
        .sort_values(by="amount", ascending=False)
    )
    total = breakdown["amount"].sum()
    breakdown["percentage"] = (breakdown["amount"] / total * 100.0).round(1)
    return breakdown


def plot_category_donut(
    cat_df: pd.DataFrame, title: str = "Category Breakdown"
) -> go.Figure:
    """Interactive Plotly Donut chart for category expense distribution."""
    if cat_df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No data for category chart", showarrow=False, font=dict(size=13, color="#64748b")
        )
        fig.update_layout(template="plotly_white", height=340, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        return fig

    custom_text = [
        f"<b>{row['category']}</b><br>Spend: {format_inr(row['amount'])}<br>Share: {row['percentage']}%"
        for _, row in cat_df.iterrows()
    ]

    palette = [
        "#2563eb", "#0ea5e9", "#10b981", "#6366f1", "#f59e0b",
        "#ec4899", "#8b5cf6", "#14b8a6", "#64748b", "#f97316"
    ]

    fig = go.Figure(
        data=[
            go.Pie(
                labels=cat_df["category"],
                values=cat_df["amount"],
                hole=0.60,
                textinfo="label+percent",
                hoverinfo="text",
                text=custom_text,
                marker=dict(colors=palette[: len(cat_df)], line=dict(color="#ffffff", width=2)),
            )
        ]
    )

    fig.update_layout(
        font=dict(family="Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif", size=11, color="#475569"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=340,
        showlegend=False,
        margin=dict(l=10, r=10, t=20, b=10),
    )
    return fig


def get_top_merchants(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """Retrieve top N merchants by total expenditure."""
    if df.empty:
        return pd.DataFrame(columns=["merchant", "amount", "count"])

    top = (
        df.groupby("merchant")
        .agg(amount=("amount", "sum"), count=("id", "count"))
        .reset_index()
        .sort_values(by="amount", ascending=False)
        .head(n)
    )
    return top


def plot_top_merchants(merchants_df: pd.DataFrame) -> go.Figure:
    """Horizontal bar chart showing top merchants by total spend."""
    if merchants_df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No merchant data available", showarrow=False, font=dict(size=13, color="#64748b")
        )
        fig.update_layout(template="plotly_white", height=340, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        return fig

    # Reverse to have highest on top in horizontal bar
    df_sorted = merchants_df.sort_values(by="amount", ascending=True)

    hover_text = [
        f"<b>{m}</b><br>Total: {format_inr(amt)}<br>Transactions: {cnt}"
        for m, amt, cnt in zip(
            df_sorted["merchant"], df_sorted["amount"], df_sorted["count"]
        )
    ]

    fig = go.Figure(
        go.Bar(
            x=df_sorted["amount"],
            y=df_sorted["merchant"],
            orientation="h",
            marker=dict(color="#1e293b", opacity=0.88),
            hoverinfo="text",
            text=hover_text,
        )
    )

    fig.update_layout(
        font=dict(family="Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif", size=12, color="#64748b"),
        xaxis=dict(gridcolor="#f1f5f9", zerolinecolor="#e2e8f0", showline=False, title_text="Total Spend (₹)"),
        yaxis=dict(gridcolor="#f1f5f9", zerolinecolor="#e2e8f0", showline=False, title_text=""),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=340,
        margin=dict(l=10, r=10, t=20, b=10),
    )
    return fig


def get_weekday_vs_weekend(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Compare spending totals and daily averages between Weekdays (Mon-Fri)
    and Weekends (Sat-Sun).
    """
    if df.empty:
        return {
            "weekday_total": 0.0,
            "weekend_total": 0.0,
            "weekday_count": 0,
            "weekend_count": 0,
            "weekday_avg_txn": 0.0,
            "weekend_avg_txn": 0.0,
            "weekday_pct": 0.0,
            "weekend_pct": 0.0,
        }

    temp = df.copy()
    temp["dt"] = pd.to_datetime(temp["date"])
    # 0=Monday ... 4=Friday, 5=Saturday, 6=Sunday
    temp["is_weekend"] = temp["dt"].dt.weekday >= 5

    weekday_rows = temp[~temp["is_weekend"]]
    weekend_rows = temp[temp["is_weekend"]]

    w_tot = float(weekday_rows["amount"].sum())
    e_tot = float(weekend_rows["amount"].sum())
    total_spend = max(0.01, w_tot + e_tot)

    return {
        "weekday_total": w_tot,
        "weekend_total": e_tot,
        "weekday_count": len(weekday_rows),
        "weekend_count": len(weekend_rows),
        "weekday_avg_txn": float(weekday_rows["amount"].mean()) if not weekday_rows.empty else 0.0,
        "weekend_avg_txn": float(weekend_rows["amount"].mean()) if not weekend_rows.empty else 0.0,
        "weekday_pct": round((w_tot / total_spend) * 100.0, 1),
        "weekend_pct": round((e_tot / total_spend) * 100.0, 1),
    }


def plot_weekday_vs_weekend(comp: Dict[str, Any]) -> go.Figure:
    """Grouped bar chart for Weekday vs Weekend spending analytics."""
    categories = ["Total Spend (₹)", "Average per Txn (₹)"]
    weekday_vals = [comp["weekday_total"], comp["weekday_avg_txn"]]
    weekend_vals = [comp["weekend_total"], comp["weekend_avg_txn"]]

    fig = go.Figure(
        data=[
            go.Bar(
                name=f"Weekday ({comp['weekday_pct']}%)",
                x=categories,
                y=weekday_vals,
                marker_color="#2563eb",
                text=[format_inr(v) for v in weekday_vals],
                textposition="auto",
                textfont=dict(size=11, color="#ffffff"),
            ),
            go.Bar(
                name=f"Weekend ({comp['weekend_pct']}%)",
                x=categories,
                y=weekend_vals,
                marker_color="#94a3b8",
                text=[format_inr(v) for v in weekend_vals],
                textposition="auto",
                textfont=dict(size=11, color="#ffffff"),
            ),
        ]
    )

    fig.update_layout(
        barmode="group",
        font=dict(family="Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif", size=12, color="#64748b"),
        xaxis=dict(gridcolor="#f1f5f9", zerolinecolor="#e2e8f0", showline=False),
        yaxis=dict(gridcolor="#f1f5f9", zerolinecolor="#e2e8f0", showline=False, title_text="Amount (₹)"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=340,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=11, color="#64748b")),
        margin=dict(l=10, r=10, t=25, b=10),
    )
    return fig


def get_budget_vs_actual(
    expenses_df: pd.DataFrame,
    budgets_df: pd.DataFrame,
    month: Optional[str] = None,
) -> pd.DataFrame:
    """
    Merge actual monthly spending per category against configured budgets.

    Computes:
      - 'spent': Actual rupees spent in category
      - 'monthly_limit': Monthly budget limit
      - 'remaining': monthly_limit - spent
      - 'pct_used': (spent / monthly_limit) * 100
      - 'status': 'Over Budget' (>100%), 'Near Limit' (80-100%), or 'Within Budget' (<80%)
    """
    if budgets_df.empty:
        return pd.DataFrame()

    temp = expenses_df.copy()
    if month and month != "All Months" and not temp.empty:
        temp["m"] = pd.to_datetime(temp["date"]).dt.strftime("%Y-%m")
        temp = temp[temp["m"] == month]

    # Actual spending per category
    if not temp.empty:
        spent_series = temp.groupby("category")["amount"].sum()
    else:
        spent_series = pd.Series(dtype=float)

    merged = budgets_df.copy()
    merged["spent"] = merged["category"].map(spent_series).fillna(0.0)
    merged["remaining"] = merged["monthly_limit"] - merged["spent"]
    merged["pct_used"] = (
        (merged["spent"] / merged["monthly_limit"].replace(0, np.nan)) * 100.0
    ).fillna(0.0).round(1)

    def determine_status(pct: float) -> str:
        if pct > 100.0:
            return "Over Budget"
        elif pct >= 80.0:
            return "Near Limit"
        return "Within Budget"

    merged["status"] = merged["pct_used"].apply(determine_status)
    return merged.sort_values(by="pct_used", ascending=False)


def plot_budget_vs_actual(budget_df: pd.DataFrame) -> go.Figure:
    """Horizontal stacked or grouped bar chart showing budget utilization."""
    if budget_df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No budget data configured", showarrow=False, font=dict(size=14)
        )
        fig.update_layout(template="plotly_white", height=420)
        return fig

    # Sort so most utilized category is on top
    df_sorted = budget_df.sort_values(by="pct_used", ascending=True)

    # Color code actual bars based on status:
    # Crimson for Over Budget, Amber for Near Limit, Emerald for Within Budget
    colors = []
    for s in df_sorted["status"]:
        if s == "Over Budget":
            colors.append("#ef4444")
        elif s == "Near Limit":
            colors.append("#f59e0b")
        else:
            colors.append("#10b981")

    fig = go.Figure()

    # Bar 1: Budget Limit Track
    fig.add_trace(
        go.Bar(
            y=df_sorted["category"],
            x=df_sorted["monthly_limit"],
            name="Budget Target",
            orientation="h",
            marker=dict(color="#f1f5f9", line=dict(color="#e2e8f0", width=1)),
            hovertemplate="<b>%{y}</b><br>Budget Target: %{text}<extra></extra>",
            text=[format_inr(l) for l in df_sorted["monthly_limit"]],
        )
    )

    # Bar 2: Actual Spent
    fig.add_trace(
        go.Bar(
            y=df_sorted["category"],
            x=df_sorted["spent"],
            name="Actual Spent",
            orientation="h",
            marker=dict(color=colors),
            hovertemplate="<b>%{y}</b><br>Actual Spent: %{text}<extra></extra>",
            text=[
                f"{format_inr(s)} ({p}% - {stat})"
                for s, p, stat in zip(
                    df_sorted["spent"], df_sorted["pct_used"], df_sorted["status"]
                )
            ],
            textposition="inside",
            insidetextanchor="end",
            textfont=dict(size=11, color="#ffffff"),
        )
    )

    fig.update_layout(
        barmode="overlay",
        font=dict(family="Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif", size=12, color="#64748b"),
        xaxis=dict(gridcolor="#f1f5f9", zerolinecolor="#e2e8f0", showline=False, title_text="Amount (₹)"),
        yaxis=dict(gridcolor="#f1f5f9", zerolinecolor="#e2e8f0", showline=False),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=360,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=11, color="#64748b")),
        margin=dict(l=10, r=10, t=25, b=25),
    )
    return fig

