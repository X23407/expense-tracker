"""
graph.py - NetworkX Spending Graph Analysis and Interactive Visualization

Constructs a tripartite network of:
- Merchants
- Categories
- Payment Methods

Computes:
1. Degree Centrality & Weighted Degree (Node Strength)
2. Betweenness Centrality (Bridge nodes across spending habits)
3. Modularity Community Detection (Greedy Modularity Clusters)
4. Interactive 2D Spring Layout visualization using Plotly
5. Plain-English natural language insights synthesized from graph topology
"""

import math
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx
from networkx.algorithms.community import greedy_modularity_communities
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# Ensure root in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.analytics import format_inr


def build_spending_graph(df: pd.DataFrame) -> nx.Graph:
    """
    Construct an undirected weighted bipartite/tripartite graph from transaction rows.

    Nodes:
      - Merchants (node_type='Merchant')
      - Categories (node_type='Category')
      - Payment Methods (node_type='Payment Method')

    Edges:
      - (Merchant, Category): weight = total spend, count = number of transactions
      - (Merchant, Payment Method): weight = total spend, count = number of transactions
    """
    G = nx.Graph()

    if df.empty:
        return G

    # Pre-aggregate transactions by (Merchant, Category) and (Merchant, Payment Method)
    # to avoid redundant graph lookups
    merch_cat = (
        df.groupby(["merchant", "category"])
        .agg(amount=("amount", "sum"), count=("id", "count"))
        .reset_index()
    )

    merch_pay = (
        df.groupby(["merchant", "payment_method"])
        .agg(amount=("amount", "sum"), count=("id", "count"))
        .reset_index()
    )

    # 1. Add Merchant <-> Category edges
    for _, row in merch_cat.iterrows():
        m = str(row["merchant"]).strip()
        c = str(row["category"]).strip()
        amt = float(row["amount"])
        cnt = int(row["count"])

        # Add nodes with type attribute
        if m not in G:
            G.add_node(m, node_type="Merchant", total_spend=0.0)
        if c not in G:
            G.add_node(c, node_type="Category", total_spend=0.0)

        G.nodes[m]["total_spend"] += amt
        G.nodes[c]["total_spend"] += amt

        G.add_edge(m, c, weight=amt, count=cnt, edge_type="Merchant-Category")

    # 2. Add Merchant <-> Payment Method edges
    for _, row in merch_pay.iterrows():
        m = str(row["merchant"]).strip()
        p = str(row["payment_method"]).strip()
        amt = float(row["amount"])
        cnt = int(row["count"])

        if m not in G:
            G.add_node(m, node_type="Merchant", total_spend=0.0)
        if p not in G:
            G.add_node(p, node_type="Payment Method", total_spend=0.0)

        # We don't double-add m total_spend here, but we record payment method spend
        G.nodes[p]["total_spend"] += amt

        G.add_edge(m, p, weight=amt, count=cnt, edge_type="Merchant-Payment")

    return G


def compute_graph_metrics(
    G: nx.Graph,
) -> Tuple[Dict[str, Dict[str, Any]], List[Set[str]]]:
    """
    Calculate degree centrality, betweenness centrality, weighted degree (strength),
    and greedy modularity community partitions.

    Returns:
        (node_metrics_dict, communities_list)
    """
    if len(G) == 0:
        return {}, []

    # 1. Degree Centrality: fraction of all possible other nodes this node connects to
    degree_centrality = nx.degree_centrality(G)

    # 2. Betweenness Centrality: measures how frequently a node acts as a bridge
    # along the shortest paths between any other pair of nodes
    betweenness_centrality = nx.betweenness_centrality(G, weight=None)

    # 3. Community Detection using Newman-Girvan style Greedy Modularity Maximization
    try:
        communities = list(greedy_modularity_communities(G, weight="weight"))
    except Exception:
        # Fallback if graph is disconnected or trivial
        communities = [set(G.nodes())]

    # Map each node to its community index (e.g. Cluster 1, Cluster 2)
    node_community_map: Dict[str, int] = {}
    for comm_idx, comm_set in enumerate(communities, start=1):
        for node in comm_set:
            node_community_map[node] = comm_idx

    # 4. Compile metrics dictionary per node
    metrics: Dict[str, Dict[str, Any]] = {}
    for node in G.nodes():
        node_data = G.nodes[node]
        # Weighted degree (node strength) = sum of weights on incident edges
        strength = sum(
            edge_data.get("weight", 0.0)
            for _, _, edge_data in G.edges(node, data=True)
        )

        metrics[node] = {
            "node": node,
            "node_type": node_data.get("node_type", "Unknown"),
            "degree": G.degree(node),
            "degree_centrality": round(degree_centrality.get(node, 0.0), 4),
            "betweenness_centrality": round(betweenness_centrality.get(node, 0.0), 4),
            "weighted_degree": round(strength, 2),
            "total_spend": round(node_data.get("total_spend", strength), 2),
            "community_id": node_community_map.get(node, 1),
            "community_name": f"Cluster {node_community_map.get(node, 1)}",
        }

    return metrics, communities


def create_network_plotly_figure(
    G: nx.Graph,
    metrics: Dict[str, Dict[str, Any]],
    color_by: str = "node_type",
) -> go.Figure:
    """
    Create an interactive 2D network visualization using Plotly and NetworkX Spring Layout.

    Args:
        G: NetworkX Graph.
        metrics: Computed metrics from compute_graph_metrics.
        color_by: 'node_type' (Merchant vs Category vs Payment Method)
                  OR 'community' (Greedy Modularity Clusters).
    """
    if len(G) == 0:
        fig = go.Figure()
        fig.add_annotation(
            text="No network data available to render graph",
            showarrow=False,
            font=dict(size=14, color="gray"),
        )
        fig.update_layout(template="plotly_white", height=600)
        return fig

    # 1. 2D Spring Layout (Fruchterman-Reingold force-directed algorithm)
    # seed fixed for layout reproducibility
    pos = nx.spring_layout(G, seed=42, k=0.55, iterations=60)

    # 2. Build Edge Traces
    edge_x = []
    edge_y = []
    edge_hover = []

    for u, v, data in G.edges(data=True):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        line=dict(width=0.8, color="#cbd5e1"),
        hoverinfo="none",
        mode="lines",
        opacity=0.5,
        showlegend=False,
    )

    # 3. Build Node Traces
    node_x = []
    node_y = []
    node_text = []
    node_sizes = []
    node_colors = []
    node_labels = []

    # Color palettes
    type_color_map = {
        "Category": "#2563eb",        # Slate Blue
        "Merchant": "#10b981",        # Emerald
        "Payment Method": "#f59e0b",  # Amber
        "Unknown": "#94a3b8",
    }

    cluster_palette = [
        "#2563eb", "#10b981", "#f59e0b", "#6366f1", "#ec4899",
        "#0ea5e9", "#8b5cf6", "#14b8a6", "#64748b", "#f97316"
    ]

    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_labels.append(node)

        m = metrics.get(node, {})
        ntype = m.get("node_type", "Unknown")
        comm_id = m.get("community_id", 1)
        spend = m.get("total_spend", 0.0)
        deg_c = m.get("degree_centrality", 0.0)
        bet_c = m.get("betweenness_centrality", 0.0)
        degree = m.get("degree", 0)

        # Node size scaled by log10 of spend (clamped between 14 and 45)
        if spend > 0:
            scaled_size = max(14.0, min(45.0, 10.0 + 5.0 * math.log10(max(10.0, spend))))
        else:
            scaled_size = 14.0
        node_sizes.append(scaled_size)

        # Determine node color
        if color_by == "community":
            color = cluster_palette[(comm_id - 1) % len(cluster_palette)]
        else:
            color = type_color_map.get(ntype, "#7f7f7f")
        node_colors.append(color)

        # Rich Tooltip
        tooltip = (
            f"<b>{node}</b><br>"
            f"<b>Type:</b> {ntype}<br>"
            f"<b>Total Volume:</b> {format_inr(spend)}<br>"
            f"<b>Connections:</b> {degree}<br>"
            f"<b>Degree Centrality:</b> {deg_c}<br>"
            f"<b>Betweenness Centrality:</b> {bet_c}<br>"
            f"<b>Cluster:</b> {m.get('community_name', 'N/A')}"
        )
        node_text.append(tooltip)

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        hoverinfo="text",
        text=node_labels,
        textposition="top center",
        hovertext=node_text,
        textfont=dict(size=10, family="Inter, -apple-system, sans-serif", color="#1e293b"),
        marker=dict(
            showscale=False,
            color=node_colors,
            size=node_sizes,
            line=dict(width=1.5, color="#ffffff"),
            opacity=0.92,
        ),
        showlegend=False,
    )

    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            font=dict(family="Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif", size=12, color="#64748b"),
            template="plotly_white",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            showlegend=False,
            hovermode="closest",
            margin=dict(b=10, l=10, r=10, t=20),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=600,
        ),
    )

    return fig


def generate_graph_insights(
    G: nx.Graph,
    metrics: Dict[str, Dict[str, Any]],
    communities: List[Set[str]],
    df: pd.DataFrame,
) -> List[Dict[str, str]]:
    """
    Synthesize plain-English insights from NetworkX graph metrics.

    Answers key financial questions:
    - What is the most dominant payment channel?
    - Which node acts as the primary financial bridge (highest betweenness)?
    - Which category accounts for the highest spend concentration?
    - What merchant clusters form natural habits?
    """
    if len(G) == 0 or df.empty:
        return [
            {
                "title": "No Data",
                "text": "Add transactions to generate financial graph insights.",
                "badge": "Info",
            }
        ]

    insights: List[Dict[str, str]] = []

    # 1. Dominant Payment Method Insight
    pay_methods = [
        node for node, d in G.nodes(data=True) if d.get("node_type") == "Payment Method"
    ]
    if pay_methods:
        top_pay = max(pay_methods, key=lambda p: metrics.get(p, {}).get("total_spend", 0.0))
        top_pay_spend = metrics[top_pay]["total_spend"]
        total_spent = df["amount"].sum()
        pct_pay = round((top_pay_spend / max(0.01, total_spent)) * 100.0, 1)

        insights.append({
            "title": "Dominant Payment Channel",
            "text": f"<b>{top_pay}</b> is your primary payment method, accounting for <b>{pct_pay}%</b> of your total outflow ({format_inr(top_pay_spend)}).",
            "badge": "Payment",
        })

    # 2. Bridge Node (Highest Betweenness Centrality)
    top_bridge = max(
        metrics.keys(), key=lambda n: metrics[n]["betweenness_centrality"]
    )
    bridge_metric = metrics[top_bridge]
    insights.append({
        "title": "Primary Spending Bridge",
        "text": (
            f"<b>{top_bridge}</b> ({bridge_metric['node_type']}) has the highest betweenness centrality "
            f"(<b>{bridge_metric['betweenness_centrality']}</b>), meaning it connects disparate categories "
            f"and serves as the central hub of your transaction habits."
        ),
        "badge": "Centrality",
    })

    # 3. Category Concentration Insight
    categories = [
        node for node, d in G.nodes(data=True) if d.get("node_type") == "Category"
    ]
    if categories:
        top_cat = max(categories, key=lambda c: metrics.get(c, {}).get("total_spend", 0.0))
        cat_spend = metrics[top_cat]["total_spend"]
        total_spent = df["amount"].sum()
        cat_pct = round((cat_spend / max(0.01, total_spent)) * 100.0, 1)

        insights.append({
            "title": "Top Spending Category",
            "text": f"<b>{top_cat}</b> is your largest category, representing <b>{cat_pct}%</b> of total spend ({format_inr(cat_spend)}).",
            "badge": "Category",
        })

    # 4. Merchant Habits / Clusters Insight
    if communities:
        # Find community with largest merchant membership
        best_comm = max(communities, key=len)
        merchants_in_comm = [
            n for n in best_comm if G.nodes[n].get("node_type") == "Merchant"
        ]
        cats_in_comm = [
            n for n in best_comm if G.nodes[n].get("node_type") == "Category"
        ]

        if merchants_in_comm:
            sample_merchants = ", ".join(merchants_in_comm[:4])
            cat_label = cats_in_comm[0] if cats_in_comm else "associated categories"
            insights.append({
                "title": f"Habit Cluster ({len(communities)} Clusters Detected)",
                "text": (
                    f"Modularity detection discovered <b>{len(communities)} spending clusters</b>. "
                    f"The largest cluster groups merchants like <b>{sample_merchants}</b> "
                    f"tightly around <b>{cat_label}</b>."
                ),
                "badge": "Community",
            })

    return insights

