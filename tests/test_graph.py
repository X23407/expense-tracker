"""
test_graph.py - Unit tests for NetworkX graph construction and metrics
"""

import networkx as nx
import pandas as pd
import pytest

from src.graph import (
    build_spending_graph,
    compute_graph_metrics,
    generate_graph_insights,
)


def test_hand_crafted_graph_construction():
    """
    Test graph construction on a tiny hand-made dataset:
    2 transactions:
    1. Swiggy: ₹500, Food & Dining, UPI
    2. Zomato: ₹300, Food & Dining, Credit Card
    Expected:
    Nodes: Swiggy, Zomato, Food & Dining, UPI, Credit Card (total 5 nodes)
    Edges:
      (Swiggy, Food & Dining), weight=500
      (Swiggy, UPI), weight=500
      (Zomato, Food & Dining), weight=300
      (Zomato, Credit Card), weight=300
      (total 4 edges)
    """
    df = pd.DataFrame([
        {"id": 1, "merchant": "Swiggy", "category": "Food & Dining", "payment_method": "UPI", "amount": 500.0},
        {"id": 2, "merchant": "Zomato", "category": "Food & Dining", "payment_method": "Credit Card", "amount": 300.0},
    ])

    G = build_spending_graph(df)

    assert len(G.nodes) == 5
    assert len(G.edges) == 4

    assert G.has_edge("Swiggy", "Food & Dining")
    assert G.get_edge_data("Swiggy", "Food & Dining")["weight"] == 500.0

    assert G.has_edge("Swiggy", "UPI")
    assert G.get_edge_data("Swiggy", "UPI")["weight"] == 500.0

    assert G.has_edge("Zomato", "Food & Dining")
    assert G.get_edge_data("Zomato", "Food & Dining")["weight"] == 300.0

    assert G.has_edge("Zomato", "Credit Card")
    assert G.get_edge_data("Zomato", "Credit Card")["weight"] == 300.0

    # Node types
    assert G.nodes["Swiggy"]["node_type"] == "Merchant"
    assert G.nodes["Food & Dining"]["node_type"] == "Category"
    assert G.nodes["UPI"]["node_type"] == "Payment Method"


def test_graph_centrality_and_communities():
    """Verify centralities and modularity communities compute without error."""
    df = pd.DataFrame([
        {"id": 1, "merchant": "Swiggy", "category": "Food & Dining", "payment_method": "UPI", "amount": 500.0},
        {"id": 2, "merchant": "Zomato", "category": "Food & Dining", "payment_method": "UPI", "amount": 400.0},
        {"id": 3, "merchant": "Uber", "category": "Transport", "payment_method": "UPI", "amount": 250.0},
        {"id": 4, "merchant": "Ola", "category": "Transport", "payment_method": "Credit Card", "amount": 300.0},
    ])

    G = build_spending_graph(df)
    metrics, communities = compute_graph_metrics(G)

    assert len(metrics) == len(G.nodes)
    assert len(communities) >= 1

    # In this network, UPI connects to Swiggy, Zomato, and Uber -> highest degree
    assert metrics["UPI"]["degree"] == 3

    # Betweenness centrality of central bridge nodes must be non-negative
    for node, m in metrics.items():
        assert m["betweenness_centrality"] >= 0.0
        assert m["degree_centrality"] >= 0.0
        assert m["weighted_degree"] > 0.0

    # Insights generation
    insights = generate_graph_insights(G, metrics, communities, df)
    assert len(insights) >= 3
    assert any("UPI" in i["text"] for i in insights)

