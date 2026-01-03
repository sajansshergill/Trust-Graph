# src/viz.py
from __future__ import annotations

import networkx as nx
import matplotlib.pyplot as plt


def plot_business_review_ring(
    G,
    business_node: str,
    max_reviewers: int = 20,
    figsize=(10, 10),
    save_path: str | None = None,
    title: str | None = None,
):
    """
    Visualize a business and its connected reviewers, plus what else those reviewers reviewed.

    Nodes:
      - business nodes: 'b:<business_id>'
      - reviewer nodes: 'u:<user_id>'
    """

    if business_node not in G:
        raise ValueError(f"{business_node} not found in graph")

    # reviewers connected to the business
    reviewers = list(G.neighbors(business_node))[:max_reviewers]

    # expand: business -> reviewers -> other businesses
    nodes = {business_node}
    for r in reviewers:
        nodes.add(r)
        nodes.update(G.neighbors(r))

    subgraph = G.subgraph(nodes)

    # node styling
    colors, sizes = [], []
    for n in subgraph.nodes():
        if n == business_node:
            colors.append("red"); sizes.append(900)
        elif n.startswith("u:"):
            colors.append("orange"); sizes.append(320)
        else:
            colors.append("lightblue"); sizes.append(420)

    pos = nx.spring_layout(subgraph, seed=42)

    plt.figure(figsize=figsize)
    nx.draw(
        subgraph,
        pos,
        node_color=colors,
        node_size=sizes,
        edge_color="gray",
        alpha=0.7,
        with_labels=False,
    )

    plt.title(title or "Suspicious Review Ring (Business + Reviewers)")

    if save_path:
        plt.tight_layout()
        plt.savefig(save_path, dpi=200)

    plt.show()


def plot_reviewer_review_ring(
    G,
    reviewer_node: str,
    max_businesses: int = 25,
    figsize=(10, 10),
    save_path: str | None = None,
    title: str | None = None,
):
    """
    Visualize a reviewer and the businesses they reviewed,
    plus other reviewers connected to those businesses (local ring).

    Nodes:
      - business nodes: 'b:<business_id>'
      - reviewer nodes: 'u:<user_id>'
    """

    if reviewer_node not in G:
        raise ValueError(f"{reviewer_node} not found in graph")

    businesses = list(G.neighbors(reviewer_node))[:max_businesses]

    # expand: reviewer -> businesses -> other reviewers
    nodes = {reviewer_node}
    for b in businesses:
        nodes.add(b)
        nodes.update(G.neighbors(b))

    subgraph = G.subgraph(nodes)

    # node styling
    colors, sizes = [], []
    for n in subgraph.nodes():
        if n == reviewer_node:
            colors.append("red"); sizes.append(900)
        elif n.startswith("b:"):
            colors.append("lightblue"); sizes.append(420)
        else:
            colors.append("orange"); sizes.append(320)

    pos = nx.spring_layout(subgraph, seed=42)

    plt.figure(figsize=figsize)
    nx.draw(
        subgraph,
        pos,
        node_color=colors,
        node_size=sizes,
        edge_color="gray",
        alpha=0.7,
        with_labels=False,
    )

    plt.title(title or "Suspicious Reviewer Ring (Reviewer + Businesses)")

    if save_path:
        plt.tight_layout()
        plt.savefig(save_path, dpi=200)

    plt.show()
