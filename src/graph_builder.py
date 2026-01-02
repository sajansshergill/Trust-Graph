from __future__ import annotations

from typing import Tuple
import networkx as nx
import pandas as pd


def build_bipartite_review_graph(reviews: pd.DataFrame) -> nx.Graph:
    """
    Bipartite graph: users (u:*) and businesses (b:*).
    Edge = at least one review. We aggregate edge attributes for basic signals.
    """
    G = nx.Graph()

    # Add nodes
    user_nodes = [f"u:{uid}" for uid in reviews["user_id"].unique()]
    biz_nodes = [f"b:{bid}" for bid in reviews["business_id"].unique()]

    G.add_nodes_from(user_nodes, bipartite="user")
    G.add_nodes_from(biz_nodes, bipartite="business")

    # Aggregate edge attributes
    grouped = (
        reviews.groupby(["user_id", "business_id"], as_index=False)
        .agg(
            n_reviews=("review_id", "count"),
            avg_stars=("stars", "mean"),
            first_date=("date", "min"),
            last_date=("date", "max"),
        )
    )

    for row in grouped.itertuples(index=False):
        u = f"u:{row.user_id}"
        b = f"b:{row.business_id}"
        G.add_edge(
            u, b,
            n_reviews=int(row.n_reviews),
            avg_stars=float(row.avg_stars),
            first_date=row.first_date,
            last_date=row.last_date,
        )

    return G


def bipartite_projections(G: nx.Graph) -> Tuple[nx.Graph, nx.Graph]:
    """
    Returns:
      Guu: user-user projection weighted by shared businesses
      Gbb: business-business projection weighted by shared users
    """
    users = {n for n, d in G.nodes(data=True) if d.get("bipartite") == "user"}
    businesses = set(G) - users

    Guu = nx.algorithms.bipartite.weighted_projected_graph(G, users)
    Gbb = nx.algorithms.bipartite.weighted_projected_graph(G, businesses)
    return Guu, Gbb
