import networkx as nx
import matplotlib.pyplot as plt


def plot_business_review_ring(
    G,
    business_node,
    max_reviewers=20,
    figsize=(10, 10),
    save_path=None,
    title=None,
):
    """
    Visualize a suspicious business and its connected reviewers,
    plus what else those reviewers reviewed.

    Parameters
    ----------
    G : networkx.Graph
        Bipartite graph with nodes like 'u:<user_id>' and 'b:<business_id>'
    business_node : str
        e.g. 'b:<business_id>'
    save_path : str | None
        If provided, saves the figure (png) to this path.
    """

    if business_node not in G:
        raise ValueError(f"{business_node} not found in graph")

    # reviewers of the target business
    reviewers = list(G.neighbors(business_node))[:max_reviewers]

    # expand to what else those reviewers reviewed
    nodes = {business_node}
    for r in reviewers:
        nodes.add(r)
        nodes.update(G.neighbors(r))

    subgraph = G.subgraph(nodes)

    # color + size nodes
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
