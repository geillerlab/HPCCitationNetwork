"""Visualization functions for citation networks."""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import networkx as nx
from pyvis.network import Network


# Color palette for review categories
CATEGORY_COLORS = {
    "general_attractor": "#888888",       # gray
    "point_attractor": "#e74c3c",         # red
    "continuous_attractor": "#3498db",     # blue
    "sequence": "#2ecc71",                # green
    "successor_representation": "#f39c12", # orange
    "btsp": "#9b59b6",                    # purple
    "bespoke": "#1abc9c",                 # teal
    "autonomous_dynamics": "#e67e22",     # dark orange
    "uncategorized": "#cccccc",           # light gray
    "": "#cccccc",                        # default
}


def get_node_color(data: dict) -> str:
    """Get color for a node based on its seed category."""
    if data.get("is_seed"):
        cat = data.get("seed_category", "")
        return CATEGORY_COLORS.get(cat, CATEGORY_COLORS[""])
    return "#dddddd"  # light gray for non-seed papers


def interactive_seed_graph(
    G: nx.DiGraph,
    output_path: str | Path = "data/processed/seed_network.html",
    height: str = "800px",
    width: str = "100%",
) -> Path:
    """Create an interactive HTML visualization of the seed subgraph.

    Uses pyvis for a force-directed layout. Nodes are colored by category,
    sized by in-degree within the network.

    Args:
        G: Full citation graph (seed subgraph will be extracted).
        output_path: Where to save the HTML file.
        height: Height of the visualization.
        width: Width of the visualization.

    Returns:
        Path to the saved HTML file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Extract seed subgraph
    seed_nodes = [n for n, d in G.nodes(data=True) if d.get("is_seed")]
    seed_G = G.subgraph(seed_nodes).copy()

    # Compute in-degree for sizing
    in_degrees = dict(G.in_degree())  # use full graph in-degree for sizing

    net = Network(
        height=height,
        width=width,
        directed=True,
        notebook=False,
        bgcolor="#ffffff",
    )

    # Configure physics
    net.set_options("""
    {
        "physics": {
            "forceAtlas2Based": {
                "gravitationalConstant": -100,
                "centralGravity": 0.01,
                "springLength": 150,
                "springConstant": 0.02,
                "damping": 0.4
            },
            "solver": "forceAtlas2Based",
            "stabilization": {
                "iterations": 200
            }
        },
        "edges": {
            "arrows": {"to": {"enabled": true, "scaleFactor": 0.5}},
            "color": {"color": "#cccccc", "opacity": 0.5},
            "smooth": {"type": "continuous"}
        },
        "interaction": {
            "hover": true,
            "tooltipDelay": 100
        }
    }
    """)

    # Add nodes
    for node in seed_G.nodes():
        data = G.nodes[node]
        year = data.get("publication_year", "?")
        author = data.get("first_author", "Unknown")
        title = data.get("title", "")
        cat = data.get("seed_category", "")
        color = get_node_color(data)

        # Size by in-degree (linear scale, min 10, max 50)
        deg = in_degrees.get(node, 0)
        size = min(50, max(10, 5 + deg * 0.2))

        label = f"{author} {year}"
        hover_text = f"<b>{author} ({year})</b><br>{title[:100]}<br>Category: {cat}<br>In-degree: {deg}"

        net.add_node(
            node,
            label=label,
            title=hover_text,
            color=color,
            size=size,
            font={"size": 10},
        )

    # Add edges
    for u, v in seed_G.edges():
        net.add_edge(u, v)

    net.save_graph(str(output_path))
    return output_path


def plot_network_overview(
    G: nx.DiGraph,
    output_path: str | Path = "data/processed/network_overview.png",
    figsize: tuple[int, int] = (16, 12),
    seed_only: bool = False,
) -> Path:
    """Create a static matplotlib plot of the citation network.

    Seeds are colored by category, non-seeds are small gray dots.

    Args:
        G: Citation graph.
        output_path: Where to save the image.
        figsize: Figure size.
        seed_only: If True, only plot seed papers.

    Returns:
        Path to the saved image.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if seed_only:
        seed_nodes = [n for n, d in G.nodes(data=True) if d.get("is_seed")]
        plot_G = G.subgraph(seed_nodes).copy()
    else:
        plot_G = G

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # Layout
    if plot_G.number_of_nodes() < 200:
        pos = nx.spring_layout(plot_G, k=2, iterations=100, seed=42)
    else:
        # For large graphs, use a faster layout
        pos = nx.spring_layout(plot_G, k=0.5, iterations=50, seed=42)

    # Separate seeds and non-seeds
    seeds = [n for n in plot_G.nodes() if plot_G.nodes[n].get("is_seed")]
    non_seeds = [n for n in plot_G.nodes() if not plot_G.nodes[n].get("is_seed")]

    # Draw non-seed nodes (small, gray)
    if non_seeds:
        nx.draw_networkx_nodes(
            plot_G, pos, nodelist=non_seeds,
            node_size=5, node_color="#dddddd", alpha=0.3, ax=ax,
        )

    # Draw seed nodes (colored by category)
    if seeds:
        seed_colors = [get_node_color(plot_G.nodes[n]) for n in seeds]
        in_deg = dict(G.in_degree())
        seed_sizes = [max(30, min(300, in_deg.get(n, 0) * 1.5)) for n in seeds]

        nx.draw_networkx_nodes(
            plot_G, pos, nodelist=seeds,
            node_size=seed_sizes, node_color=seed_colors,
            edgecolors="black", linewidths=0.5, alpha=0.9, ax=ax,
        )

    # Draw edges
    nx.draw_networkx_edges(
        plot_G, pos, alpha=0.1, arrows=True,
        arrowsize=5, edge_color="#999999", ax=ax,
    )

    # Labels for seed nodes (if not too many)
    if len(seeds) <= 100:
        seed_labels = {}
        for n in seeds:
            d = plot_G.nodes[n]
            author = (d.get("first_author") or "?").split(",")[0].split()[-1]  # last name
            year = d.get("publication_year", "")
            seed_labels[n] = f"{author}\n{year}"

        nx.draw_networkx_labels(
            plot_G, pos, labels=seed_labels,
            font_size=5, font_weight="bold", ax=ax,
        )

    # Legend
    legend_handles = []
    for cat, color in CATEGORY_COLORS.items():
        if cat and cat != "uncategorized":
            legend_handles.append(
                plt.Line2D([0], [0], marker='o', color='w',
                          markerfacecolor=color, markersize=8,
                          label=cat.replace("_", " "))
            )
    ax.legend(handles=legend_handles, loc="upper left", fontsize=8, framealpha=0.9)

    ax.set_title(f"Citation Network ({plot_G.number_of_nodes()} papers, {plot_G.number_of_edges()} citations)")
    ax.axis("off")

    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close()
    return output_path
