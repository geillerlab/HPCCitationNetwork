"""Build networkx citation graphs from the SQLite database."""

import networkx as nx

from src.data.storage import CitationDB


def build_citation_graph(db: CitationDB) -> nx.DiGraph:
    """Build a directed citation graph from all papers and citations in the DB.

    Edges go from citing paper → cited paper (A → B means A cites B).
    Node attributes include paper metadata from the DB.

    Args:
        db: Citation database with papers and citation edges.

    Returns:
        networkx DiGraph with paper metadata as node attributes.
    """
    G = nx.DiGraph()

    # Add all papers as nodes with attributes
    for paper in db.get_all_papers():
        G.add_node(
            paper["openalex_id"],
            title=paper.get("title", ""),
            publication_year=paper.get("publication_year"),
            first_author=paper.get("first_author", ""),
            journal=paper.get("journal", ""),
            cited_by_count=paper.get("cited_by_count", 0),
            is_seed=bool(paper.get("is_seed")),
            seed_category=paper.get("seed_category", ""),
            snowball_level=paper.get("snowball_level", -1),
            doi=paper.get("doi", ""),
        )

    # Add citation edges
    for citing_id, cited_id in db.get_all_citations():
        G.add_edge(citing_id, cited_id)

    return G


def get_seed_subgraph(G: nx.DiGraph) -> nx.DiGraph:
    """Extract the subgraph containing only seed papers.

    Includes edges between seed papers (direct citations among seeds).

    Args:
        G: Full citation graph.

    Returns:
        Subgraph of seed papers only.
    """
    seed_nodes = [n for n, d in G.nodes(data=True) if d.get("is_seed")]
    return G.subgraph(seed_nodes).copy()


def get_level_subgraph(G: nx.DiGraph, max_level: int) -> nx.DiGraph:
    """Extract subgraph of papers up to a given snowball level.

    Args:
        G: Full citation graph.
        max_level: Maximum snowball level to include (0 = seeds only).

    Returns:
        Subgraph with papers at snowball_level <= max_level.
    """
    nodes = [
        n for n, d in G.nodes(data=True)
        if d.get("snowball_level", -1) >= 0 and d.get("snowball_level", -1) <= max_level
    ]
    return G.subgraph(nodes).copy()


def graph_summary(G: nx.DiGraph) -> dict:
    """Generate summary statistics for a citation graph.

    Args:
        G: Citation graph.

    Returns:
        Dict with node count, edge count, seed count, level breakdown, etc.
    """
    nodes_by_level: dict[int, int] = {}
    seed_count = 0
    for _, data in G.nodes(data=True):
        level = data.get("snowball_level", -1)
        nodes_by_level[level] = nodes_by_level.get(level, 0) + 1
        if data.get("is_seed"):
            seed_count += 1

    # In-degree = how many papers cite this one (within the network)
    in_degrees = [d for _, d in G.in_degree()]
    out_degrees = [d for _, d in G.out_degree()]

    return {
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "seed_papers": seed_count,
        "nodes_by_level": dict(sorted(nodes_by_level.items())),
        "density": nx.density(G),
        "avg_in_degree": sum(in_degrees) / len(in_degrees) if in_degrees else 0,
        "max_in_degree": max(in_degrees) if in_degrees else 0,
        "avg_out_degree": sum(out_degrees) / len(out_degrees) if out_degrees else 0,
        "weakly_connected_components": nx.number_weakly_connected_components(G),
    }


def find_top_cited_in_network(G: nx.DiGraph, n: int = 20) -> list[dict]:
    """Find papers with highest in-degree within the network.

    These are the most-cited papers *within the collected network*,
    which may differ from their global citation count.

    Args:
        G: Citation graph.
        n: Number of top papers to return.

    Returns:
        List of dicts with paper info and in-degree, sorted by in-degree desc.
    """
    in_degrees = dict(G.in_degree())
    top_nodes = sorted(in_degrees, key=in_degrees.get, reverse=True)[:n]

    results = []
    for node in top_nodes:
        data = G.nodes[node]
        results.append({
            "openalex_id": node,
            "title": data.get("title", ""),
            "first_author": data.get("first_author", ""),
            "publication_year": data.get("publication_year"),
            "is_seed": data.get("is_seed", False),
            "seed_category": data.get("seed_category", ""),
            "in_degree": in_degrees[node],
            "cited_by_count_global": data.get("cited_by_count", 0),
        })

    return results
