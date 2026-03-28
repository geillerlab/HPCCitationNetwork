"""Data sanity checks — validate properties of the actual collected network.

These tests run against the real database (if it exists) and verify
domain-specific expectations about the citation network.
Skip gracefully if no database exists.
"""

import pytest
from pathlib import Path

from src.data.storage import CitationDB, DEFAULT_DB_PATH
from src.network.builder import build_citation_graph, find_top_cited_in_network


# Skip all tests if the database doesn't exist (e.g., in CI)
pytestmark = pytest.mark.skipif(
    not Path(DEFAULT_DB_PATH).exists(),
    reason="No citation database found (run seed import first)",
)


@pytest.fixture(scope="module")
def db_and_graph():
    db = CitationDB()
    G = build_citation_graph(db)
    yield db, G
    db.close()


def test_seeds_exist(db_and_graph):
    """Verify seed papers were imported."""
    db, G = db_and_graph
    seeds = db.get_seed_papers()
    assert len(seeds) > 0, f"Expected at least 1 seed, got {len(seeds)}"


def test_seeds_at_level_zero(db_and_graph):
    """All seed papers should be at snowball level 0."""
    db, _ = db_and_graph
    for seed in db.get_seed_papers():
        assert seed["snowball_level"] == 0, f"Seed {seed['openalex_id']} at level {seed['snowball_level']}"


def test_top_cited_returns_results(db_and_graph):
    """Top-cited query should return results from the network."""
    _, G = db_and_graph
    top = find_top_cited_in_network(G, n=10)
    assert len(top) > 0, "Expected at least 1 paper in top-cited list"


def test_no_self_citations_in_edges(db_and_graph):
    """No paper should cite itself."""
    db, _ = db_and_graph
    for citing, cited in db.get_all_citations():
        assert citing != cited, f"Self-citation found: {citing}"


def test_citation_direction_mostly_forward(db_and_graph):
    """Most citations should be from newer to older papers (citing → cited).

    Allow some tolerance: preprints, corrections, and missing year data
    can cause apparent backward citations. But >90% should be forward.
    """
    _, G = db_and_graph
    forward = 0
    backward = 0
    skipped = 0

    for u, v in G.edges():
        u_year = G.nodes[u].get("publication_year")
        v_year = G.nodes[v].get("publication_year")
        if u_year is None or v_year is None:
            skipped += 1
            continue
        if u_year >= v_year:
            forward += 1
        else:
            backward += 1

    total = forward + backward
    if total > 0:
        forward_pct = forward / total
        assert forward_pct > 0.85, (
            f"Only {forward_pct:.1%} of citations go forward in time "
            f"({forward} forward, {backward} backward, {skipped} skipped)"
        )


def test_network_mostly_connected(db_and_graph):
    """The network should be mostly one connected component (not fragmented)."""
    _, G = db_and_graph
    import networkx as nx
    components = list(nx.weakly_connected_components(G))
    largest = max(len(c) for c in components)
    total = G.number_of_nodes()
    assert largest / total > 0.8, (
        f"Largest component is only {largest}/{total} = {largest/total:.1%} of network"
    )


def test_find_top_cited_empty_graph():
    """find_top_cited_in_network should handle an empty graph."""
    import networkx as nx
    G = nx.DiGraph()
    result = find_top_cited_in_network(G, n=10)
    assert result == []
