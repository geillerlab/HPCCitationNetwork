"""Tests for the citation graph builder."""

from src.data.storage import CitationDB
from src.network.builder import (
    build_citation_graph,
    get_seed_subgraph,
    get_level_subgraph,
    graph_summary,
    find_top_cited_in_network,
)


def _populate_test_db(db: CitationDB) -> None:
    """Add a small test network to the database."""
    # Seeds (level 0)
    db.upsert_paper({"openalex_id": "W1", "title": "Hopfield 1982", "publication_year": 1982,
                      "first_author": "Hopfield", "is_seed": True, "seed_category": "point_attractor",
                      "snowball_level": 0, "cited_by_count": 500})
    db.upsert_paper({"openalex_id": "W2", "title": "Samsonovich 1997", "publication_year": 1997,
                      "first_author": "Samsonovich", "is_seed": True, "seed_category": "continuous_attractor",
                      "snowball_level": 0, "cited_by_count": 300})
    db.upsert_paper({"openalex_id": "W3", "title": "Levy 1996", "publication_year": 1996,
                      "first_author": "Levy", "is_seed": True, "seed_category": "sequence",
                      "snowball_level": 0, "cited_by_count": 200})

    # Level 1 neighbors
    db.upsert_paper({"openalex_id": "W10", "title": "PointFollower 2010", "publication_year": 2010,
                      "first_author": "Follower", "snowball_level": 1, "cited_by_count": 50})
    db.upsert_paper({"openalex_id": "W11", "title": "Bridge Paper 2015", "publication_year": 2015,
                      "first_author": "Bridge", "snowball_level": 1, "cited_by_count": 80})

    # Citations: W10 cites W1; W11 cites W1 and W2 (bridge); W2 cites W3
    db.add_citations_bulk([
        ("W10", "W1"),
        ("W11", "W1"),
        ("W11", "W2"),
        ("W2", "W3"),
    ])


def test_build_citation_graph(tmp_path):
    """Test building a graph from the test database."""
    db = CitationDB(tmp_path / "test.db")
    _populate_test_db(db)

    G = build_citation_graph(db)
    assert G.number_of_nodes() == 5
    assert G.number_of_edges() == 4

    # Check node attributes
    assert G.nodes["W1"]["title"] == "Hopfield 1982"
    assert G.nodes["W1"]["is_seed"] is True
    assert G.nodes["W1"]["seed_category"] == "point_attractor"
    assert G.nodes["W10"]["is_seed"] is False
    db.close()


def test_seed_subgraph(tmp_path):
    """Test extracting seed-only subgraph."""
    db = CitationDB(tmp_path / "test.db")
    _populate_test_db(db)
    G = build_citation_graph(db)

    seed_G = get_seed_subgraph(G)
    assert seed_G.number_of_nodes() == 3  # W1, W2, W3
    assert ("W2", "W3") in seed_G.edges()  # W2 cites W3
    assert ("W10", "W1") not in seed_G.edges()  # non-seed edge excluded
    db.close()


def test_level_subgraph(tmp_path):
    """Test extracting subgraph by snowball level."""
    db = CitationDB(tmp_path / "test.db")
    _populate_test_db(db)
    G = build_citation_graph(db)

    level_0 = get_level_subgraph(G, max_level=0)
    assert level_0.number_of_nodes() == 3  # seeds only

    level_1 = get_level_subgraph(G, max_level=1)
    assert level_1.number_of_nodes() == 5  # all papers
    db.close()


def test_graph_summary(tmp_path):
    """Test summary statistics."""
    db = CitationDB(tmp_path / "test.db")
    _populate_test_db(db)
    G = build_citation_graph(db)

    stats = graph_summary(G)
    assert stats["nodes"] == 5
    assert stats["edges"] == 4
    assert stats["seed_papers"] == 3
    assert stats["nodes_by_level"][0] == 3
    assert stats["nodes_by_level"][1] == 2
    assert stats["weakly_connected_components"] >= 1
    db.close()


def test_find_top_cited(tmp_path):
    """Test finding most-cited papers in the network."""
    db = CitationDB(tmp_path / "test.db")
    _populate_test_db(db)
    G = build_citation_graph(db)

    top = find_top_cited_in_network(G, n=3)
    assert len(top) == 3
    # W1 (Hopfield) has highest in-degree (cited by W10 and W11)
    assert top[0]["openalex_id"] == "W1"
    assert top[0]["in_degree"] == 2
    assert top[0]["is_seed"] is True
    db.close()


def test_empty_graph(tmp_path):
    """Test building a graph from an empty database."""
    db = CitationDB(tmp_path / "test.db")
    G = build_citation_graph(db)
    assert G.number_of_nodes() == 0
    assert G.number_of_edges() == 0

    stats = graph_summary(G)
    assert stats["nodes"] == 0
    assert stats["edges"] == 0
    db.close()
