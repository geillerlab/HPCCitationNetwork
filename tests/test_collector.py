"""Tests for the snowball sampling collector."""

import responses

from src.data.collector import SnowballCollector
from src.data.openalex_client import OpenAlexClient
from src.data.storage import CitationDB


def _make_work(oa_id: str, doi: str = "", title: str = "Test", year: int = 2020, refs: list | None = None):
    """Helper to create a minimal OpenAlex work dict."""
    return {
        "id": oa_id,
        "doi": doi,
        "title": title,
        "publication_year": year,
        "authorships": [{"author": {"display_name": "Test Author"}}],
        "primary_location": {"source": {"display_name": "Test Journal"}},
        "cited_by_count": 10,
        "type": "article",
        "abstract": "",
        "referenced_works": refs or [],
        "concepts": [],
        "topics": [],
    }


SAMPLE_DOC = """
Point Attractors
Test paper https://doi.org/10.1234/seed1
"""


@responses.activate
def test_import_seeds(tmp_path):
    """Test that seed import resolves DOIs and stores papers."""
    work = _make_work("https://openalex.org/W1", "https://doi.org/10.1234/seed1", "Seed Paper", 2020,
                       refs=["https://openalex.org/W100"])

    responses.add(
        responses.GET,
        "https://api.openalex.org/works/doi:10.1234/seed1",
        json=work, status=200,
    )
    # Mock the manual supplement DOIs as 404s (they're not in our test doc)
    responses.add(
        responses.GET,
        responses.matchers.urlencoded_params_matcher({}) if False else "https://api.openalex.org/works/doi:10.48550/arxiv.1606.01164",
        json={"error": "not found"}, status=404,
    )
    responses.add(responses.GET, "https://api.openalex.org/works/doi:10.48550/arxiv.2008.02217", json={"error": "not found"}, status=404)
    responses.add(responses.GET, "https://api.openalex.org/works/doi:10.1002/(sici)1098-1063(1996)6:3%3C271::aid-hipo5%3E3.0.co;2-q", json={"error": "not found"}, status=404)
    responses.add(responses.GET, "https://api.openalex.org/works/doi:10.48550/arxiv.2505.01098", json={"error": "not found"}, status=404)
    responses.add(responses.GET, "https://api.openalex.org/works/doi:10.1101/lm.3.2-3.279", json={"error": "not found"}, status=404)
    responses.add(responses.GET, "https://api.openalex.org/works/doi:10.1023/a:1008820728122", json={"error": "not found"}, status=404)

    db = CitationDB(tmp_path / "test.db")
    client = OpenAlexClient(rate_limit_delay=0)
    collector = SnowballCollector(client, db)

    stats = collector.import_seeds(SAMPLE_DOC)
    assert stats["resolved"] >= 1

    seeds = db.get_seed_papers()
    assert len(seeds) >= 1
    assert seeds[0]["is_seed"] == 1
    assert seeds[0]["snowball_level"] == 0

    # References should be stored as citation edges
    citations = db.get_all_citations()
    assert len(citations) >= 1
    assert ("https://openalex.org/W1", "https://openalex.org/W100") in citations
    db.close()


@responses.activate
def test_collect_level_1(tmp_path):
    """Test level-1 snowball collection."""
    db = CitationDB(tmp_path / "test.db")
    client = OpenAlexClient(rate_limit_delay=0)
    collector = SnowballCollector(client, db)

    # Pre-populate a seed paper at level 0
    db.upsert_paper({
        "openalex_id": "https://openalex.org/W1",
        "title": "Seed Paper",
        "is_seed": True,
        "snowball_level": 0,
    })

    # Mock cited-by response (filters by cites:W1)
    responses.add(
        responses.GET,
        "https://api.openalex.org/works",
        json={
            "results": [
                {"id": "https://openalex.org/W200"},
                {"id": "https://openalex.org/W300"},
            ],
            "meta": {"next_cursor": None},
        },
        status=200,
    )

    # Mock get_work for the seed (collector fetches this to get references)
    # The collector strips https://openalex.org/ prefix, so it requests /works/W1
    responses.add(
        responses.GET,
        "https://api.openalex.org/works/W1",
        json=_make_work("https://openalex.org/W1", refs=["https://openalex.org/W400"]),
        status=200,
    )

    stats = collector.collect_level(level=1, max_cited_by=200)
    assert stats["papers_processed"] == 1
    assert stats["papers_added"] >= 2  # W200, W300 (and possibly W400)
    assert stats["errors"] == 0

    # Check that new papers are at level 1
    p200 = db.get_paper("https://openalex.org/W200")
    assert p200 is not None
    assert p200["snowball_level"] == 1
    db.close()


def test_graph_from_collector_data(tmp_path):
    """Integration test: collector data -> graph builder."""
    from src.network.builder import build_citation_graph, graph_summary

    db = CitationDB(tmp_path / "test.db")

    # Add seed + neighbors
    db.upsert_paper({"openalex_id": "W1", "title": "Seed", "is_seed": True, "snowball_level": 0, "publication_year": 2020})
    db.upsert_paper({"openalex_id": "W2", "title": "Neighbor", "snowball_level": 1, "publication_year": 2021})
    db.upsert_paper({"openalex_id": "W3", "title": "Neighbor2", "snowball_level": 1, "publication_year": 2019})
    db.add_citations_bulk([("W2", "W1"), ("W1", "W3")])

    G = build_citation_graph(db)
    assert G.number_of_nodes() == 3
    assert G.number_of_edges() == 2
    assert G.nodes["W1"]["is_seed"] is True

    stats = graph_summary(G)
    assert stats["seed_papers"] == 1
    assert stats["nodes_by_level"][0] == 1
    assert stats["nodes_by_level"][1] == 2
    db.close()
