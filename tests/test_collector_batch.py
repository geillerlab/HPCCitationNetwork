"""Tests for batch metadata fetching in the collector."""

import responses

from src.data.collector import SnowballCollector
from src.data.openalex_client import OpenAlexClient
from src.data.storage import CitationDB


def _make_batch_response(ids):
    """Create a mock batch API response for given IDs."""
    return {
        "results": [
            {
                "id": oa_id,
                "doi": None,
                "title": f"Paper {oa_id.split('/')[-1]}",
                "publication_year": 2020,
                "authorships": [{"author": {"display_name": f"Author {oa_id.split('/')[-1]}"}}],
                "primary_location": {"source": {"display_name": "Test Journal"}},
                "cited_by_count": 10,
                "type": "article",
                "abstract": "Test abstract",
                "referenced_works": [],
                "concepts": [],
                "topics": [],
            }
            for oa_id in ids
        ],
    }


@responses.activate
def test_batch_store_papers(tmp_path):
    """Test that _batch_store_papers fetches and stores full metadata."""
    db = CitationDB(tmp_path / "test.db")
    client = OpenAlexClient(rate_limit_delay=0)
    collector = SnowballCollector(client, db)

    new_ids = ["https://openalex.org/W100", "https://openalex.org/W200"]

    responses.add(
        responses.GET,
        "https://api.openalex.org/works",
        json=_make_batch_response(new_ids),
        status=200,
    )

    collector._batch_store_papers(new_ids, level=1)

    # Papers should have full metadata, not stubs
    p100 = db.get_paper("https://openalex.org/W100")
    assert p100 is not None
    assert p100["title"] == "Paper W100"
    assert p100["first_author"] == "Author W100"
    assert p100["snowball_level"] == 1

    p200 = db.get_paper("https://openalex.org/W200")
    assert p200 is not None
    assert p200["title"] == "Paper W200"
    db.close()


@responses.activate
def test_batch_store_falls_back_to_stubs(tmp_path):
    """Test that batch store creates stubs when API fails."""
    db = CitationDB(tmp_path / "test.db")
    client = OpenAlexClient(rate_limit_delay=0)
    collector = SnowballCollector(client, db)

    new_ids = ["https://openalex.org/W100"]

    # Simulate API failure
    responses.add(
        responses.GET,
        "https://api.openalex.org/works",
        json={"error": "server error"},
        status=500,
    )

    collector._batch_store_papers(new_ids, level=1)

    # Should still have a stub record
    p100 = db.get_paper("https://openalex.org/W100")
    assert p100 is not None
    assert p100["snowball_level"] == 1
    # But no title (stub)
    assert p100.get("title") is None or p100["title"] == ""
    db.close()


@responses.activate
def test_batch_store_partial_response(tmp_path):
    """Test that papers missing from batch response get stored as stubs."""
    db = CitationDB(tmp_path / "test.db")
    client = OpenAlexClient(rate_limit_delay=0)
    collector = SnowballCollector(client, db)

    # Request 2, but API only returns 1
    new_ids = ["https://openalex.org/W100", "https://openalex.org/W200"]

    responses.add(
        responses.GET,
        "https://api.openalex.org/works",
        json=_make_batch_response(["https://openalex.org/W100"]),
        status=200,
    )

    collector._batch_store_papers(new_ids, level=1)

    # W100 has metadata
    p100 = db.get_paper("https://openalex.org/W100")
    assert p100["title"] == "Paper W100"

    # W200 is a stub
    p200 = db.get_paper("https://openalex.org/W200")
    assert p200 is not None
    assert p200["snowball_level"] == 1
    db.close()
