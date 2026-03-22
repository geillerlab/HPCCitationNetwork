"""Tests for batch OpenAlex operations and edge cases."""

import responses

from src.data.openalex_client import OpenAlexClient


@responses.activate
def test_get_works_batch():
    """Test batch fetching multiple works in one API call."""
    responses.add(
        responses.GET,
        "https://api.openalex.org/works",
        json={
            "results": [
                {"id": "https://openalex.org/W1", "doi": None, "title": "Paper 1",
                 "publication_year": 2020, "authorships": [], "primary_location": None,
                 "cited_by_count": 10, "type": "article", "abstract": None,
                 "referenced_works": [], "concepts": [], "topics": []},
                {"id": "https://openalex.org/W2", "doi": None, "title": "Paper 2",
                 "publication_year": 2021, "authorships": [], "primary_location": None,
                 "cited_by_count": 5, "type": "article", "abstract": None,
                 "referenced_works": [], "concepts": [], "topics": []},
            ],
        },
        status=200,
    )

    client = OpenAlexClient(rate_limit_delay=0)
    results = client.get_works_batch(["https://openalex.org/W1", "https://openalex.org/W2"])
    assert len(results) == 2
    assert results[0]["title"] == "Paper 1"


@responses.activate
def test_get_works_batch_empty():
    """Test batch fetch with empty list."""
    client = OpenAlexClient(rate_limit_delay=0)
    results = client.get_works_batch([])
    assert results == []


def test_extract_metadata_null_fields():
    """Test metadata extraction handles null primary_location and source."""
    work = {
        "id": "https://openalex.org/W1",
        "doi": None,
        "title": "Test",
        "publication_year": 2020,
        "authorships": [],
        "primary_location": None,
        "cited_by_count": 0,
        "type": "article",
        "abstract": None,
        "referenced_works": [],
        "concepts": [],
        "topics": [],
    }
    meta = OpenAlexClient.extract_paper_metadata(work)
    assert meta["journal"] == ""
    assert meta["first_author"] == "Unknown"
    assert meta["authors"] == []


def test_extract_metadata_source_null():
    """Test metadata extraction when source is null but location exists."""
    work = {
        "id": "https://openalex.org/W1",
        "doi": None,
        "title": "Test",
        "publication_year": 2020,
        "authorships": [{"author": {"display_name": "Smith"}}],
        "primary_location": {"source": None},
        "cited_by_count": 0,
        "type": "article",
        "abstract": None,
        "referenced_works": [],
        "concepts": [],
        "topics": [],
    }
    meta = OpenAlexClient.extract_paper_metadata(work)
    assert meta["journal"] == ""
    assert meta["first_author"] == "Smith"
