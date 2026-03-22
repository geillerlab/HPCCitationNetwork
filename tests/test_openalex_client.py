"""Tests for the OpenAlex API client."""

import responses

from src.data.openalex_client import OpenAlexClient


@responses.activate
def test_resolve_doi(sample_openalex_response):
    """Test resolving a DOI to an OpenAlex work."""
    responses.add(
        responses.GET,
        "https://api.openalex.org/works/doi:10.1038/nature09633",
        json=sample_openalex_response,
        status=200,
    )
    client = OpenAlexClient(rate_limit_delay=0)
    result = client.resolve_doi("10.1038/nature09633")
    assert result is not None
    assert result["id"] == "https://openalex.org/W2741809807"


@responses.activate
def test_resolve_doi_from_url(sample_openalex_response):
    """Test resolving a DOI from a full URL."""
    responses.add(
        responses.GET,
        "https://api.openalex.org/works/doi:10.1038/nature09633",
        json=sample_openalex_response,
        status=200,
    )
    client = OpenAlexClient(rate_limit_delay=0)
    result = client.resolve_doi("https://doi.org/10.1038/nature09633")
    assert result is not None
    assert result["title"] == "Preplay of future place cell sequences by hippocampal cellular assemblies"


@responses.activate
def test_resolve_doi_not_found():
    """Test that a missing DOI returns None."""
    responses.add(
        responses.GET,
        "https://api.openalex.org/works/doi:10.1234/nonexistent",
        json={"error": "not found"},
        status=404,
    )
    client = OpenAlexClient(rate_limit_delay=0)
    result = client.resolve_doi("10.1234/nonexistent")
    assert result is None


@responses.activate
def test_get_references(sample_openalex_response):
    """Test fetching references for a work."""
    responses.add(
        responses.GET,
        "https://api.openalex.org/works/W2741809807",
        json=sample_openalex_response,
        status=200,
    )
    client = OpenAlexClient(rate_limit_delay=0)
    refs = client.get_references("W2741809807")
    assert len(refs) == 2
    assert "https://openalex.org/W100" in refs


@responses.activate
def test_get_cited_by():
    """Test fetching papers that cite a work."""
    responses.add(
        responses.GET,
        "https://api.openalex.org/works",
        json={
            "results": [
                {"id": "https://openalex.org/W300"},
                {"id": "https://openalex.org/W400"},
            ],
            "meta": {"next_cursor": None},
        },
        status=200,
    )
    client = OpenAlexClient(rate_limit_delay=0)
    cited_by = client.get_cited_by("W2741809807")
    assert len(cited_by) == 2


@responses.activate
def test_extract_paper_metadata(sample_openalex_response):
    """Test metadata extraction from OpenAlex work."""
    meta = OpenAlexClient.extract_paper_metadata(sample_openalex_response)
    assert meta["openalex_id"] == "https://openalex.org/W2741809807"
    assert meta["first_author"] == "George Dragoi"
    assert meta["publication_year"] == 2011
    assert meta["journal"] == "Nature"
    assert len(meta["referenced_works"]) == 2
    assert meta["concepts"][0]["name"] == "Hippocampus"
