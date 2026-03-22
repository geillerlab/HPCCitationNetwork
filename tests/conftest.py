"""Shared test fixtures for the citation network test suite."""

import pytest

from src.data.storage import CitationDB


@pytest.fixture
def tmp_db(tmp_path):
    """Create a temporary CitationDB for testing."""
    db_path = tmp_path / "test_citations.db"
    db = CitationDB(db_path)
    yield db
    db.close()


@pytest.fixture
def sample_paper():
    """A sample paper dict matching OpenAlex extract format."""
    return {
        "openalex_id": "https://openalex.org/W1234567890",
        "doi": "https://doi.org/10.1234/test.2024",
        "title": "A Test Paper About Hippocampal Attractors",
        "publication_year": 2024,
        "first_author": "Smith, J.",
        "authors": ["Smith, J.", "Jones, A."],
        "journal": "Journal of Neuroscience",
        "cited_by_count": 42,
        "type": "article",
        "abstract": "We propose a model of hippocampal computation.",
        "concepts": [{"name": "Hippocampus", "score": 0.9}],
        "topics": [{"name": "Neural circuits", "score": 0.8}],
    }


@pytest.fixture
def sample_openalex_response():
    """A minimal OpenAlex API response for a work."""
    return {
        "id": "https://openalex.org/W2741809807",
        "doi": "https://doi.org/10.1038/nature09633",
        "title": "Preplay of future place cell sequences by hippocampal cellular assemblies",
        "publication_year": 2011,
        "authorships": [
            {"author": {"display_name": "George Dragoi"}},
            {"author": {"display_name": "Susumu Tonegawa"}},
        ],
        "primary_location": {
            "source": {"display_name": "Nature"},
        },
        "cited_by_count": 350,
        "type": "article",
        "abstract": None,
        "referenced_works": [
            "https://openalex.org/W100",
            "https://openalex.org/W200",
        ],
        "concepts": [
            {"display_name": "Hippocampus", "score": 0.95},
            {"display_name": "Place cell", "score": 0.88},
        ],
        "topics": [
            {"display_name": "Hippocampal memory", "score": 0.9},
        ],
    }
