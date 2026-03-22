"""Tests for the SQLite storage layer."""

from src.data.storage import CitationDB


def test_upsert_and_get_paper(tmp_db, sample_paper):
    """Test inserting and retrieving a paper."""
    tmp_db.upsert_paper(sample_paper)
    result = tmp_db.get_paper("https://openalex.org/W1234567890")
    assert result is not None
    assert result["title"] == "A Test Paper About Hippocampal Attractors"
    assert result["first_author"] == "Smith, J."


def test_upsert_preserves_seed_status(tmp_db, sample_paper):
    """Test that upserting a non-seed paper doesn't overwrite seed status."""
    # First insert as seed
    sample_paper["is_seed"] = True
    sample_paper["seed_category"] = "point_attractor"
    tmp_db.upsert_paper(sample_paper)

    # Re-insert without seed flag
    non_seed = {**sample_paper, "is_seed": False, "seed_category": None}
    tmp_db.upsert_paper(non_seed)

    result = tmp_db.get_paper("https://openalex.org/W1234567890")
    assert result["is_seed"] == 1  # Still a seed
    assert result["seed_category"] == "point_attractor"


def test_upsert_keeps_lower_snowball_level(tmp_db, sample_paper):
    """Test that upsert keeps the lower snowball level."""
    sample_paper["snowball_level"] = 2
    tmp_db.upsert_paper(sample_paper)

    sample_paper["snowball_level"] = 1
    tmp_db.upsert_paper(sample_paper)

    result = tmp_db.get_paper("https://openalex.org/W1234567890")
    assert result["snowball_level"] == 1

    # Higher level shouldn't overwrite
    sample_paper["snowball_level"] = 3
    tmp_db.upsert_paper(sample_paper)
    result = tmp_db.get_paper("https://openalex.org/W1234567890")
    assert result["snowball_level"] == 1


def test_add_and_get_citations(tmp_db):
    """Test adding and retrieving citation edges."""
    tmp_db.add_citation("W100", "W200")
    tmp_db.add_citation("W100", "W300")
    tmp_db.add_citation("W200", "W300")

    edges = tmp_db.get_all_citations()
    assert len(edges) == 3
    assert ("W100", "W200") in edges


def test_add_citations_bulk(tmp_db):
    """Test bulk citation insertion."""
    edges = [("W1", "W2"), ("W1", "W3"), ("W2", "W3")]
    tmp_db.add_citations_bulk(edges)
    assert tmp_db.get_citation_count() == 3


def test_duplicate_citations_ignored(tmp_db):
    """Test that duplicate citation edges are silently ignored."""
    tmp_db.add_citation("W100", "W200")
    tmp_db.add_citation("W100", "W200")  # duplicate
    assert tmp_db.get_citation_count() == 1


def test_get_seed_papers(tmp_db, sample_paper):
    """Test retrieving seed papers."""
    sample_paper["is_seed"] = True
    sample_paper["seed_category"] = "continuous_attractor"
    tmp_db.upsert_paper(sample_paper)

    non_seed = {
        "openalex_id": "https://openalex.org/W999",
        "title": "Non-seed paper",
        "is_seed": False,
    }
    tmp_db.upsert_paper(non_seed)

    seeds = tmp_db.get_seed_papers()
    assert len(seeds) == 1
    assert seeds[0]["seed_category"] == "continuous_attractor"


def test_get_paper_by_doi(tmp_db, sample_paper):
    """Test looking up a paper by DOI."""
    tmp_db.upsert_paper(sample_paper)
    result = tmp_db.get_paper_by_doi("https://doi.org/10.1234/test.2024")
    assert result is not None
    assert result["title"] == "A Test Paper About Hippocampal Attractors"


def test_paper_count_and_citation_count(tmp_db, sample_paper):
    """Test count methods."""
    assert tmp_db.get_paper_count() == 0
    tmp_db.upsert_paper(sample_paper)
    assert tmp_db.get_paper_count() == 1

    assert tmp_db.get_citation_count() == 0
    tmp_db.add_citation("W1", "W2")
    assert tmp_db.get_citation_count() == 1


def test_context_manager(tmp_path):
    """Test using CitationDB as a context manager."""
    db_path = tmp_path / "ctx_test.db"
    with CitationDB(db_path) as db:
        db.upsert_paper({
            "openalex_id": "W1",
            "title": "Test",
        })
        assert db.get_paper_count() == 1
