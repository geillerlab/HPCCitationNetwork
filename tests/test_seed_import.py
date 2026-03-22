"""Tests for the DOI parser / seed import module."""

from src.data.seed_import import extract_dois_from_text, parse_seed_papers, _clean_doi


SAMPLE_DOC_TEXT = """
Big bucket o' models/reviews

Attractors - general

Khona, M., Fiete, I.R. Attractor and integrator networks in the brain. Nat Rev Neurosci 23, 744–766 (2022). https://doi.org/10.1038/s41583-022-00642-0

Point Attractors (=discrete memories)

Marr D. Simple memory: a theory for archicortex. Phil Trans R Soc Lond B. 1971;262(841):23-81. https://doi.org/10.1098/rstb.1971.0078

Hopfield JJ. Neural networks and physical systems. Proc Natl Acad Sci. 1982;79(8):2554-8. https://doi.org/10.1073/pnas.79.8.2554

Continuous attractors (HPC store relational map)

Samsonovich A, McNaughton BL. Path integration model. J Neurosci. 1997. https://doi.org/10.1523/JNEUROSCI.17-15-05900.1997

Spalla et al. Continuous attractors for dynamic memories eLife. https://doi.org/10.7554/eLife.69499

Sequences (intrinsic dynamics/reservoir)

Levy WB. A sequence predicting CA3. Hippocampus. 1996. https://doi.org/10.1002/(SICI)1098-1063(1996)6:6<579::AID-HIPO3>3.0.CO;2-C

Spalla et al. (cross-listed) https://doi.org/10.7554/eLife.69499

Successor Representation

Dayan P. Improving generalization. Neural Comput. 1993. https://doi.org/10.1162/neco.1993.5.4.613

BTSP (learning mechanism)

Bittner KC et al. Science. 2017. https://doi.org/10.1126/science.aan3846

Bespoke

O'Keefe J, Burgess N. Nature. 1996. DOI: 10.1038/381425a0
"""


def test_extract_dois_basic():
    """Test basic DOI extraction with categories."""
    results = extract_dois_from_text(SAMPLE_DOC_TEXT)
    assert len(results) >= 7

    # Build lookup by DOI
    doi_to_cats = {r["doi"]: r["categories"] for r in results}

    assert "general_attractor" in doi_to_cats.get("10.1038/s41583-022-00642-0", [])
    assert "point_attractor" in doi_to_cats.get("10.1098/rstb.1971.0078", [])
    assert "point_attractor" in doi_to_cats.get("10.1073/pnas.79.8.2554", [])
    assert "continuous_attractor" in doi_to_cats.get("10.1523/JNEUROSCI.17-15-05900.1997", [])
    assert "successor_representation" in doi_to_cats.get("10.1162/neco.1993.5.4.613", [])
    assert "btsp" in doi_to_cats.get("10.1126/science.aan3846", [])
    assert "bespoke" in doi_to_cats.get("10.1038/381425a0", [])


def test_cross_listing():
    """Test that papers appearing in multiple sections get all categories."""
    results = extract_dois_from_text(SAMPLE_DOC_TEXT)
    doi_to_cats = {r["doi"]: r["categories"] for r in results}

    # Spalla appears under both continuous_attractor and sequence
    spalla_cats = doi_to_cats.get("10.7554/eLife.69499", [])
    assert "continuous_attractor" in spalla_cats
    assert "sequence" in spalla_cats


def test_extract_deduplicates():
    """Test that duplicate DOIs are merged, not duplicated."""
    text = """
    Point Attractors
    https://doi.org/10.1234/test1
    https://doi.org/10.1234/test1
    """
    results = extract_dois_from_text(text)
    assert len(results) == 1


def test_clean_doi_trailing_punctuation():
    """Test DOI cleaning removes trailing punctuation."""
    assert _clean_doi("10.1038/nature09633.") == "10.1038/nature09633"
    assert _clean_doi("10.1038/nature09633,") == "10.1038/nature09633"


def test_clean_doi_preserves_balanced_parens():
    """Test that DOIs with balanced parentheses are preserved."""
    # Old-style Wiley DOIs have parens
    doi = "10.1002/(SICI)1098-1063(1996)6:6<579::AID-HIPO3>3.0.CO;2-C"
    assert _clean_doi(doi) == doi


def test_clean_doi_strips_unbalanced_trailing_paren():
    """Test that unbalanced trailing paren is stripped."""
    # This happens when the DOI is inside parentheses in text
    assert _clean_doi("10.1038/nature09633)") == "10.1038/nature09633"
    # But balanced parens should be kept
    assert _clean_doi("10.1016/0166-2236(87)90011-7") == "10.1016/0166-2236(87)90011-7"


def test_clean_doi_invalid():
    """Test that invalid DOIs return None."""
    assert _clean_doi("not-a-doi") is None
    assert _clean_doi("10.") is None


def test_parse_seed_papers():
    """Test the full parsing pipeline with manual supplements."""
    papers = parse_seed_papers(SAMPLE_DOC_TEXT)
    assert all(p["is_seed"] for p in papers)
    assert all("doi" in p for p in papers)
    assert all("seed_category" in p for p in papers)
    assert all("seed_categories" in p for p in papers)

    # Should include manual DOIs (Krotov, Ramsauer, etc.)
    all_dois = {p["doi"] for p in papers}
    assert "10.48550/arxiv.1606.01164" in all_dois  # Krotov manual
    assert "10.48550/arxiv.2008.02217" in all_dois  # Ramsauer manual


def test_parse_seed_papers_no_manual():
    """Test parsing without manual supplements."""
    papers = parse_seed_papers(SAMPLE_DOC_TEXT, include_manual=False)
    all_dois = {p["doi"] for p in papers}
    assert "10.48550/arxiv.1606.01164" not in all_dois  # No Krotov


def test_doi_formats():
    """Test various DOI formats are handled."""
    text = """
    Point Attractors
    https://doi.org/10.1038/nn.4062
    doi: 10.1126/science.aan3846
    DOI: 10.1002/hipo.22355
    https://10.7554/eLife.87055
    """
    results = extract_dois_from_text(text)
    dois = {r["doi"] for r in results}
    assert "10.1038/nn.4062" in dois
    assert "10.1126/science.aan3846" in dois
    assert "10.1002/hipo.22355" in dois


def test_general_attractor_category():
    """Test that papers under 'Attractors - general' get the right category."""
    results = extract_dois_from_text(SAMPLE_DOC_TEXT)
    doi_to_cats = {r["doi"]: r["categories"] for r in results}
    assert "general_attractor" in doi_to_cats.get("10.1038/s41583-022-00642-0", [])
