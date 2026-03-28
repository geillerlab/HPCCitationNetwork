"""Parse EndNote XML exports into seed paper records for citation network building."""

import logging
import xml.etree.ElementTree as ET
from typing import Any

logger = logging.getLogger(__name__)


def _extract_style_text(element: ET.Element | None) -> str:
    """Extract text from an element that may contain <style> wrappers.

    EndNote XML wraps most text content in ``<style>`` elements.
    This helper extracts the concatenated text whether or not the
    wrapper is present.

    Args:
        element: An XML element, or *None*.

    Returns:
        The stripped text content, or ``""`` if the element is *None*
        or contains no text.
    """
    if element is None:
        return ""
    # Try <style> children first
    style_texts = [s.text.strip() for s in element.findall("style") if s.text]
    if style_texts:
        return " ".join(style_texts)
    # Fall back to direct text
    return (element.text or "").strip()


def _extract_category(record: ET.Element) -> str:
    """Determine the seed category for a record.

    Checks several fields in priority order.  The first non-empty value
    wins; if nothing is found the record is marked ``"uncategorized"``.

    Priority:
        1. ``<label>``
        2. ``<custom1>`` … ``<custom7>``
        3. ``<research-notes>``
        4. First ``<keyword>``
    """
    # 1. <label>
    label = _extract_style_text(record.find("label"))
    if label:
        return _normalise_category(label)

    # 2. <custom1> … <custom7>
    for i in range(1, 8):
        custom = _extract_style_text(record.find(f"custom{i}"))
        if custom:
            return _normalise_category(custom)

    # 3. <research-notes>
    notes = _extract_style_text(record.find("research-notes"))
    if notes:
        return _normalise_category(notes)

    # 4. First keyword
    keywords_el = record.find("keywords")
    if keywords_el is not None:
        first_kw = keywords_el.find("keyword")
        kw_text = _extract_style_text(first_kw)
        if kw_text:
            return _normalise_category(kw_text)

    return "uncategorized"


def _normalise_category(raw: str) -> str:
    """Turn a free-text category into a clean snake_case identifier."""
    # Lowercase, strip, collapse whitespace → underscores
    clean = raw.strip().lower()
    # Replace non-alphanumeric chars with underscores
    clean = "".join(c if c.isalnum() or c == "_" else "_" for c in clean)
    # Collapse consecutive underscores and strip leading/trailing
    while "__" in clean:
        clean = clean.replace("__", "_")
    return clean.strip("_") or "uncategorized"


def parse_endnote_xml(xml_bytes: bytes) -> list[dict[str, Any]]:
    """Parse an EndNote XML export into seed paper records.

    Args:
        xml_bytes: Raw bytes of the uploaded ``.xml`` file.

    Returns:
        List of dicts with keys:

        - ``doi`` (*str | None*) — the DOI, or *None* if not present.
        - ``title`` (*str*) — paper title.
        - ``authors`` (*list[str]*) — author names.
        - ``year`` (*int | None*) — publication year.
        - ``journal`` (*str*) — journal / source title.
        - ``seed_category`` (*str*) — primary category (snake_case).
        - ``seed_categories`` (*list[str]*) — all categories.
        - ``is_seed`` (*bool*) — always ``True``.

    Raises:
        ET.ParseError: If *xml_bytes* is not valid XML.
    """
    root = ET.fromstring(xml_bytes)

    # Find all <record> elements (may be nested under <records>)
    records_el = root.findall(".//record")
    if not records_el:
        logger.warning("No <record> elements found in the XML")
        return []

    # First pass — extract raw records
    raw: list[dict[str, Any]] = []
    for rec in records_el:
        # DOI
        doi_el = rec.find(".//electronic-resource-num")
        doi = _extract_style_text(doi_el) or None
        if doi:
            # Clean DOI: strip URL prefix if present
            for prefix in ("https://doi.org/", "http://doi.org/",
                           "https://dx.doi.org/", "http://dx.doi.org/",
                           "doi:", "DOI:"):
                if doi.startswith(prefix):
                    doi = doi[len(prefix):]
            doi = doi.strip()

        # Title
        title_el = rec.find(".//titles/title")
        title = _extract_style_text(title_el)

        # Authors
        authors: list[str] = []
        authors_el = rec.find(".//contributors/authors")
        if authors_el is not None:
            for author_el in authors_el.findall("author"):
                name = _extract_style_text(author_el)
                if name:
                    authors.append(name)

        # Year
        year_el = rec.find(".//dates/year")
        year_str = _extract_style_text(year_el)
        year: int | None = None
        if year_str:
            try:
                year = int(year_str)
            except ValueError:
                pass

        # Journal
        journal_el = rec.find(".//periodical/full-title")
        if journal_el is None:
            journal_el = rec.find(".//secondary-title")
        journal = _extract_style_text(journal_el)

        # Category
        category = _extract_category(rec)

        if not title and not doi:
            logger.warning("Skipping record with no title and no DOI")
            continue

        raw.append({
            "doi": doi,
            "title": title,
            "authors": authors,
            "year": year,
            "journal": journal,
            "seed_category": category,
            "seed_categories": [category],
            "is_seed": True,
        })

    # Deduplicate by DOI (merge categories for duplicates)
    seen_dois: dict[str, int] = {}  # doi → index in result list
    result: list[dict[str, Any]] = []

    for entry in raw:
        doi = entry["doi"]
        if doi and doi in seen_dois:
            # Merge categories
            idx = seen_dois[doi]
            existing_cats = result[idx]["seed_categories"]
            new_cat = entry["seed_category"]
            if new_cat not in existing_cats:
                existing_cats.append(new_cat)
            logger.debug("Merged duplicate DOI %s, categories: %s", doi, existing_cats)
        else:
            if doi:
                seen_dois[doi] = len(result)
            result.append(entry)

    logger.info(
        "Parsed %d records from EndNote XML (%d with DOIs, %d unique categories)",
        len(result),
        sum(1 for r in result if r["doi"]),
        len(set(r["seed_category"] for r in result)),
    )
    return result
