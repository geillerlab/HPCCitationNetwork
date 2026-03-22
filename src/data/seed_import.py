"""Parse DOIs from the Google Doc and import as seed papers."""

import re
from typing import Any


# Categories as they appear in the Google Doc, mapped to short labels.
# Order matters: more specific headers should match before general ones.
CATEGORY_HEADERS = [
    ("Attractors - general", "general_attractor"),
    ("Attractors \\- general", "general_attractor"),  # escaped dash variant
    ("Big bucket o' models/reviews", "general_attractor"),
    ("Point Attractors", "point_attractor"),
    ("Continuous attractors", "continuous_attractor"),
    ("Sequences", "sequence"),
    ("Successor Representation", "successor_representation"),
    ("BTSP", "btsp"),
    ("Bespoke", "bespoke"),
    ("Evidence for autonomous dynamics", "autonomous_dynamics"),
]

# Regex to match DOIs in various formats.
# Key change: allow parentheses *within* the DOI (common in old Wiley/Elsevier DOIs)
# but strip unbalanced trailing parens.
DOI_PATTERN = re.compile(
    r'(?:https?://(?:dx\.)?doi\.org/|doi:\s*|DOI:\s*)?'
    r'(10\.\d{4,9}/[^\s\]>,\"\']+)',
    re.IGNORECASE,
)


def extract_dois_from_text(text: str) -> list[dict[str, list[str]]]:
    """Extract DOIs and their associated categories from the Google Doc text.

    Papers appearing under multiple category headers get ALL categories
    (cross-listing). Deduplication merges categories.

    Args:
        text: The full text of the Google Doc.

    Returns:
        List of dicts with 'doi' and 'categories' (list of str) keys.
    """
    # First pass: collect (doi, category) pairs — allow duplicates
    doi_categories: dict[str, list[str]] = {}
    current_category = "uncategorized"

    for line in text.split("\n"):
        # Check if this line is a category header
        for header, label in CATEGORY_HEADERS:
            if header.lower() in line.lower() and len(line) < 200:
                current_category = label
                break

        # Find all DOIs on this line
        for match in DOI_PATTERN.finditer(line):
            doi = _clean_doi(match.group(1))
            if doi:
                if doi not in doi_categories:
                    doi_categories[doi] = []
                if current_category not in doi_categories[doi]:
                    doi_categories[doi].append(current_category)

    return [
        {"doi": doi, "categories": cats}
        for doi, cats in doi_categories.items()
    ]


def _clean_doi(doi: str) -> str | None:
    """Clean up a raw DOI string, handling parentheses correctly.

    Args:
        doi: Raw DOI string from regex match.

    Returns:
        Cleaned DOI, or None if invalid.
    """
    # Remove trailing punctuation that might have been captured
    doi = doi.rstrip(".,;:\"'")

    # Remove URL-encoded characters at the end
    doi = re.sub(r'%[0-9A-Fa-f]{2}$', '', doi)

    # Balance parentheses: some DOIs legitimately contain parens
    # e.g., 10.1002/(SICI)1098-1063(1999)9:4<481::AID-HIPO14>3.0.CO;2-S
    # Strip only unbalanced trailing close-parens
    open_count = doi.count("(")
    close_count = doi.count(")")
    while close_count > open_count and doi.endswith(")"):
        doi = doi[:-1]
        close_count -= 1

    # Also strip unbalanced trailing brackets/braces
    for open_ch, close_ch in [("[", "]"), ("{", "}")]:
        o = doi.count(open_ch)
        c = doi.count(close_ch)
        while c > o and doi.endswith(close_ch):
            doi = doi[:-1]
            c -= 1

    # Handle angle brackets in DOIs (e.g., <481::AID-HIPO14>)
    # These are part of old-style DOIs; keep them
    # But strip trailing > if unbalanced
    o = doi.count("<")
    c = doi.count(">")
    while c > o and doi.endswith(">"):
        doi = doi[:-1]
        c -= 1

    # Basic validation: must start with 10.
    if not doi.startswith("10."):
        return None

    # Must have something after the prefix
    parts = doi.split("/", 1)
    if len(parts) < 2 or not parts[1]:
        return None

    return doi


def parse_seed_papers(
    doc_text: str,
    include_manual: bool = True,
) -> list[dict[str, Any]]:
    """Parse the Google Doc into a list of seed paper records.

    This is the main entry point. Extracts DOIs with categories,
    handles cross-listing, merges manual supplements (papers with
    non-DOI URLs and explicit cross-listing overrides).

    Args:
        doc_text: Full text of the Google Doc.
        include_manual: Whether to include manual DOI supplements and
            cross-listing overrides from manual_seeds.py.

    Returns:
        List of dicts with 'doi', 'seed_categories' (list), and 'is_seed' keys.
    """
    raw = extract_dois_from_text(doc_text)

    # Build doi -> categories dict for merging
    doi_cats: dict[str, list[str]] = {}
    for item in raw:
        doi_cats[item["doi"]] = list(item["categories"])

    if include_manual:
        from src.data.manual_seeds import MANUAL_DOIS, CROSS_LISTINGS, DOI_CORRECTIONS

        # Apply DOI corrections (truncated or incorrect DOIs in the doc)
        for wrong_doi, correct_doi in DOI_CORRECTIONS.items():
            if wrong_doi in doi_cats:
                cats = doi_cats.pop(wrong_doi)
                if correct_doi in doi_cats:
                    for c in cats:
                        if c not in doi_cats[correct_doi]:
                            doi_cats[correct_doi].append(c)
                else:
                    doi_cats[correct_doi] = cats

        # Add manual DOIs (papers not found by parser)
        for entry in MANUAL_DOIS:
            doi = entry["doi"]
            cats = entry["categories"]
            if doi not in doi_cats:
                doi_cats[doi] = list(cats)
            else:
                for c in cats:
                    if c not in doi_cats[doi]:
                        doi_cats[doi].append(c)

        # Apply cross-listing overrides
        for doi, extra_cats in CROSS_LISTINGS.items():
            if doi in doi_cats:
                for c in extra_cats:
                    if c not in doi_cats[doi]:
                        doi_cats[doi].append(c)

    return [
        {
            "doi": doi,
            "seed_categories": cats,
            "seed_category": cats[0],  # primary category
            "is_seed": True,
        }
        for doi, cats in doi_cats.items()
    ]


def summarize_seed_papers(papers: list[dict[str, Any]]) -> str:
    """Create a summary string of parsed seed papers by category.

    Args:
        papers: Output of parse_seed_papers.

    Returns:
        Human-readable summary string.
    """
    from collections import Counter

    # Count primary categories
    primary_counts = Counter(p["seed_category"] for p in papers)
    # Count all categories (including cross-listings)
    all_counts: Counter[str] = Counter()
    cross_listed = 0
    for p in papers:
        cats = p.get("seed_categories", [p["seed_category"]])
        for c in cats:
            all_counts[c] += 1
        if len(cats) > 1:
            cross_listed += 1

    lines = [f"Total unique seed papers: {len(papers)}"]
    lines.append(f"Cross-listed papers: {cross_listed}")
    lines.append("")
    lines.append("Category breakdown (primary / including cross-listings):")
    for cat in sorted(all_counts.keys()):
        primary = primary_counts.get(cat, 0)
        total = all_counts[cat]
        if total > primary:
            lines.append(f"  {cat:<30s} {primary:>3d} / {total}")
        else:
            lines.append(f"  {cat:<30s} {primary:>3d}")

    return "\n".join(lines)
