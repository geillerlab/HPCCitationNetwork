"""Quick script to parse seed papers from the Google Doc text.

Run: uv run python scripts/parse_seeds.py
"""

import json
from pathlib import Path

from src.data.seed_import import parse_seed_papers, summarize_seed_papers

# The Google Doc text is passed in via stdin or loaded from a file
DOC_TEXT_PATH = Path(__file__).parent / "google_doc_text.txt"


def main() -> None:
    if not DOC_TEXT_PATH.exists():
        print(f"Please save the Google Doc text to {DOC_TEXT_PATH}")
        return

    text = DOC_TEXT_PATH.read_text()
    papers = parse_seed_papers(text)

    print(summarize_seed_papers(papers))
    print("\n--- Full list ---\n")

    for i, p in enumerate(papers, 1):
        print(f"{i:3d}. [{p['seed_category']:<25s}] {p['doi']}")

    # Save to JSON for review
    output_path = Path(__file__).parent / "parsed_seeds.json"
    with open(output_path, "w") as f:
        json.dump(papers, f, indent=2)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
