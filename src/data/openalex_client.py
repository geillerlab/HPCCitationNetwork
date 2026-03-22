"""OpenAlex API client for resolving DOIs and fetching citation data."""

import time
from typing import Any

import requests

BASE_URL = "https://api.openalex.org"


class OpenAlexClient:
    """Client for the OpenAlex API.

    Args:
        email: Contact email for the polite pool (higher rate limits).
        rate_limit_delay: Seconds between requests to avoid rate limiting.
    """

    def __init__(
        self,
        email: str | None = None,
        rate_limit_delay: float = 0.1,
    ) -> None:
        self.email = email
        self.rate_limit_delay = rate_limit_delay
        self.session = requests.Session()
        if email:
            self.session.params = {"mailto": email}  # type: ignore[assignment]

    def _get(self, url: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        """Make a GET request with rate limiting."""
        time.sleep(self.rate_limit_delay)
        response = self.session.get(url, params=params)
        response.raise_for_status()
        return response.json()

    def resolve_doi(self, doi: str) -> dict[str, Any] | None:
        """Resolve a DOI to an OpenAlex work record.

        Args:
            doi: A DOI string (e.g., '10.1038/nature09633' or full URL).

        Returns:
            OpenAlex work dict, or None if not found.
        """
        # Normalize DOI
        doi = doi.strip()
        if doi.startswith("http"):
            # Extract DOI from URL like https://doi.org/10.1038/...
            doi = doi.split("doi.org/")[-1]

        try:
            return self._get(f"{BASE_URL}/works/doi:{doi}")
        except requests.HTTPError as e:
            if e.response is not None and e.response.status_code == 404:
                return None
            raise

    def get_work(self, openalex_id: str) -> dict[str, Any] | None:
        """Fetch a work by its OpenAlex ID.

        Args:
            openalex_id: An OpenAlex ID (e.g., 'W2741809807' or full URL).

        Returns:
            OpenAlex work dict, or None if not found.
        """
        # Normalize ID
        if openalex_id.startswith("https://"):
            openalex_id = openalex_id.split("/")[-1]

        try:
            return self._get(f"{BASE_URL}/works/{openalex_id}")
        except requests.HTTPError as e:
            if e.response is not None and e.response.status_code == 404:
                return None
            raise

    def get_references(self, openalex_id: str) -> list[str]:
        """Get OpenAlex IDs of works referenced by this paper.

        Args:
            openalex_id: An OpenAlex work ID.

        Returns:
            List of OpenAlex IDs that this work cites.
        """
        work = self.get_work(openalex_id)
        if work is None:
            return []
        # referenced_works is a list of OpenAlex ID strings
        return [ref for ref in work.get("referenced_works", []) if ref]

    def get_cited_by(
        self,
        openalex_id: str,
        per_page: int = 200,
        max_results: int | None = None,
    ) -> list[str]:
        """Get OpenAlex IDs of works that cite this paper.

        Args:
            openalex_id: An OpenAlex work ID.
            per_page: Results per API page (max 200).
            max_results: Maximum total results to return. None = all.

        Returns:
            List of OpenAlex IDs that cite this work.
        """
        if openalex_id.startswith("https://"):
            openalex_id = openalex_id.split("/")[-1]

        cited_by_ids: list[str] = []
        cursor = "*"

        while True:
            params: dict[str, Any] = {
                "filter": f"cites:{openalex_id}",
                "per-page": per_page,
                "cursor": cursor,
                "select": "id",
            }
            result = self._get(f"{BASE_URL}/works", params=params)
            works = result.get("results", [])
            if not works:
                break

            cited_by_ids.extend(w["id"] for w in works)

            if max_results and len(cited_by_ids) >= max_results:
                cited_by_ids = cited_by_ids[:max_results]
                break

            cursor = result.get("meta", {}).get("next_cursor")
            if not cursor:
                break

        return cited_by_ids

    def get_works_batch(self, openalex_ids: list[str], per_page: int = 50) -> list[dict[str, Any]]:
        """Fetch multiple works in a single API call using pipe-separated IDs.

        OpenAlex supports filtering by multiple IDs: filter=openalex:W1|W2|W3
        This is much faster than individual get_work() calls.

        Args:
            openalex_ids: List of OpenAlex IDs (full URLs or short form).
            per_page: Results per page (max 50 for filter queries).

        Returns:
            List of OpenAlex work dicts.
        """
        if not openalex_ids:
            return []

        results = []
        # Process in chunks of 50 (OpenAlex limit for pipe-separated filters)
        for i in range(0, len(openalex_ids), per_page):
            chunk = openalex_ids[i:i + per_page]
            # Build pipe-separated filter
            id_filter = "|".join(chunk)
            params: dict[str, Any] = {
                "filter": f"openalex:{id_filter}",
                "per-page": per_page,
            }
            resp = self._get(f"{BASE_URL}/works", params=params)
            results.extend(resp.get("results", []))

        return results

    def search_works(
        self,
        query: str,
        per_page: int = 25,
        max_results: int = 25,
    ) -> list[dict[str, Any]]:
        """Search for works by keyword query.

        Args:
            query: Search query string.
            per_page: Results per page.
            max_results: Maximum results to return.

        Returns:
            List of OpenAlex work dicts.
        """
        params: dict[str, Any] = {
            "search": query,
            "per-page": min(per_page, max_results),
        }
        result = self._get(f"{BASE_URL}/works", params=params)
        return result.get("results", [])[:max_results]

    @staticmethod
    def extract_paper_metadata(work: dict[str, Any]) -> dict[str, Any]:
        """Extract key metadata from an OpenAlex work record.

        Args:
            work: An OpenAlex work dict.

        Returns:
            Simplified dict with key fields.
        """
        authorships = work.get("authorships", [])
        authors = [
            a.get("author", {}).get("display_name", "Unknown")
            for a in authorships
        ]

        return {
            "openalex_id": work.get("id", ""),
            "doi": work.get("doi", ""),
            "title": work.get("title", ""),
            "publication_year": work.get("publication_year"),
            "authors": authors,
            "first_author": authors[0] if authors else "Unknown",
            "journal": ((work.get("primary_location") or {}).get("source") or {}).get("display_name", ""),
            "cited_by_count": work.get("cited_by_count", 0),
            "type": work.get("type", ""),
            "abstract": work.get("abstract", ""),
            "referenced_works": [r for r in work.get("referenced_works", [])],
            "concepts": [
                {"name": c.get("display_name", ""), "score": c.get("score", 0)}
                for c in work.get("concepts", [])
            ],
            "topics": [
                {"name": t.get("display_name", ""), "score": t.get("score", 0)}
                for t in work.get("topics", [])
            ],
        }
