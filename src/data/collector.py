"""Snowball sampling collector for building citation networks."""

import logging
import time
from typing import Any

from src.data.openalex_client import OpenAlexClient
from src.data.seed_import import parse_seed_papers
from src.data.storage import CitationDB

logger = logging.getLogger(__name__)


class SnowballCollector:
    """Orchestrates snowball sampling from seed papers.

    Args:
        client: OpenAlex API client.
        db: Citation database for storage.
    """

    def __init__(self, client: OpenAlexClient, db: CitationDB) -> None:
        self.client = client
        self.db = db

    def import_seeds(self, doc_text: str) -> dict[str, int]:
        """Parse DOIs from Google Doc, resolve via OpenAlex, store as seeds.

        Args:
            doc_text: Full text of the Google Doc.

        Returns:
            Stats dict: {resolved, failed, total}.
        """
        papers = parse_seed_papers(doc_text)
        resolved = 0
        failed = 0

        for i, p in enumerate(papers):
            doi = p["doi"]
            work = self.client.resolve_doi(doi)
            if work is None:
                logger.warning("Failed to resolve DOI: %s", doi)
                failed += 1
                continue

            meta = OpenAlexClient.extract_paper_metadata(work)
            meta["is_seed"] = True
            meta["seed_category"] = p["seed_category"]
            meta["snowball_level"] = 0

            self.db.upsert_paper(meta)

            # Store references as citation edges
            ref_edges = [
                (meta["openalex_id"], ref_id)
                for ref_id in meta.get("referenced_works", [])
                if ref_id
            ]
            if ref_edges:
                self.db.add_citations_bulk(ref_edges)

            resolved += 1
            if (i + 1) % 10 == 0:
                logger.info("Seed import: %d/%d processed", i + 1, len(papers))

        logger.info(
            "Seed import complete: %d resolved, %d failed, %d total",
            resolved, failed, len(papers),
        )
        return {"resolved": resolved, "failed": failed, "total": len(papers)}

    def collect_level(
        self,
        level: int,
        max_cited_by: int = 200,
        fetch_metadata: bool = True,
    ) -> dict[str, int]:
        """Collect one level of snowball sampling.

        For each paper at (level - 1), fetches:
        - References (papers it cites) — already stored during seed import for level 0
        - Cited-by papers (papers that cite it) — requires API call

        New papers are stored at the given snowball level.

        Args:
            level: The snowball level to collect (1 = direct neighbors of seeds).
            max_cited_by: Maximum cited-by papers to fetch per source paper.
            fetch_metadata: Whether to fetch full metadata for new papers.
                Set False for a faster, edges-only collection.

        Returns:
            Stats dict: {papers_processed, papers_added, citations_added, errors}.
        """
        source_papers = self.db.get_papers_at_level(level - 1)
        if not source_papers:
            logger.warning("No papers at level %d to snowball from", level - 1)
            return {"papers_processed": 0, "papers_added": 0, "citations_added": 0, "errors": 0}

        papers_added = 0
        citations_before = self.db.get_citation_count()
        errors = 0

        for i, paper in enumerate(source_papers):
            oa_id = paper["openalex_id"]
            try:
                # Fetch cited-by papers
                cited_by_ids = self.client.get_cited_by(
                    oa_id, max_results=max_cited_by
                )
                # Store citation edges: each citing paper → this paper
                if cited_by_ids:
                    edges = [(citer, oa_id) for citer in cited_by_ids]
                    self.db.add_citations_bulk(edges)

                # For references: if this paper's references aren't in citations yet,
                # fetch the work and store them
                refs = paper.get("referenced_works")
                if refs is None:
                    # Need to fetch the work to get references
                    work = self.client.get_work(oa_id)
                    if work:
                        refs = work.get("referenced_works", [])
                        ref_edges = [(oa_id, ref_id) for ref_id in refs if ref_id]
                        if ref_edges:
                            self.db.add_citations_bulk(ref_edges)

                # Store stub papers for all new IDs at this level
                all_neighbor_ids = set(cited_by_ids)
                if refs:
                    all_neighbor_ids.update(r for r in refs if r)

                for neighbor_id in all_neighbor_ids:
                    existing = self.db.get_paper(neighbor_id)
                    if existing is None:
                        # Store a stub — metadata will be fetched if needed
                        self.db.upsert_paper({
                            "openalex_id": neighbor_id,
                            "snowball_level": level,
                        })
                        papers_added += 1

            except Exception as e:
                logger.error("Error processing %s: %s", oa_id, e)
                errors += 1

            if (i + 1) % 10 == 0 or (i + 1) == len(source_papers):
                logger.info(
                    "Level %d: %d/%d papers processed, %d new papers",
                    level, i + 1, len(source_papers), papers_added,
                )

        citations_added = self.db.get_citation_count() - citations_before

        logger.info(
            "Level %d complete: %d processed, %d papers added, %d citations added, %d errors",
            level, len(source_papers), papers_added, citations_added, errors,
        )
        return {
            "papers_processed": len(source_papers),
            "papers_added": papers_added,
            "citations_added": citations_added,
            "errors": errors,
        }

    def fetch_metadata_for_stubs(self, batch_size: int = 100) -> dict[str, int]:
        """Fetch full metadata for papers that only have stub records.

        Stubs are papers with an openalex_id but no title (added during snowball).

        Args:
            batch_size: How many to fetch before logging progress.

        Returns:
            Stats dict: {fetched, failed, total}.
        """
        # Find stubs: papers with no title
        rows = self.db.conn.execute(
            "SELECT openalex_id FROM papers WHERE title IS NULL OR title = ''"
        ).fetchall()
        stub_ids = [r["openalex_id"] for r in rows]

        if not stub_ids:
            logger.info("No stub papers to fetch metadata for")
            return {"fetched": 0, "failed": 0, "total": 0}

        fetched = 0
        failed = 0
        for i, oa_id in enumerate(stub_ids):
            work = self.client.get_work(oa_id)
            if work:
                meta = OpenAlexClient.extract_paper_metadata(work)
                # Preserve existing snowball_level and seed status
                existing = self.db.get_paper(oa_id)
                if existing:
                    meta["snowball_level"] = existing["snowball_level"]
                    meta["is_seed"] = existing["is_seed"]
                    meta["seed_category"] = existing.get("seed_category")
                self.db.upsert_paper(meta)
                fetched += 1
            else:
                failed += 1

            if (i + 1) % batch_size == 0:
                logger.info("Metadata fetch: %d/%d done", i + 1, len(stub_ids))

        logger.info(
            "Metadata fetch complete: %d fetched, %d failed, %d total",
            fetched, failed, len(stub_ids),
        )
        return {"fetched": fetched, "failed": failed, "total": len(stub_ids)}

    def run(
        self,
        doc_text: str,
        max_level: int = 1,
        max_cited_by: int = 200,
    ) -> dict[str, Any]:
        """Run the full snowball pipeline: import seeds, collect levels.

        Args:
            doc_text: Google Doc text for seed parsing.
            max_level: Maximum snowball depth (1 = direct neighbors).
            max_cited_by: Max cited-by papers per source paper per level.

        Returns:
            Combined stats dict.
        """
        stats: dict[str, Any] = {}

        # Import seeds if not already done
        if not self.db.get_seed_papers():
            logger.info("Importing seed papers...")
            stats["seed_import"] = self.import_seeds(doc_text)
        else:
            logger.info("Seeds already imported (%d papers)", len(self.db.get_seed_papers()))

        # Collect each level
        for level in range(1, max_level + 1):
            logger.info("Starting level %d snowball...", level)
            stats[f"level_{level}"] = self.collect_level(
                level=level, max_cited_by=max_cited_by
            )

        # Summary
        stats["total_papers"] = self.db.get_paper_count()
        stats["total_citations"] = self.db.get_citation_count()
        logger.info(
            "Snowball complete: %d papers, %d citations",
            stats["total_papers"], stats["total_citations"],
        )
        return stats
