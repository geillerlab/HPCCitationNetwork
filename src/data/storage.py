"""SQLite storage layer for papers and citation edges."""

import json
import sqlite3
from pathlib import Path
from typing import Any


DEFAULT_DB_PATH = Path(__file__).parent.parent.parent / "data" / "processed" / "citations.db"


class CitationDB:
    """SQLite database for storing papers and citation relationships.

    Args:
        db_path: Path to the SQLite database file. Created if it doesn't exist.
    """

    def __init__(self, db_path: Path | str = DEFAULT_DB_PATH) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row
        self._create_tables()

    def _create_tables(self) -> None:
        """Create tables if they don't exist."""
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS papers (
                openalex_id TEXT PRIMARY KEY,
                doi TEXT,
                title TEXT,
                publication_year INTEGER,
                first_author TEXT,
                authors_json TEXT,
                journal TEXT,
                cited_by_count INTEGER,
                type TEXT,
                abstract TEXT,
                concepts_json TEXT,
                topics_json TEXT,
                is_seed INTEGER DEFAULT 0,
                seed_category TEXT,
                snowball_level INTEGER DEFAULT -1,
                paper_class TEXT
            );

            CREATE TABLE IF NOT EXISTS citations (
                citing_id TEXT NOT NULL,
                cited_id TEXT NOT NULL,
                PRIMARY KEY (citing_id, cited_id)
            );

            CREATE INDEX IF NOT EXISTS idx_citations_citing ON citations(citing_id);
            CREATE INDEX IF NOT EXISTS idx_citations_cited ON citations(cited_id);
            CREATE INDEX IF NOT EXISTS idx_papers_doi ON papers(doi);
            CREATE INDEX IF NOT EXISTS idx_papers_seed ON papers(is_seed);
        """)
        self.conn.commit()

    def upsert_paper(self, paper: dict[str, Any]) -> None:
        """Insert or update a paper record.

        Args:
            paper: Dict with paper metadata. Must include 'openalex_id'.
        """
        self.conn.execute("""
            INSERT INTO papers (
                openalex_id, doi, title, publication_year, first_author,
                authors_json, journal, cited_by_count, type, abstract,
                concepts_json, topics_json, is_seed, seed_category,
                snowball_level, paper_class
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(openalex_id) DO UPDATE SET
                doi = COALESCE(excluded.doi, doi),
                title = COALESCE(excluded.title, title),
                publication_year = COALESCE(excluded.publication_year, publication_year),
                first_author = COALESCE(excluded.first_author, first_author),
                authors_json = COALESCE(excluded.authors_json, authors_json),
                journal = COALESCE(excluded.journal, journal),
                cited_by_count = COALESCE(excluded.cited_by_count, cited_by_count),
                type = COALESCE(excluded.type, type),
                abstract = COALESCE(excluded.abstract, abstract),
                concepts_json = COALESCE(excluded.concepts_json, concepts_json),
                topics_json = COALESCE(excluded.topics_json, topics_json),
                is_seed = MAX(is_seed, excluded.is_seed),
                seed_category = COALESCE(excluded.seed_category, seed_category),
                snowball_level = CASE
                    WHEN excluded.snowball_level >= 0 AND (snowball_level < 0 OR excluded.snowball_level < snowball_level)
                    THEN excluded.snowball_level
                    ELSE snowball_level
                END,
                paper_class = COALESCE(excluded.paper_class, paper_class)
        """, (
            paper.get("openalex_id"),
            paper.get("doi"),
            paper.get("title"),
            paper.get("publication_year"),
            paper.get("first_author"),
            json.dumps(paper.get("authors", [])),
            paper.get("journal"),
            paper.get("cited_by_count"),
            paper.get("type"),
            paper.get("abstract"),
            json.dumps(paper.get("concepts", [])),
            json.dumps(paper.get("topics", [])),
            1 if paper.get("is_seed") else 0,
            paper.get("seed_category"),
            paper.get("snowball_level", -1),
            paper.get("paper_class"),
        ))
        self.conn.commit()

    def add_citation(self, citing_id: str, cited_id: str) -> None:
        """Add a citation edge (citing_id cites cited_id).

        Args:
            citing_id: OpenAlex ID of the citing paper.
            cited_id: OpenAlex ID of the cited paper.
        """
        self.conn.execute(
            "INSERT OR IGNORE INTO citations (citing_id, cited_id) VALUES (?, ?)",
            (citing_id, cited_id),
        )
        self.conn.commit()

    def add_citations_bulk(self, edges: list[tuple[str, str]]) -> None:
        """Add multiple citation edges at once.

        Args:
            edges: List of (citing_id, cited_id) tuples.
        """
        self.conn.executemany(
            "INSERT OR IGNORE INTO citations (citing_id, cited_id) VALUES (?, ?)",
            edges,
        )
        self.conn.commit()

    def get_paper(self, openalex_id: str) -> dict[str, Any] | None:
        """Get a paper by OpenAlex ID.

        Returns:
            Paper dict or None if not found.
        """
        row = self.conn.execute(
            "SELECT * FROM papers WHERE openalex_id = ?", (openalex_id,)
        ).fetchone()
        if row is None:
            return None
        return self._row_to_dict(row)

    def get_paper_by_doi(self, doi: str) -> dict[str, Any] | None:
        """Get a paper by DOI.

        Returns:
            Paper dict or None if not found.
        """
        row = self.conn.execute(
            "SELECT * FROM papers WHERE doi = ?", (doi,)
        ).fetchone()
        if row is None:
            return None
        return self._row_to_dict(row)

    def get_seed_papers(self) -> list[dict[str, Any]]:
        """Get all seed papers."""
        rows = self.conn.execute(
            "SELECT * FROM papers WHERE is_seed = 1 ORDER BY publication_year"
        ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def get_all_papers(self) -> list[dict[str, Any]]:
        """Get all papers in the database."""
        rows = self.conn.execute(
            "SELECT * FROM papers ORDER BY publication_year"
        ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def get_all_citations(self) -> list[tuple[str, str]]:
        """Get all citation edges as (citing_id, cited_id) tuples."""
        rows = self.conn.execute("SELECT citing_id, cited_id FROM citations").fetchall()
        return [(r["citing_id"], r["cited_id"]) for r in rows]

    def get_paper_count(self) -> int:
        """Get the total number of papers."""
        row = self.conn.execute("SELECT COUNT(*) as cnt FROM papers").fetchone()
        return row["cnt"] if row else 0

    def get_citation_count(self) -> int:
        """Get the total number of citation edges."""
        row = self.conn.execute("SELECT COUNT(*) as cnt FROM citations").fetchone()
        return row["cnt"] if row else 0

    def get_papers_at_level(self, level: int) -> list[dict[str, Any]]:
        """Get papers at a specific snowball level."""
        rows = self.conn.execute(
            "SELECT * FROM papers WHERE snowball_level = ? ORDER BY publication_year",
            (level,),
        ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def get_uncollected_paper_ids(self) -> list[str]:
        """Get OpenAlex IDs of papers that appear in citations but aren't in the papers table."""
        rows = self.conn.execute("""
            SELECT DISTINCT c.cited_id as id FROM citations c
            WHERE c.cited_id NOT IN (SELECT openalex_id FROM papers)
            UNION
            SELECT DISTINCT c.citing_id as id FROM citations c
            WHERE c.citing_id NOT IN (SELECT openalex_id FROM papers)
        """).fetchall()
        return [r["id"] for r in rows]

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
        """Convert a sqlite3.Row to a dict, parsing JSON fields."""
        d = dict(row)
        for field in ("authors_json", "concepts_json", "topics_json"):
            if d.get(field):
                try:
                    d[field.replace("_json", "")] = json.loads(d[field])
                except json.JSONDecodeError:
                    d[field.replace("_json", "")] = []
            else:
                d[field.replace("_json", "")] = []
        return d

    def close(self) -> None:
        """Close the database connection."""
        self.conn.close()

    def __enter__(self) -> "CitationDB":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
