"""Tests for pure functions extracted from the Streamlit dashboard (app.py).

These tests import and exercise the logic functions without needing Streamlit.
"""

import networkx as nx
import numpy as np
import pandas as pd
import pytest
from cdlib import algorithms


# ---------------------------------------------------------------------------
# Helpers: recreate the pure functions from app.py so they're testable
# (In a future refactor these should be extracted to a module.)
# ---------------------------------------------------------------------------

def run_community_detection(G_undirected: nx.Graph, resolution: float,
                            min_degree: int = 0) -> tuple[dict, int]:
    """Louvain community detection with optional min-degree core."""
    if min_degree > 0:
        core_nodes = [n for n, d in G_undirected.degree() if d >= min_degree]
        if len(core_nodes) < 3:
            core_nodes = list(G_undirected.nodes())
        core_G = G_undirected.subgraph(core_nodes).copy()
    else:
        core_G = G_undirected

    comms = algorithms.louvain(core_G, resolution=resolution, randomize=False)
    node_to_comm = {}
    for i, comm in enumerate(comms.communities):
        for node in comm:
            node_to_comm[node] = i
    n_communities = len(comms.communities)

    if min_degree > 0:
        for node in G_undirected.nodes():
            if node not in node_to_comm:
                neighbors = list(G_undirected.neighbors(node))
                core_neighbors = [nb for nb in neighbors if nb in node_to_comm]
                if core_neighbors:
                    best = max(core_neighbors,
                               key=lambda nb: G_undirected.degree(nb))
                    node_to_comm[node] = node_to_comm[best]
                else:
                    node_to_comm[node] = -1

    return node_to_comm, n_communities


def build_confusion_matrix(nodes: list, node_to_comm: dict,
                           G: nx.DiGraph) -> pd.DataFrame:
    """Build category × community confusion matrix."""
    rows = []
    for n in nodes:
        data = G.nodes[n]
        if data.get("is_seed"):
            cat = data.get("seed_category", "unknown")
            comm = node_to_comm.get(n, -1)
            rows.append({"category": cat, "community": f"C{comm}"})
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    ct = pd.crosstab(df["category"], df["community"])
    col_to_cat = {col: ct[col].idxmax() for col in ct.columns}
    seen_cats = []
    ordered_cols = []
    for col in sorted(ct.columns, key=lambda c: ct[c].max(), reverse=True):
        cat = col_to_cat[col]
        if cat not in seen_cats:
            seen_cats.append(cat)
        ordered_cols.append(col)
    for cat in ct.index:
        if cat not in seen_cats:
            seen_cats.append(cat)
    ct = ct.reindex(index=seen_cats, columns=ordered_cols, fill_value=0)
    return ct


def build_timeline(nodes: list, node_to_comm: dict, G: nx.DiGraph,
                   group_by: str = "community", bin_size: int = 1) -> pd.DataFrame:
    """Build timeline data: papers by year, grouped by community or category."""
    rows = []
    for n in nodes:
        data = G.nodes[n]
        year = data.get("publication_year")
        if year is None:
            continue
        if bin_size > 1:
            year = (year // bin_size) * bin_size
        if group_by == "community":
            group = f"C{node_to_comm.get(n, -1)}"
        else:
            group = (data.get("seed_category", "non-seed")
                     if data.get("is_seed") else "non-seed")
        rows.append({"year": year, "group": group})
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    return df.groupby(["year", "group"]).size().reset_index(name="count")


import re
from collections import Counter, defaultdict

_STOPWORDS = {
    "the", "a", "an", "of", "in", "and", "for", "to", "with", "on", "by",
    "is", "are", "from", "at", "as", "that", "this", "its", "or", "be",
    "was", "were", "been", "has", "have", "had", "not", "but", "can",
    "it", "no", "up", "out", "new", "one", "two", "role", "study",
    "results", "evidence", "effect", "effects", "model", "models",
    "based", "analysis", "data",
}


def extract_community_keywords(nodes: list, node_to_comm: dict,
                               G: nx.DiGraph, top_k: int = 8) -> dict:
    """Extract top TF-IDF-like keywords per community from paper titles."""
    comm_words = defaultdict(Counter)
    for n in nodes:
        title = G.nodes[n].get("title") or ""
        comm = node_to_comm.get(n, -1)
        words = re.findall(r"[a-z]{3,}", title.lower())
        words = [w for w in words if w not in _STOPWORDS]
        for w in words:
            comm_words[comm][w] += 1

    n_comms = len(comm_words)
    result = {}
    for comm_id, word_counts in comm_words.items():
        scored = {}
        for word, count in word_counts.items():
            n_with_word = sum(1 for c in comm_words if word in comm_words[c])
            tf = count / max(sum(word_counts.values()), 1)
            idf = np.log(1 + n_comms / max(n_with_word, 1))
            scored[word] = tf * idf
        top_words = sorted(scored, key=scored.get, reverse=True)[:top_k]
        result[comm_id] = top_words
    return result


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------

def _make_test_graph() -> nx.DiGraph:
    """Create a small test citation graph with two clear communities."""
    G = nx.DiGraph()

    # Community A: point attractor papers
    for i in range(1, 6):
        G.add_node(f"A{i}", title=f"Attractor dynamics paper {i}",
                   publication_year=2000 + i, first_author=f"Author_A{i}",
                   is_seed=True, seed_category="point_attractor",
                   cited_by_count=100 * i)
    # Dense intra-community edges
    for i in range(1, 6):
        for j in range(1, 6):
            if i != j:
                G.add_edge(f"A{i}", f"A{j}")

    # Community B: sequence papers
    for i in range(1, 6):
        G.add_node(f"B{i}", title=f"Hippocampal sequence replay paper {i}",
                   publication_year=2005 + i, first_author=f"Author_B{i}",
                   is_seed=True, seed_category="sequence",
                   cited_by_count=50 * i)
    for i in range(1, 6):
        for j in range(1, 6):
            if i != j:
                G.add_edge(f"B{i}", f"B{j}")

    # A few cross-community edges (sparse)
    G.add_edge("A1", "B1")
    G.add_edge("B3", "A2")

    # Non-seed papers
    G.add_node("X1", title="General neuroscience review", publication_year=2010,
               first_author="Reviewer", is_seed=False, cited_by_count=500)
    G.add_edge("A1", "X1")
    G.add_edge("B1", "X1")

    # Low-degree peripheral node
    G.add_node("P1", title="Peripheral paper", publication_year=2015,
               first_author="Peripheral", is_seed=False, cited_by_count=5)
    G.add_edge("P1", "A1")

    return G


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestCommunityDetection:
    """Tests for community detection with min-degree core."""

    def test_basic_detection(self):
        """Louvain finds at least 1 community."""
        G = _make_test_graph().to_undirected()
        node_to_comm, n_comm = run_community_detection(G, resolution=1.0)
        assert n_comm >= 1
        assert len(node_to_comm) == G.number_of_nodes()

    def test_min_degree_assigns_periphery(self):
        """Peripheral nodes get assigned to a community via neighbors."""
        G = _make_test_graph().to_undirected()
        node_to_comm, _ = run_community_detection(G, resolution=1.0,
                                                   min_degree=3)
        # P1 has degree 1, should still get a community assignment
        assert "P1" in node_to_comm
        # P1's only neighbor is A1, so it should be in A1's community
        assert node_to_comm["P1"] == node_to_comm["A1"]

    def test_min_degree_isolated_gets_minus_one(self):
        """Isolated peripheral nodes (no core neighbors) get community -1."""
        G = nx.Graph()
        # Need enough core nodes (>=3) so the fallback doesn't kick in
        for i in range(5):
            G.add_node(f"core{i}")
        for i in range(4):
            G.add_edge(f"core{i}", f"core{i+1}")
        G.add_node("isolated")  # no edges at all
        node_to_comm, _ = run_community_detection(G, resolution=1.0,
                                                   min_degree=1)
        assert node_to_comm["isolated"] == -1

    def test_min_degree_fallback_small_core(self):
        """If min_degree filters to <3 nodes, use all nodes."""
        G = nx.path_graph(5)  # degrees: 1, 2, 2, 2, 1
        node_to_comm, _ = run_community_detection(G, resolution=1.0,
                                                   min_degree=10)
        # Should fall back to all nodes
        assert len(node_to_comm) == 5


class TestConfusionMatrix:
    """Tests for the category × community confusion matrix."""

    def test_basic_matrix(self):
        """Confusion matrix has correct shape and values."""
        G = _make_test_graph()
        nodes = list(G.nodes())
        # Assign communities manually
        node_to_comm = {n: 0 for n in nodes if n.startswith("A")}
        node_to_comm.update({n: 1 for n in nodes if n.startswith("B")})
        node_to_comm.update({"X1": 0, "P1": 0})

        ct = build_confusion_matrix(nodes, node_to_comm, G)
        assert not ct.empty
        # Should have 2 categories (point_attractor, sequence)
        assert len(ct.index) == 2
        # point_attractor should be mostly in C0
        assert ct.loc["point_attractor", "C0"] == 5

    def test_no_seeds_returns_empty(self):
        """If no seed papers in nodes, return empty DataFrame."""
        G = _make_test_graph()
        ct = build_confusion_matrix(["X1", "P1"], {}, G)
        assert ct.empty

    def test_diagonal_sorting(self):
        """Largest values should be near the diagonal."""
        G = _make_test_graph()
        nodes = list(G.nodes())
        node_to_comm = {n: 0 for n in nodes if n.startswith("A")}
        node_to_comm.update({n: 1 for n in nodes if n.startswith("B")})
        node_to_comm.update({"X1": 0, "P1": 0})

        ct = build_confusion_matrix(nodes, node_to_comm, G)
        # The first row's max should be in the first column
        first_row = ct.iloc[0]
        assert first_row.idxmax() == ct.columns[0]


class TestTimeline:
    """Tests for timeline data construction."""

    def test_basic_timeline(self):
        """Timeline returns year/group/count data."""
        G = _make_test_graph()
        nodes = list(G.nodes())
        node_to_comm = {n: 0 for n in nodes}
        df = build_timeline(nodes, node_to_comm, G)
        assert not df.empty
        assert set(df.columns) == {"year", "group", "count"}

    def test_bin_size(self):
        """Year binning groups papers into bins."""
        G = _make_test_graph()
        nodes = list(G.nodes())
        node_to_comm = {n: 0 for n in nodes}
        df = build_timeline(nodes, node_to_comm, G, bin_size=5)
        # All years should be multiples of 5
        assert all(y % 5 == 0 for y in df["year"])

    def test_group_by_category(self):
        """Group by category shows seed categories and 'non-seed'."""
        G = _make_test_graph()
        nodes = list(G.nodes())
        node_to_comm = {n: 0 for n in nodes}
        df = build_timeline(nodes, node_to_comm, G, group_by="category")
        groups = set(df["group"])
        assert "point_attractor" in groups
        assert "sequence" in groups
        assert "non-seed" in groups

    def test_no_year_data(self):
        """Papers without publication_year are excluded."""
        G = nx.DiGraph()
        G.add_node("N1", title="No year")  # no publication_year
        df = build_timeline(["N1"], {"N1": 0}, G)
        assert df.empty


class TestKeywordExtraction:
    """Tests for community keyword extraction."""

    def test_basic_keywords(self):
        """Keywords are extracted from titles."""
        G = _make_test_graph()
        nodes = [n for n in G.nodes() if n.startswith("A")]
        node_to_comm = {n: 0 for n in nodes}
        keywords = extract_community_keywords(nodes, node_to_comm, G)
        assert 0 in keywords
        assert len(keywords[0]) > 0

    def test_stopwords_excluded(self):
        """Common stopwords are not in the keywords."""
        G = nx.DiGraph()
        G.add_node("N1", title="The role of the hippocampus in memory")
        keywords = extract_community_keywords(["N1"], {"N1": 0}, G)
        words = keywords[0]
        assert "the" not in words
        assert "role" not in words

    def test_distinctive_words_ranked_higher(self):
        """Words unique to a community should rank higher (TF-IDF)."""
        G = nx.DiGraph()
        # Community 0: all about "attractor"
        for i in range(5):
            G.add_node(f"A{i}", title=f"Attractor dynamics model {i}")
        # Community 1: all about "sequence"
        for i in range(5):
            G.add_node(f"B{i}", title=f"Sequence replay memory {i}")

        node_to_comm = {f"A{i}": 0 for i in range(5)}
        node_to_comm.update({f"B{i}": 1 for i in range(5)})
        nodes = list(G.nodes())

        keywords = extract_community_keywords(nodes, node_to_comm, G)
        assert "attractor" in keywords[0]
        assert "sequence" in keywords[1]

    def test_empty_titles(self):
        """Handles nodes with no title gracefully — returns empty list."""
        G = nx.DiGraph()
        G.add_node("N1")  # no title attribute
        keywords = extract_community_keywords(["N1"], {"N1": 0}, G)
        # No words extracted, so community 0 won't appear in results
        assert keywords.get(0, []) == []
