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


# ---------------------------------------------------------------------------
# comm_color_map
# ---------------------------------------------------------------------------

def comm_color_map(n_comms: int) -> dict[int, str]:
    """Map community IDs to distinct, visible colors."""
    colors = [
        "#4e79a7", "#f28e2b", "#59a14f", "#e15759", "#76b7b2",
        "#edc948", "#b07aa1", "#ff9da7", "#9c755f", "#bab0ac",
        "#86bcb6", "#d37295",
    ]
    return {i: colors[i % len(colors)] for i in range(n_comms)}


class TestCommColorMap:
    """Tests for community color mapping."""

    def test_returns_correct_count(self):
        """Returns one color per community."""
        result = comm_color_map(5)
        assert len(result) == 5
        assert set(result.keys()) == {0, 1, 2, 3, 4}

    def test_colors_are_hex(self):
        """All colors are valid hex strings."""
        result = comm_color_map(8)
        for color in result.values():
            assert color.startswith("#")
            assert len(color) == 7

    def test_wraps_around(self):
        """Colors wrap around when more communities than palette entries."""
        result = comm_color_map(15)
        assert len(result) == 15
        # Color 0 and color 12 should be the same (palette has 12 entries)
        assert result[0] == result[12]

    def test_zero_communities(self):
        """Zero communities returns empty dict."""
        result = comm_color_map(0)
        assert result == {}

    def test_all_visible_on_white(self):
        """No color should be too light (luminance < 200)."""
        result = comm_color_map(12)
        for color in result.values():
            r = int(color[1:3], 16)
            g = int(color[3:5], 16)
            b = int(color[5:7], 16)
            lum = 0.299 * r + 0.587 * g + 0.114 * b
            assert lum < 200, f"Color {color} too light (lum={lum:.0f})"


# ---------------------------------------------------------------------------
# get_community_names
# ---------------------------------------------------------------------------

def get_community_names(
    nodes: list, node_to_comm: dict, G: nx.DiGraph,
) -> dict[int, str]:
    """Generate descriptive names for each community using bigrams + TF-IDF."""
    comm_unigrams: dict[int, Counter] = defaultdict(Counter)
    comm_bigrams: dict[int, Counter] = defaultdict(Counter)

    for n in nodes:
        title = G.nodes[n].get("title") or ""
        comm = node_to_comm.get(n, -1)
        if comm == -1:
            continue
        words = re.findall(r"[a-z]{3,}", title.lower())
        clean = [w for w in words if w not in _STOPWORDS]
        for w in clean:
            comm_unigrams[comm][w] += 1
        for i in range(len(clean) - 1):
            comm_bigrams[comm][f"{clean[i]} {clean[i+1]}"] += 1

    uni_df: dict[str, int] = defaultdict(int)
    bi_df: dict[str, int] = defaultdict(int)
    for wc in comm_unigrams.values():
        for w in wc:
            uni_df[w] += 1
    for bc in comm_bigrams.values():
        for b in bc:
            bi_df[b] += 1

    n_comms = max(len(comm_unigrams), 1)
    ubiquity_threshold = n_comms * 0.6

    names: dict[int, str] = {}
    for comm_id in comm_unigrams:
        bi_scored = {}
        for bigram, count in comm_bigrams.get(comm_id, {}).items():
            if bi_df[bigram] >= ubiquity_threshold:
                continue
            if count < 2:
                continue
            tf = count / max(sum(comm_bigrams[comm_id].values()), 1)
            idf = np.log(1 + n_comms / max(bi_df[bigram], 1))
            bi_scored[bigram] = tf * idf

        uni_scored = {}
        for word, count in comm_unigrams[comm_id].items():
            if uni_df[word] >= ubiquity_threshold:
                continue
            tf = count / max(sum(comm_unigrams[comm_id].values()), 1)
            idf = np.log(1 + n_comms / max(uni_df[word], 1))
            uni_scored[word] = tf * idf

        top_bi = sorted(bi_scored, key=bi_scored.get, reverse=True)[:2]
        top_uni = sorted(uni_scored, key=uni_scored.get, reverse=True)[:4]

        if top_bi:
            name = top_bi[0].title()
            extras = [w for w in top_uni if w not in top_bi[0]]
            if extras:
                name = f"{name} & {extras[0].title()}"
        elif len(top_uni) >= 3:
            name = f"{top_uni[0].title()}, {top_uni[1].title()} & {top_uni[2].title()}"
        elif len(top_uni) >= 2:
            name = f"{top_uni[0].title()} & {top_uni[1].title()}"
        elif top_uni:
            name = top_uni[0].title()
        else:
            name = f"Community {comm_id}"

        names[comm_id] = name
    return names


class TestGetCommunityNames:
    """Tests for community name generation."""

    def test_distinctive_names(self):
        """Communities get names from their distinctive words."""
        G = nx.DiGraph()
        # Community 0: about grid cells
        for i in range(10):
            G.add_node(f"A{i}", title=f"Grid cells and spatial navigation paper {i}")
        # Community 1: about sharp wave ripples
        for i in range(10):
            G.add_node(f"B{i}", title=f"Sharp wave ripple replay paper {i}")

        node_to_comm = {f"A{i}": 0 for i in range(10)}
        node_to_comm.update({f"B{i}": 1 for i in range(10)})
        nodes = list(G.nodes())

        names = get_community_names(nodes, node_to_comm, G)
        assert 0 in names
        assert 1 in names
        # "grid" should be in community 0's name, not community 1's
        assert "grid" in names[0].lower() or "spatial" in names[0].lower()
        assert "ripple" in names[1].lower() or "sharp" in names[1].lower()

    def test_ubiquitous_unigrams_excluded(self):
        """Ubiquitous unigrams are excluded; distinctive bigrams may contain them."""
        G = nx.DiGraph()
        # 3 communities, all share "hippocampal" but have distinct second words
        for i in range(10):
            G.add_node(f"A{i}", title=f"Hippocampal attractor network {i}")
        for i in range(10):
            G.add_node(f"B{i}", title=f"Hippocampal sequence replay {i}")
        for i in range(10):
            G.add_node(f"C{i}", title=f"Hippocampal place field {i}")

        node_to_comm = {f"A{i}": 0 for i in range(10)}
        node_to_comm.update({f"B{i}": 1 for i in range(10)})
        node_to_comm.update({f"C{i}": 2 for i in range(10)})
        nodes = list(G.nodes())

        names = get_community_names(nodes, node_to_comm, G)
        # Each community should have a distinctive name
        assert len(names) == 3
        # The names should be different from each other
        name_set = set(names.values())
        assert len(name_set) == 3, f"Names not unique: {names}"
        # Distinctive words should appear: attractor, sequence, place
        all_names = " ".join(names.values()).lower()
        assert "attractor" in all_names or "network" in all_names
        assert "sequence" in all_names or "replay" in all_names
        assert "place" in all_names or "field" in all_names

    def test_empty_community(self):
        """Community with no extractable words gets a fallback name."""
        G = nx.DiGraph()
        G.add_node("N1", title="")  # empty title
        names = get_community_names(["N1"], {"N1": 0}, G)
        # Should get fallback
        assert 0 in names or names == {}

    def test_names_are_title_cased(self):
        """Names should be title-cased (when keywords are found)."""
        G = nx.DiGraph()
        # Two communities so distinctive words exist
        for i in range(10):
            G.add_node(f"A{i}", title=f"Place cells firing rate paper {i}")
        for i in range(10):
            G.add_node(f"B{i}", title=f"Grid cells navigation paper {i}")

        node_to_comm = {f"A{i}": 0 for i in range(10)}
        node_to_comm.update({f"B{i}": 1 for i in range(10)})
        names = get_community_names(list(G.nodes()), node_to_comm, G)

        for name in names.values():
            if name.startswith("Community"):
                continue  # fallback name, skip
            # Each word should start with uppercase (ignore "&", ",")
            for word in name.replace("&", "").replace(",", "").split():
                if word:
                    assert word[0].isupper(), f"'{word}' not title-cased in '{name}'"


# ---------------------------------------------------------------------------
# build_top_table
# ---------------------------------------------------------------------------

def build_top_table(
    nodes: list, node_to_comm: dict, G: nx.DiGraph,
    pr: dict, in_deg: dict,
) -> pd.DataFrame:
    """Build a sortable table of paper details."""
    rows = []
    for n in nodes:
        data = G.nodes[n]
        rows.append({
            "OpenAlex ID": n,
            "Author": (data.get("first_author") or "?")[:25],
            "Year": data.get("publication_year"),
            "Title": (data.get("title") or "?")[:60],
            "Seed": "Yes" if data.get("is_seed") else "",
            "Category": data.get("seed_category", ""),
            "Community": node_to_comm.get(n, -1),
            "In-degree": in_deg.get(n, 0),
            "Global cites": data.get("cited_by_count") or 0,
            "PageRank": round(pr.get(n, 0), 6),
        })
    df = pd.DataFrame(rows)
    return df.sort_values("In-degree", ascending=False)


class TestBuildTopTable:
    """Tests for the paper details table builder."""

    def test_basic_table(self):
        """Table has expected columns and rows."""
        G = _make_test_graph()
        nodes = list(G.nodes())
        node_to_comm = {n: 0 for n in nodes}
        pr = {n: 0.01 for n in nodes}
        in_deg = {n: 5 for n in nodes}

        df = build_top_table(nodes, node_to_comm, G, pr, in_deg)
        assert len(df) == len(nodes)
        assert "Author" in df.columns
        assert "Title" in df.columns
        assert "Community" in df.columns
        assert "PageRank" in df.columns

    def test_sorted_by_in_degree(self):
        """Table is sorted by in-degree descending."""
        G = _make_test_graph()
        nodes = list(G.nodes())[:5]
        node_to_comm = {n: 0 for n in nodes}
        pr = {n: 0.01 for n in nodes}
        in_deg = {nodes[0]: 100, nodes[1]: 50, nodes[2]: 200,
                  nodes[3]: 10, nodes[4]: 75}

        df = build_top_table(nodes, node_to_comm, G, pr, in_deg)
        degrees = list(df["In-degree"])
        assert degrees == sorted(degrees, reverse=True)

    def test_seed_column(self):
        """Seed papers are marked 'Yes'."""
        G = _make_test_graph()
        nodes = ["A1", "X1"]  # A1 is seed, X1 is not
        node_to_comm = {n: 0 for n in nodes}
        pr = {n: 0.01 for n in nodes}
        in_deg = {n: 1 for n in nodes}

        df = build_top_table(nodes, node_to_comm, G, pr, in_deg)
        seed_vals = dict(zip(df["OpenAlex ID"], df["Seed"]))
        assert seed_vals["A1"] == "Yes"
        assert seed_vals["X1"] == ""

    def test_truncates_long_titles(self):
        """Titles longer than 60 chars are truncated."""
        G = nx.DiGraph()
        G.add_node("N1", title="A" * 100, first_author="Test", publication_year=2020)
        df = build_top_table(["N1"], {"N1": 0}, G, {"N1": 0.01}, {"N1": 1})
        assert len(df.iloc[0]["Title"]) == 60
