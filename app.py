"""Interactive citation network dashboard.

Run with: uv run streamlit run app.py
"""

import re
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from cdlib import algorithms
from fa2 import ForceAtlas2

from src.data.endnote_parser import parse_endnote_xml
from src.data.openalex_client import OpenAlexClient
from src.data.collector import SnowballCollector
from src.data.storage import CitationDB, DEFAULT_DB_PATH
from src.network.builder import build_citation_graph, graph_summary
from src.viz.plots import CATEGORY_COLORS, ensure_category_colors, get_node_color

# Common stopwords for title-based keyword extraction
_STOPWORDS = frozenset({
    "the", "a", "an", "of", "in", "and", "for", "to", "with", "on", "by",
    "is", "are", "from", "at", "as", "that", "this", "its", "or", "be",
    "was", "were", "been", "has", "have", "had", "not", "but", "can",
    "do", "does", "did", "will", "would", "could", "should", "may",
    "their", "which", "we", "our", "via", "using", "between", "during",
    "through", "after", "before", "into", "about", "than", "both",
    "each", "all", "more", "also", "how", "what", "when", "where",
    "it", "no", "up", "out", "new", "one", "two", "role", "study",
    "results", "evidence", "effect", "effects", "model", "models",
    "based", "analysis", "data", "i", "ii", "iii",
})

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Citation Network Explorer",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Data loading (cached with TTL so metadata updates are picked up)
# ---------------------------------------------------------------------------
@st.cache_resource(ttl=300)
def load_graph(db_path: str | Path = DEFAULT_DB_PATH) -> tuple[nx.DiGraph, nx.Graph]:
    """Load citation graph and its undirected version (cached)."""
    db = CitationDB(db_path)
    G = build_citation_graph(db)
    db.close()
    G_undirected = G.to_undirected()
    return G, G_undirected


@st.cache_data
def compute_metrics(_G):
    """Compute node metrics (PageRank, betweenness, etc.)."""
    pr = nx.pagerank(_G)
    in_deg = dict(_G.in_degree())
    out_deg = dict(_G.out_degree())
    return pr, in_deg, out_deg


@st.cache_data
def run_community_detection(
    _G_undirected: nx.Graph, resolution: float, min_degree: int = 0,
    _graph_key: tuple[int, int] = (0, 0),
) -> tuple[dict[str, int], int]:
    """Run Louvain with a given resolution parameter.

    Args:
        _G_undirected: Undirected graph to detect communities on.
        resolution: Louvain resolution parameter.
        min_degree: Minimum degree for core community detection.
            Nodes below threshold are assigned to their highest-degree
            neighbor's community.
        _graph_key: (n_nodes, n_edges) tuple for cache invalidation.
            The graph itself is unhashable, so this fingerprint ensures
            the cache busts when the graph changes.
    """
    if min_degree > 0:
        # Detect on k-core, assign periphery to nearest community
        core_nodes = [n for n, d in _G_undirected.degree() if d >= min_degree]
        if len(core_nodes) < 3:
            core_nodes = list(_G_undirected.nodes())
        core_G = _G_undirected.subgraph(core_nodes).copy()
    else:
        core_G = _G_undirected

    comms = algorithms.louvain(core_G, resolution=resolution, randomize=False)
    node_to_comm = {}
    for i, comm in enumerate(comms.communities):
        for node in comm:
            node_to_comm[node] = i
    n_communities = len(comms.communities)

    # Assign peripheral nodes to their most-connected neighbor's community
    if min_degree > 0:
        for node in _G_undirected.nodes():
            if node not in node_to_comm:
                neighbors = list(_G_undirected.neighbors(node))
                core_neighbors = [nb for nb in neighbors if nb in node_to_comm]
                if core_neighbors:
                    # Assign to community of highest-degree core neighbor
                    best = max(core_neighbors, key=lambda nb: _G_undirected.degree(nb))
                    node_to_comm[node] = node_to_comm[best]
                else:
                    node_to_comm[node] = -1  # Unassigned

    return node_to_comm, n_communities


# ---------------------------------------------------------------------------
# Upload flow — shown when no data is available yet
# ---------------------------------------------------------------------------
# Determine which DB to use
if "db_path" not in st.session_state:
    # Check if the default DB already has data
    if DEFAULT_DB_PATH.exists():
        _test_db = CitationDB(DEFAULT_DB_PATH)
        _has_data = _test_db.get_paper_count() > 0
        _test_db.close()
        if _has_data:
            st.session_state["db_path"] = str(DEFAULT_DB_PATH)
            st.session_state["pipeline_complete"] = True


def _show_upload_page() -> None:
    """Render the upload page and run the pipeline on submit."""
    st.title("📚 Citation Network Explorer")
    st.markdown(
        "Upload an **EndNote XML** export and this app will automatically "
        "resolve your papers via [OpenAlex](https://openalex.org), run a "
        "snowball citation collection, and build an interactive network."
    )

    uploaded = st.file_uploader(
        "Upload an EndNote XML export (.xml)",
        type=["xml"],
        help="In EndNote: File → Export → select XML format.",
    )

    email = st.text_input(
        "Email (optional — for faster OpenAlex rate limits)",
        placeholder="you@university.edu",
        help="OpenAlex gives higher rate limits to requests with a contact email.",
    )

    if uploaded is None:
        st.info("Upload a file to get started.")
        st.stop()

    # Parse the XML
    try:
        records = parse_endnote_xml(uploaded.read())
    except Exception as exc:
        st.error(f"Failed to parse XML: {exc}")
        st.stop()

    if not records:
        st.error("No records found in the XML file.")
        st.stop()

    # Show summary
    n_with_doi = sum(1 for r in records if r.get("doi"))
    categories = sorted(set(r["seed_category"] for r in records))
    cat_counts = Counter(r["seed_category"] for r in records)

    st.success(f"**{len(records)}** papers parsed ({n_with_doi} with DOIs)")

    if len(categories) > 1 or (len(categories) == 1 and categories[0] != "uncategorized"):
        st.markdown("**Groups found:**")
        cat_df = pd.DataFrame(
            [{"Group": c, "Papers": cat_counts[c]} for c in categories]
        )
        st.dataframe(cat_df, hide_index=True, use_container_width=False)

    # Run pipeline
    st.markdown("---")
    st.subheader("Building citation network…")

    # Create a session-specific DB in a temp directory
    tmp_dir = tempfile.mkdtemp(prefix="citation_net_")
    db_path = Path(tmp_dir) / "citations.db"
    db = CitationDB(db_path)
    client = OpenAlexClient(email=email or "citation-network-app@example.com")
    collector = SnowballCollector(client, db)

    # Phase 1: Resolve seeds
    progress_bar = st.progress(0, text="Resolving papers via OpenAlex…")
    status_area = st.empty()

    def update_progress(current: int, total: int, msg: str) -> None:
        progress_bar.progress(
            current / max(total, 1),
            text=f"Resolving papers ({current}/{total})",
        )
        status_area.caption(msg)

    seed_stats = collector.import_seed_records(
        records, progress_callback=update_progress,
    )
    progress_bar.progress(1.0, text="Seed resolution complete!")
    status_area.caption(
        f"Resolved {seed_stats['resolved']}/{seed_stats['total']} papers "
        f"({seed_stats['failed']} failed)"
    )

    # Phase 2: Snowball (level 1)
    if seed_stats["resolved"] > 0:
        snow_bar = st.progress(0, text="Running snowball collection (level 1)…")
        level_stats = collector.collect_level(level=1, max_cited_by=200)
        snow_bar.progress(1.0, text="Snowball complete!")
        st.caption(
            f"Added {level_stats['papers_added']} papers, "
            f"{level_stats['citations_added']} citation edges"
        )

    db.close()

    # Register dynamic category colors
    ensure_category_colors(categories)

    # Store in session state
    st.session_state["db_path"] = str(db_path)
    st.session_state["pipeline_complete"] = True
    st.rerun()


# Gate: if no data yet, show upload page
if not st.session_state.get("pipeline_complete"):
    _show_upload_page()
    st.stop()

# Load graph from the session's DB
_db_path = st.session_state.get("db_path", str(DEFAULT_DB_PATH))
G, G_undirected = load_graph(_db_path)

st.title("📚 Citation Network Explorer")

if G.number_of_nodes() == 0:
    st.warning("No papers in the database. Please upload an EndNote XML file.")
    if st.button("↩ Upload new file"):
        st.session_state.pop("db_path", None)
        st.session_state.pop("pipeline_complete", None)
        st.cache_resource.clear()
        st.rerun()
    st.stop()

pr, in_deg, out_deg = compute_metrics(G)
stats = graph_summary(G)  # cheap: just len() calls on cached graph

# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------
st.sidebar.header("Controls")

# Upload new file button in sidebar
if st.sidebar.button("📤 Upload new file"):
    st.session_state.pop("db_path", None)
    st.session_state.pop("pipeline_complete", None)
    st.cache_resource.clear()
    st.cache_data.clear()
    st.rerun()

# Top N
top_n = st.sidebar.slider("Top N papers (by in-degree)", 50, 1000, 800, step=50)

# Minimum year filter
all_graph_years = sorted(set(
    G.nodes[n].get("publication_year") or 0
    for n in G.nodes
))
valid_years = [y for y in all_graph_years if y > 0]
if valid_years:
    min_year_filter = st.sidebar.slider(
        "Minimum publication year",
        min(valid_years), max(valid_years), min(valid_years),
        help="Only show papers published in or after this year.",
    )
else:
    min_year_filter = 0

# Get top N nodes, filtered by minimum year
year_filtered = [
    n for n in G.nodes
    if (G.nodes[n].get("publication_year") or 0) >= min_year_filter
]
top_nodes = sorted(year_filtered, key=lambda n: in_deg.get(n, 0), reverse=True)[:top_n]
sub_G = G.subgraph(top_nodes).copy()
sub_U = sub_G.to_undirected()

# Color by — default to community if no meaningful categories exist
_all_cats = set(
    G.nodes[n].get("seed_category", "") for n in G.nodes if G.nodes[n].get("is_seed")
)
_has_real_categories = bool(_all_cats - {"", "uncategorized"})
_color_options = ["Detected community", "Review category", "Publication year"]
_color_default = 0 if not _has_real_categories else 0
color_by = st.sidebar.selectbox("Color by", _color_options, index=_color_default)

# Node size by
size_by = st.sidebar.selectbox(
    "Node size by",
    ["In-degree (network)", "Global citation count", "PageRank"],
    index=1,
)

# Layout algorithm
layout_algo = st.sidebar.selectbox(
    "Layout algorithm",
    ["ForceAtlas2", "Spring", "ForceAtlas2 (LinLog)"],
    help="Spring: default force-directed. ForceAtlas2: better cluster separation. "
         "LinLog: mathematically equivalent to modularity (Noack 2009).",
)

# FA2 spread parameter (only shown when FA2 is selected)
if layout_algo.startswith("ForceAtlas2"):
    fa2_scaling = st.sidebar.slider(
        "FA2 spread (scaling ratio)", 1.0, 50.0, 10.0, step=1.0,
        help="Higher = more spread between nodes. Try 10–30 for good cluster separation.",
    )
    fa2_gravity = st.sidebar.slider(
        "FA2 gravity", 0.1, 10.0, 1.0, step=0.1,
        help="Pulls nodes toward center. Higher = tighter overall, lower = more spread.",
    )
else:
    fa2_scaling = 10.0
    fa2_gravity = 1.0

# Community detection scope
comm_scope = st.sidebar.selectbox(
    "Detect communities on",
    ["Full network", "Displayed nodes"],
    help="'Full network' runs Louvain on all papers — communities stay "
         "stable as you change Top N. 'Displayed nodes' re-detects on the current subgraph.",
)

# Louvain resolution
resolution = st.sidebar.slider(
    "Louvain resolution", 0.3, 3.0, 1.0, step=0.1,
    help="1.0 = default (modularity optimum). Higher = more, finer communities. "
         "Below 1 may over-merge and produce artifacts.",
)

# Min degree for community detection
min_comm_degree = st.sidebar.slider(
    "Min degree for community core", 0, 10, 6, step=1,
    help="Detect communities on nodes with at least this many connections. "
         "Peripheral nodes are assigned to their best neighbor's community. "
         "Reduces noise from degree-1 papers.",
)

# Year range
all_years = [
    G.nodes[n].get("publication_year") or 2000
    for n in top_nodes
]
min_year = min(all_years)
max_year = max(all_years)
year_range = st.sidebar.slider(
    "Publication year range",
    min_year, max_year, (min_year, max_year),
)

# Run community detection
if comm_scope.startswith("Full network"):
    # Detect on all papers — stable communities regardless of Top N slider
    node_to_comm, n_communities = run_community_detection(
        G_undirected, resolution, min_degree=min_comm_degree,
        _graph_key=(G_undirected.number_of_nodes(), G_undirected.number_of_edges()),
    )
else:
    # Detect on the currently displayed subgraph
    node_to_comm, n_communities = run_community_detection(
        sub_U, resolution, min_degree=min_comm_degree,
        _graph_key=(sub_U.number_of_nodes(), sub_U.number_of_edges()),
    )

# Category filter
all_categories = sorted(set(
    G.nodes[n].get("seed_category", "") for n in top_nodes if G.nodes[n].get("is_seed")
))
all_categories = [c for c in all_categories if c]

# Edge opacity
st.sidebar.header("Display")
edge_opacity = st.sidebar.slider(
    "Edge opacity", 0.0, 1.0, 0.15, step=0.05,
    help="Reduce to declutter edges. Set to 0 to hide edges entirely.",
)

st.sidebar.header("Filters")
show_non_seeds = st.sidebar.checkbox("Show non-seed papers", value=True)
selected_categories = st.sidebar.multiselect(
    "Filter seed categories",
    all_categories,
    default=all_categories,
)

# Search
search_query = st.sidebar.text_input("Search (author or title)")

# Timeline bin size
st.sidebar.header("Timeline")
bin_years = st.sidebar.select_slider(
    "Year bin size",
    options=[1, 2, 5, 10],
    value=5,
    help="Group papers into bins of N years",
)

# Clear selection
if st.session_state.get("selected_node") or st.session_state.get("selected_community") is not None:
    if st.sidebar.button("✕ Clear selection"):
        st.session_state.pop("selected_node", None)
        st.session_state.pop("selected_community", None)
        st.session_state.pop("_highlight_rerun", None)
        st.rerun()

# Cache management
if st.sidebar.button("🔄 Clear cache & reload"):
    st.cache_resource.clear()
    st.cache_data.clear()
    st.rerun()

# ---------------------------------------------------------------------------
# Filter nodes
# ---------------------------------------------------------------------------
filtered_nodes = []
for n in top_nodes:
    data = G.nodes[n]
    year = data.get("publication_year") or 2000
    if year < year_range[0] or year > year_range[1]:
        continue
    if data.get("is_seed"):
        cat = data.get("seed_category", "")
        if cat and cat not in selected_categories:
            continue
    elif not show_non_seeds:
        continue
    if search_query:
        title = (data.get("title") or "").lower()
        author = (data.get("first_author") or "").lower()
        if search_query.lower() not in title and search_query.lower() not in author:
            continue
    filtered_nodes.append(n)

filtered_G = sub_G.subgraph(filtered_nodes).copy()

# ---------------------------------------------------------------------------
# Compute layout (cached per subgraph)
# ---------------------------------------------------------------------------
@st.cache_data
def compute_layout(nodes, edges, algorithm, scaling=10.0, gravity=1.0, seed=42):
    """Compute layout positions for the filtered graph.

    Args:
        nodes: Tuple of node IDs.
        edges: Tuple of (u, v) edge pairs.
        algorithm: One of "Spring", "ForceAtlas2", "ForceAtlas2 (LinLog)".
        scaling: FA2 scaling ratio (higher = more spread).
        gravity: FA2 gravity (higher = tighter).
        seed: Random seed for reproducibility.
    """
    tmp_G = nx.DiGraph()
    tmp_G.add_nodes_from(nodes)
    tmp_G.add_edges_from(edges)

    if algorithm == "Spring":
        k = 2.0 / (len(nodes) ** 0.3) if nodes else 1.0
        return nx.spring_layout(tmp_G, k=k, iterations=80, seed=seed)

    # ForceAtlas2 variants
    linlog = algorithm == "ForceAtlas2 (LinLog)"
    fa2_instance = ForceAtlas2(
        outboundAttractionDistribution=True,
        linLogMode=linlog,
        adjustSizes=False,
        edgeWeightInfluence=1.0,
        jitterTolerance=1.0,
        barnesHutOptimize=True,
        barnesHutTheta=1.2,
        scalingRatio=scaling,
        strongGravityMode=False,
        gravity=gravity,
        verbose=False,
    )
    # FA2 needs an undirected graph
    tmp_U = tmp_G.to_undirected()
    positions = fa2_instance.forceatlas2_networkx_layout(
        tmp_U, pos=None, iterations=200
    )
    return positions


pos = compute_layout(
    tuple(sorted(filtered_nodes)),
    tuple(sorted(filtered_G.edges())),
    algorithm=layout_algo,
    scaling=fa2_scaling,
    gravity=fa2_gravity,
    seed=42,
)

# Clip outlier positions so the main cluster fills the viewport
if pos and len(pos) > 10:
    xs = np.array([p[0] for p in pos.values()])
    ys = np.array([p[1] for p in pos.values()])
    # Use 5th/95th percentile to define the "main cluster" bounds
    x_lo, x_hi = np.percentile(xs, 2), np.percentile(xs, 98)
    y_lo, y_hi = np.percentile(ys, 2), np.percentile(ys, 98)
    x_pad = (x_hi - x_lo) * 0.3
    y_pad = (y_hi - y_lo) * 0.3
    pos = {
        n: (
            np.clip(p[0], x_lo - x_pad, x_hi + x_pad),
            np.clip(p[1], y_lo - y_pad, y_hi + y_pad),
        )
        for n, p in pos.items()
    }

# ---------------------------------------------------------------------------
# Filter to nodes that have layout positions (do this BEFORE building arrays)
# ---------------------------------------------------------------------------
filtered_nodes = [n for n in filtered_nodes if n in pos]
filtered_set = set(filtered_nodes)
node_x = [pos[n][0] for n in filtered_nodes]
node_y = [pos[n][1] for n in filtered_nodes]

# ---------------------------------------------------------------------------
# Color + size computation
# ---------------------------------------------------------------------------
def comm_color_map(n_comms: int) -> dict[int, str]:
    """Map community IDs to distinct, visible colors."""
    # Hand-picked palette: all colors readable on white, distinct from each other
    colors = [
        "#4e79a7",  # steel blue
        "#f28e2b",  # orange
        "#59a14f",  # green
        "#e15759",  # red
        "#76b7b2",  # teal
        "#edc948",  # gold (not pale yellow)
        "#b07aa1",  # purple
        "#ff9da7",  # pink
        "#9c755f",  # brown
        "#bab0ac",  # gray
        "#86bcb6",  # light teal
        "#d37295",  # rose
    ]
    return {i: colors[i % len(colors)] for i in range(n_comms)}


comm_colors = comm_color_map(n_communities)

node_colors: list = []
node_sizes: list[float] = []
node_texts: list[str] = []
node_hovers: list[str] = []

for n in filtered_nodes:
    data = G.nodes[n]
    is_seed = data.get("is_seed", False)
    cat = data.get("seed_category", "")
    year = data.get("publication_year") or "?"
    author = data.get("first_author") or "Unknown"
    title = data.get("title") or "(no metadata)"
    comm_id = node_to_comm.get(n, -1)

    # Color — use get_node_color for category mode, community map otherwise
    if color_by == "Review category":
        node_colors.append(get_node_color(data))
    elif color_by == "Detected community":
        node_colors.append(comm_colors.get(comm_id, "#cccccc"))
    else:  # Publication year
        node_colors.append(year if isinstance(year, int) else 2000)

    # Size
    if size_by == "In-degree (network)":
        node_sizes.append(max(5, in_deg.get(n, 0) * 0.15))
    elif size_by == "Global citation count":
        node_sizes.append(max(5, (data.get("cited_by_count") or 0) ** 0.4))
    else:  # PageRank
        node_sizes.append(max(5, pr.get(n, 0) * 50000))

    # Text label (seeds only)
    if is_seed:
        parts = (author or "").replace(",", " ").split()
        last = parts[-1] if parts else "?"
        node_texts.append(f"{last} {year}")
    else:
        node_texts.append("")

    # Hover tooltip
    seed_tag = " [SEED]" if is_seed else ""
    cat_tag = f"<br>Category: {cat}" if cat else ""
    node_hovers.append(
        f"<b>{author} ({year})</b>{seed_tag}"
        f"<br>Community: {comm_id}{cat_tag}"
        f"<br>{title[:100]}"
        f"<br>In-degree: {in_deg.get(n, 0)}"
        f"<br>PageRank: {pr.get(n, 0):.5f}"
    )

# ---------------------------------------------------------------------------
# Network graph (plotly) — with click-to-highlight

# Selection state: node OR community
sel_focus = st.session_state.get("selected_node", None)
sel_community = st.session_state.get("selected_community", None)
if sel_focus and sel_focus not in filtered_set:
    sel_focus = None

# Compute active set: either a node + neighbors, or an entire community
neighbors = set()
active_set = set()
if sel_focus and sel_focus in filtered_G:
    neighbors = (
        set(filtered_G.predecessors(sel_focus))
        | set(filtered_G.successors(sel_focus))
    )
    active_set = neighbors | {sel_focus}
elif sel_community is not None:
    active_set = {n for n in filtered_nodes if node_to_comm.get(n) == sel_community}

has_selection = bool(sel_focus) or sel_community is not None

# --- Edges ---
edge_x, edge_y = [], []
hi_edge_x, hi_edge_y = [], []
for u, v in filtered_G.edges():
    if u not in pos or v not in pos:
        continue
    x0, y0 = pos[u]
    x1, y1 = pos[v]
    # Highlighted: edges within active set
    if has_selection and u in active_set and v in active_set:
        hi_edge_x.extend([x0, x1, None])
        hi_edge_y.extend([y0, y1, None])
    else:
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

dim_edge_alpha = 0.03 if has_selection else edge_opacity
edge_trace = go.Scatter(
    x=edge_x, y=edge_y,
    line=dict(width=0.3, color=f"rgba(180,180,180,{dim_edge_alpha})"),
    hoverinfo="none",
    mode="lines",
)
traces = [edge_trace]

if hi_edge_x:
    hi_edge_trace = go.Scatter(
        x=hi_edge_x, y=hi_edge_y,
        line=dict(width=1.5, color="rgba(50,50,50,0.6)"),
        hoverinfo="none",
        mode="lines",
    )
    traces.append(hi_edge_trace)

# --- Nodes ---
# Per-node opacity: full for active set, dimmed for everything else
node_opacities = []
node_border_widths = []
node_border_colors = []
display_texts = list(node_texts)  # copy

for i, n in enumerate(filtered_nodes):
    if not has_selection:
        # No selection — everything normal
        node_opacities.append(1.0)
        node_border_widths.append(0.5)
        node_border_colors.append("black")
    elif n == sel_focus:
        # Selected node: full opacity + circle highlight
        node_opacities.append(1.0)
        node_border_widths.append(4)
        node_border_colors.append("red")
    elif n in active_set:
        # In active set (neighbor or same community): full opacity
        node_opacities.append(1.0)
        node_border_widths.append(1)
        node_border_colors.append("black")
    else:
        # Non-active: dimmed
        node_opacities.append(0.08)
        node_border_widths.append(0)
        node_border_colors.append("rgba(0,0,0,0)")
        display_texts[i] = ""  # hide labels

marker_common = dict(
    size=node_sizes,
    line=dict(width=node_border_widths, color=node_border_colors),
    opacity=node_opacities,
)

if color_by == "Publication year":
    marker_common.update(
        color=node_colors, colorscale="Viridis",
        colorbar=dict(title="Year"),
    )
else:
    marker_common["color"] = node_colors

node_trace = go.Scatter(
    x=node_x, y=node_y,
    mode="markers+text",
    hoverinfo="text",
    hovertext=node_hovers,
    text=display_texts,
    textposition="top center",
    textfont=dict(size=7),
    marker=marker_common,
)
traces.append(node_trace)
node_trace_idx = len(traces) - 1

# --- Community names using TF-IDF distinctive keywords ---
def get_community_names(
    nodes: list[str], node_to_comm: dict[str, int], G: nx.DiGraph,
) -> dict[int, str]:
    """Generate descriptive names for each community.

    Strategy: extract bigrams and unigrams from titles, score by TF-IDF
    (excluding terms common across all communities), then assemble a
    short readable phrase from the top-scoring terms.
    """
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
        # Bigrams from cleaned words
        for i in range(len(clean) - 1):
            comm_bigrams[comm][f"{clean[i]} {clean[i+1]}"] += 1

    # Document frequency: how many communities contain each term
    uni_df: dict[str, int] = defaultdict(int)
    bi_df: dict[str, int] = defaultdict(int)
    for wc in comm_unigrams.values():
        for w in wc:
            uni_df[w] += 1
    for bc in comm_bigrams.values():
        for b in bc:
            bi_df[b] += 1

    n_comms = max(len(comm_unigrams), 1)
    ubiquity_threshold = n_comms * 0.6  # skip terms in >60% of communities

    names: dict[int, str] = {}
    for comm_id in comm_unigrams:
        # Score bigrams (prefer these — more readable)
        bi_scored = {}
        for bigram, count in comm_bigrams.get(comm_id, {}).items():
            if bi_df[bigram] >= ubiquity_threshold:
                continue
            if count < 2:  # bigram must appear at least twice
                continue
            tf = count / max(sum(comm_bigrams[comm_id].values()), 1)
            idf = np.log(1 + n_comms / max(bi_df[bigram], 1))
            bi_scored[bigram] = tf * idf

        # Score unigrams
        uni_scored = {}
        for word, count in comm_unigrams[comm_id].items():
            if uni_df[word] >= ubiquity_threshold:
                continue
            tf = count / max(sum(comm_unigrams[comm_id].values()), 1)
            idf = np.log(1 + n_comms / max(uni_df[word], 1))
            uni_scored[word] = tf * idf

        # Build name: best bigram + a qualifying unigram, or top 3 unigrams
        top_bi = sorted(bi_scored, key=bi_scored.get, reverse=True)[:2]
        top_uni = sorted(uni_scored, key=uni_scored.get, reverse=True)[:4]

        if top_bi:
            # Use best bigram as the core phrase
            name = top_bi[0].title()
            # Add a qualifying word if it's not redundant
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


comm_names = get_community_names(filtered_nodes, node_to_comm, G)

fig_network = go.Figure(
    data=traces,
    layout=go.Layout(
        showlegend=False,
        hovermode="closest",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        margin=dict(l=0, r=125, t=30, b=0),
        height=650,
        plot_bgcolor="white",
    ),
)

# Community labels as plotly annotations on the right side of the graph
label_trace_idx = None
label_comm_ids: list[int] = []
if comm_names and color_by == "Detected community":
    n_labels = len(comm_names)
    label_spacing = min(0.035, 0.7 / max(n_labels, 1))

    # Shrink the plot area to make room for labels on the right
    # Use a second y-axis for the label trace so labels sit in paper-like coords
    x_all = [pos[n][0] for n in filtered_nodes if n in pos]
    y_all = [pos[n][1] for n in filtered_nodes if n in pos]
    x_min, x_max = (min(x_all), max(x_all)) if x_all else (0, 1)
    y_min, y_max = (min(y_all), max(y_all)) if y_all else (0, 1)
    x_range = x_max - x_min or 1
    y_range = y_max - y_min or 1

    # Place labels in data coordinates, to the right of the graph
    label_x_pos = x_max + x_range * 0.02
    label_y_top = y_max - y_range * 0.02
    label_y_step = y_range * label_spacing * 1.2

    # Title annotation (not clickable, just text)
    fig_network.add_annotation(
        x=label_x_pos, y=label_y_top + label_y_step * 0.8,
        text="<b>Clusters & Keywords</b>",
        showarrow=False,
        font=dict(size=11, color="black"),
        xanchor="left", yanchor="middle",
    )

    # Labels as a clickable scatter trace with text
    label_xs = []
    label_ys = []
    label_texts_list = []
    label_colors_list = []
    for i, (cid, name) in enumerate(sorted(comm_names.items())):
        color = comm_colors.get(cid, "#888888")
        label_xs.append(label_x_pos)
        label_ys.append(label_y_top - i * label_y_step)
        label_texts_list.append(f"<b>C{cid}</b> — {name}")
        label_colors_list.append(color)
        label_comm_ids.append(cid)

    label_trace = go.Scatter(
        x=label_xs, y=label_ys,
        mode="text",
        text=label_texts_list,
        textposition="middle right",
        textfont=dict(size=11, color=label_colors_list),
        hovertext=[f"Click to highlight C{cid}" for cid, _ in sorted(comm_names.items())],
        hoverinfo="text",
        cliponaxis=False,
    )
    fig_network.add_trace(label_trace)
    label_trace_idx = len(fig_network.data) - 1

# ---------------------------------------------------------------------------
# Confusion matrix: categories × communities
# ---------------------------------------------------------------------------
def build_confusion_matrix(
    nodes: list[str], node_to_comm: dict[str, int], G: nx.DiGraph,
) -> pd.DataFrame:
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

    # Sort so the largest values fall along the diagonal:
    # For each community (column), find the category with the max count,
    # then order columns by that category's row order.
    # First, assign each column to its dominant category
    col_to_cat = {col: ct[col].idxmax() for col in ct.columns}
    # Order categories by their first dominant community appearance
    seen_cats = []
    ordered_cols = []
    for col in sorted(ct.columns, key=lambda c: ct[c].max(), reverse=True):
        cat = col_to_cat[col]
        if cat not in seen_cats:
            seen_cats.append(cat)
        ordered_cols.append(col)
    # Remaining categories not dominant in any community
    for cat in ct.index:
        if cat not in seen_cats:
            seen_cats.append(cat)
    # Reorder
    ct = ct.reindex(index=seen_cats, columns=ordered_cols, fill_value=0)
    return ct


confusion = build_confusion_matrix(filtered_nodes, node_to_comm, G)

if not confusion.empty:
    fig_confusion = px.imshow(
        confusion,
        labels=dict(x="Detected Community", y="Review Category", color="Count"),
        color_continuous_scale="Blues",
        aspect="auto",
        text_auto=True,
    )
    # Add community names to x-axis labels and color-code via annotations
    # Replace tick labels with "C0: Name" format
    short_names = {}
    for col_label in confusion.columns:
        cid = int(col_label.replace("C", "")) if col_label.startswith("C") else -1
        cname = comm_names.get(cid, "")
        # Truncate name for tick label
        short = cname.split(" & ")[0][:15] if cname else ""
        short_names[col_label] = f"{col_label}\n{short}" if short else col_label

    fig_confusion.update_xaxes(
        tickvals=list(range(len(confusion.columns))),
        ticktext=[short_names[c] for c in confusion.columns],
    )
    fig_confusion.update_layout(
        height=400, margin=dict(l=0, r=0, t=30, b=40),
    )

# ---------------------------------------------------------------------------
# Timeline: papers by year, stacked by community or category
# ---------------------------------------------------------------------------
def build_timeline(
    nodes: list[str], node_to_comm: dict[str, int], G: nx.DiGraph,
    group_by: str = "community", bin_size: int = 1,
) -> pd.DataFrame:
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
            group = data.get("seed_category", "non-seed") if data.get("is_seed") else "non-seed"
        rows.append({"year": year, "group": group})
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    return df.groupby(["year", "group"]).size().reset_index(name="count")


timeline_group = "community" if color_by == "Detected community" else "category"
timeline_df = build_timeline(
    filtered_nodes, node_to_comm, G,
    group_by=timeline_group, bin_size=bin_years,
)

if not timeline_df.empty:
    # Build color map: "C0" -> color, "C1" -> color, etc.
    timeline_color_map = {}
    if timeline_group == "community":
        for grp in timeline_df["group"].unique():
            cid = int(grp.replace("C", "")) if grp.startswith("C") else -1
            timeline_color_map[grp] = comm_colors.get(cid, "#888888")

    fig_timeline = px.bar(
        timeline_df, x="year", y="count", color="group",
        labels={"year": "Publication Year", "count": "Papers", "group": ""},
        color_discrete_map=timeline_color_map if timeline_color_map else None,
        height=250,
    )
    fig_timeline.update_layout(
        margin=dict(l=0, r=0, t=30, b=0),
        barmode="stack",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    # Overlay community trend lines
    if timeline_group == "community":
        groups = sorted(timeline_df["group"].unique())
        for grp in groups:
            grp_df = timeline_df[timeline_df["group"] == grp].sort_values("year")
            if len(grp_df) < 2:
                continue
            # Get the community index for color lookup
            comm_idx = int(grp.replace("C", "")) if grp.startswith("C") else -1
            color = comm_colors.get(comm_idx, "#888888")
            fig_timeline.add_trace(go.Scatter(
                x=grp_df["year"],
                y=grp_df["count"],
                mode="lines",
                name=grp,
                line=dict(color=color, width=2),
                showlegend=False,
                hoverinfo="skip",
                opacity=0.7,
            ))

# ---------------------------------------------------------------------------
# Top papers table
# ---------------------------------------------------------------------------
def build_top_table(
    nodes: list[str], node_to_comm: dict[str, int], G: nx.DiGraph,
    pr: dict[str, float], in_deg: dict[str, int],
) -> pd.DataFrame:
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


table_df = build_top_table(filtered_nodes, node_to_comm, G, pr, in_deg)

# ---------------------------------------------------------------------------
# Community keyword extraction
# ---------------------------------------------------------------------------
def extract_community_keywords(
    nodes: list[str], node_to_comm: dict[str, int],
    G: nx.DiGraph, top_k: int = 8,
) -> dict[int, list[str]]:
    """Extract top TF-IDF-like keywords per community from paper titles."""
    comm_words: dict[int, Counter] = defaultdict(Counter)

    for n in nodes:
        title = G.nodes[n].get("title") or ""
        comm = node_to_comm.get(n, -1)
        words = re.findall(r"[a-z]{3,}", title.lower())
        words = [w for w in words if w not in _STOPWORDS]
        for w in words:
            comm_words[comm][w] += 1

    # TF-IDF-like: upweight words distinctive to a community
    n_comms = len(comm_words)
    result = {}
    for comm_id, word_counts in comm_words.items():
        scored = {}
        for word, count in word_counts.items():
            # How many communities use this word?
            n_with_word = sum(1 for c in comm_words if word in comm_words[c])
            # TF * IDF-like score
            tf = count / max(sum(word_counts.values()), 1)
            idf = np.log(1 + n_comms / max(n_with_word, 1))
            scored[word] = tf * idf
        top_words = sorted(scored, key=scored.get, reverse=True)[:top_k]
        result[comm_id] = top_words
    return result


comm_keywords = extract_community_keywords(filtered_nodes, node_to_comm, G)

# ---------------------------------------------------------------------------
# Page layout
# ---------------------------------------------------------------------------
# Stats bar (compact single line)
n_seeds = sum(1 for n in filtered_nodes if G.nodes[n].get("is_seed"))
n_meta = sum(1 for n in filtered_nodes if G.nodes[n].get("title"))
st.markdown(
    f"**{len(filtered_nodes)}** papers · **{filtered_G.number_of_edges()}** edges · "
    f"**{n_seeds}** seeds · **{n_communities}** communities · "
    f"**{n_meta}** with metadata · **{stats['nodes']}** total in DB",
)

# Main panels
col_left, col_right = st.columns([3, 1.4])

with col_left:
    st.subheader("Citation Network")
    # Click-to-inspect: use plotly click events
    selected = st.plotly_chart(
        fig_network, use_container_width=True, on_select="rerun",
        key="network_select",
    )

    # Export button
    st.download_button(
        "📥 Export network as HTML",
        data=fig_network.to_html(include_plotlyjs="cdn"),
        file_name="citation_network.html",
        mime="text/html",
        help="Download interactive plotly figure as HTML file",
    )

with col_right:
    st.subheader("Categories × Communities")
    if not confusion.empty:
        st.plotly_chart(fig_confusion, use_container_width=True)
    else:
        st.info("No seed papers in current filter")

    st.subheader("Timeline")
    if not timeline_df.empty:
        st.plotly_chart(fig_timeline, use_container_width=True)
    else:
        st.info("No papers with year data in current filter")

# ---------------------------------------------------------------------------
# Click-to-inspect: detect selection, update state, rerun if needed
# ---------------------------------------------------------------------------
clicked_node = None
clicked_community = None
if selected and selected.get("selection", {}).get("points"):
    for pt in selected["selection"]["points"]:
        curve = pt.get("curve_number")
        idx = pt.get("point_index", -1)
        if curve == node_trace_idx and 0 <= idx < len(filtered_nodes):
            clicked_node = filtered_nodes[idx]
            break
        elif label_trace_idx is not None and curve == label_trace_idx:
            if 0 <= idx < len(label_comm_ids):
                clicked_community = label_comm_ids[idx]
                break

# Update session state and rerun to apply highlights
just_reran = st.session_state.pop("_highlight_rerun", False)
if not just_reran:
    if clicked_node and clicked_node != sel_focus:
        st.session_state["selected_node"] = clicked_node
        st.session_state.pop("selected_community", None)
        st.session_state["_highlight_rerun"] = True
        st.rerun()
    elif clicked_community is not None and clicked_community != sel_community:
        st.session_state["selected_community"] = clicked_community
        st.session_state.pop("selected_node", None)
        st.session_state["_highlight_rerun"] = True
        st.rerun()

sel_node = st.session_state.get("selected_node", None)
if sel_node and sel_node not in filtered_set:
    sel_node = None
    st.session_state.pop("selected_node", None)

if sel_node is not None:
    sel_data = G.nodes[sel_node]

    col_title, col_clear = st.columns([5, 1])
    with col_title:
        st.subheader("🔍 Selected Paper")
    with col_clear:
        if st.button("✕ Clear", key="clear_selection"):
            st.session_state.pop("selected_node", None)
            st.session_state.pop("selected_community", None)
            st.session_state.pop("_highlight_rerun", None)
            st.rerun()
    sel_cols = st.columns([2, 1])
    with sel_cols[0]:
        st.markdown(f"**{sel_data.get('title', 'No title')}**")
        st.markdown(
            f"{sel_data.get('first_author', '?')} "
            f"({sel_data.get('publication_year', '?')})"
        )
        journal = sel_data.get("journal", "")
        if journal:
            st.markdown(f"*{journal}*")
        doi = sel_data.get("doi", "")
        if doi:
            st.markdown(f"[DOI: {doi}](https://doi.org/{doi})")
        st.markdown(f"[OpenAlex]({sel_node})")
    with sel_cols[1]:
        comm_id = node_to_comm.get(sel_node, -1)
        st.metric("Community", comm_id)
        st.metric("In-degree", in_deg.get(sel_node, 0))
        st.metric("Global citations", sel_data.get("cited_by_count", 0))
        st.metric("PageRank", f"{pr.get(sel_node, 0):.5f}")
        if sel_data.get("is_seed"):
            st.success(f"Seed: {sel_data.get('seed_category', '?')}")

    # Show neighbors in the filtered network
    sel_neighbors = list(filtered_G.predecessors(sel_node)) + list(filtered_G.successors(sel_node))
    if sel_neighbors:
        with st.expander(f"Network neighbors ({len(sel_neighbors)} in view)"):
            for nb in sorted(sel_neighbors, key=lambda n: in_deg.get(n, 0), reverse=True)[:15]:
                nb_data = G.nodes[nb]
                direction = "→" if filtered_G.has_edge(nb, sel_node) else "←"
                seed_mark = " 🌱" if nb_data.get("is_seed") else ""
                st.markdown(
                    f"  {direction} {nb_data.get('first_author', '?')} "
                    f"({nb_data.get('publication_year', '?')}){seed_mark} — "
                    f"{(nb_data.get('title') or '?')[:70]}"
                )

# Table
st.subheader("Paper Details")
st.dataframe(
    table_df.drop(columns=["OpenAlex ID"]),
    use_container_width=True,
    height=400,
    hide_index=True,
)

# ---------------------------------------------------------------------------
# Community summary panel
# ---------------------------------------------------------------------------
with st.expander("Community Summaries", expanded=False):
    for comm_id in sorted(set(node_to_comm.get(n, -1) for n in filtered_nodes)):
        if comm_id == -1:
            continue
        comm_nodes = [n for n in filtered_nodes if node_to_comm.get(n) == comm_id]
        seeds = [n for n in comm_nodes if G.nodes[n].get("is_seed")]
        cat_counts = Counter(G.nodes[n].get("seed_category", "") for n in seeds)
        top_by_deg = sorted(comm_nodes, key=lambda n: in_deg.get(n, 0), reverse=True)[:5]

        keywords = comm_keywords.get(comm_id, [])
        kw_str = ", ".join(keywords) if keywords else "(no keywords)"
        cat_str = ", ".join(f"{c} ({v})" for c, v in cat_counts.most_common(3) if c)

        color = comm_colors.get(comm_id, "#cccccc")
        st.markdown(
            f"### Community {comm_id} "
            f"<span style='background:{color};padding:2px 8px;border-radius:4px;'>"
            f"&nbsp;</span>",
            unsafe_allow_html=True,
        )
        st.markdown(f"**{len(comm_nodes)} papers**, {len(seeds)} seeds")
        st.markdown(f"**Keywords**: {kw_str}")
        if cat_str:
            st.markdown(f"**Dominant categories**: {cat_str}")

        st.markdown("**Top papers:**")
        for n in top_by_deg:
            d = G.nodes[n]
            tag = " 🌱" if d.get("is_seed") else ""
            st.markdown(
                f"  - {d.get('first_author', '?')} ({d.get('publication_year', '?')}){tag} "
                f"— *{(d.get('title') or '?')[:80]}* "
                f"(in-deg: {in_deg.get(n, 0)})"
            )
        st.divider()
