"""Interactive citation network dashboard.

Run with: uv run streamlit run app.py
"""

import streamlit as st
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from cdlib import algorithms
from collections import Counter

from src.data.storage import CitationDB
from src.network.builder import build_citation_graph, graph_summary
from src.viz.plots import CATEGORY_COLORS

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="HPC Citation Network",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("HPC Recurrent Circuit — Citation Network Explorer")


# ---------------------------------------------------------------------------
# Data loading (cached)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_graph():
    db = CitationDB()
    G = build_citation_graph(db)
    db.close()
    return G


@st.cache_data
def compute_metrics(_G):
    """Compute node metrics (PageRank, betweenness, etc.)."""
    pr = nx.pagerank(_G)
    in_deg = dict(_G.in_degree())
    out_deg = dict(_G.out_degree())
    return pr, in_deg, out_deg


@st.cache_data
def run_community_detection(_G_undirected, resolution):
    """Run Louvain with a given resolution parameter."""
    comms = algorithms.louvain(_G_undirected, resolution=resolution, randomize=False)
    node_to_comm = {}
    for i, comm in enumerate(comms.communities):
        for node in comm:
            node_to_comm[node] = i
    return node_to_comm, len(comms.communities)


G = load_graph()
pr, in_deg, out_deg = compute_metrics(G)

stats = graph_summary(G)

# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------
st.sidebar.header("Controls")

# Top N
top_n = st.sidebar.slider("Top N papers (by in-degree)", 50, 500, 200, step=50)

# Get top N nodes
top_nodes = sorted(in_deg, key=in_deg.get, reverse=True)[:top_n]
sub_G = G.subgraph(top_nodes).copy()
sub_U = sub_G.to_undirected()

# Color by
color_by = st.sidebar.selectbox(
    "Color by",
    ["Detected community", "Review category", "Publication year"],
)

# Node size by
size_by = st.sidebar.selectbox(
    "Node size by",
    ["In-degree (network)", "Global citation count", "PageRank"],
)

# Louvain resolution
resolution = st.sidebar.slider(
    "Louvain resolution", 0.3, 3.0, 1.0, step=0.1,
    help="Lower = fewer larger communities, higher = more smaller communities",
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
node_to_comm, n_communities = run_community_detection(sub_U, resolution)

# Category filter
all_categories = sorted(set(
    G.nodes[n].get("seed_category", "") for n in top_nodes if G.nodes[n].get("is_seed")
))
all_categories = [c for c in all_categories if c]

st.sidebar.header("Filters")
show_non_seeds = st.sidebar.checkbox("Show non-seed papers", value=True)
selected_categories = st.sidebar.multiselect(
    "Filter seed categories",
    all_categories,
    default=all_categories,
)

# Search
search_query = st.sidebar.text_input("Search (author or title)")

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
# Compute layout (cached per subgraph size)
# ---------------------------------------------------------------------------
@st.cache_data
def compute_layout(_nodes, _edges, seed):
    """Compute spring layout for the filtered graph."""
    tmp_G = nx.DiGraph()
    tmp_G.add_nodes_from(_nodes)
    tmp_G.add_edges_from(_edges)
    return nx.spring_layout(tmp_G, k=2.0 / (len(_nodes) ** 0.3), iterations=80, seed=seed)


pos = compute_layout(
    tuple(sorted(filtered_nodes)),
    tuple(sorted(filtered_G.edges())),
    seed=42,
)

# ---------------------------------------------------------------------------
# Color + size computation
# ---------------------------------------------------------------------------
# Generate community colors
def comm_color_map(n_comms):
    colors = px.colors.qualitative.Set3 + px.colors.qualitative.Pastel + px.colors.qualitative.Bold
    return {i: colors[i % len(colors)] for i in range(n_comms)}

comm_colors = comm_color_map(n_communities)

node_colors = []
node_sizes = []
node_texts = []
node_hovers = []

for n in filtered_nodes:
    data = G.nodes[n]
    is_seed = data.get("is_seed", False)
    cat = data.get("seed_category", "")
    year = data.get("publication_year") or "?"
    author = data.get("first_author") or "Unknown"
    title = data.get("title") or "(no metadata)"
    comm_id = node_to_comm.get(n, -1)

    # Color
    if color_by == "Review category":
        if is_seed:
            node_colors.append(CATEGORY_COLORS.get(cat, "#cccccc"))
        else:
            node_colors.append("#dddddd")
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

    # Hover
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
# Network graph (plotly)
# ---------------------------------------------------------------------------
# Edge traces
edge_x, edge_y = [], []
for u, v in filtered_G.edges():
    if u in pos and v in pos:
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

edge_trace = go.Scatter(
    x=edge_x, y=edge_y,
    line=dict(width=0.3, color="#cccccc"),
    hoverinfo="none",
    mode="lines",
)

# Node positions
node_x = [pos[n][0] for n in filtered_nodes]
node_y = [pos[n][1] for n in filtered_nodes]

# Handle year-based colorscale
if color_by == "Publication year":
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode="markers+text",
        hoverinfo="text",
        hovertext=node_hovers,
        text=node_texts,
        textposition="top center",
        textfont=dict(size=7),
        marker=dict(
            size=node_sizes,
            color=node_colors,
            colorscale="Viridis",
            colorbar=dict(title="Year"),
            line=dict(width=0.5, color="black"),
        ),
    )
else:
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode="markers+text",
        hoverinfo="text",
        hovertext=node_hovers,
        text=node_texts,
        textposition="top center",
        textfont=dict(size=7),
        marker=dict(
            size=node_sizes,
            color=node_colors,
            line=dict(width=0.5, color="black"),
        ),
    )

fig_network = go.Figure(
    data=[edge_trace, node_trace],
    layout=go.Layout(
        showlegend=False,
        hovermode="closest",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        margin=dict(l=0, r=0, t=30, b=0),
        height=650,
        plot_bgcolor="white",
    ),
)

# ---------------------------------------------------------------------------
# Confusion matrix: categories × communities
# ---------------------------------------------------------------------------
def build_confusion_matrix(nodes, node_to_comm, G):
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
    return pd.crosstab(df["category"], df["community"])


confusion = build_confusion_matrix(filtered_nodes, node_to_comm, G)

if not confusion.empty:
    fig_confusion = px.imshow(
        confusion,
        labels=dict(x="Detected Community", y="Review Category", color="Count"),
        color_continuous_scale="Blues",
        aspect="auto",
        text_auto=True,
    )
    fig_confusion.update_layout(height=400, margin=dict(l=0, r=0, t=30, b=0))

# ---------------------------------------------------------------------------
# Timeline: papers by year, stacked by community or category
# ---------------------------------------------------------------------------
def build_timeline(nodes, node_to_comm, G, group_by="community"):
    rows = []
    for n in nodes:
        data = G.nodes[n]
        year = data.get("publication_year")
        if year is None:
            continue
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
timeline_df = build_timeline(filtered_nodes, node_to_comm, G, group_by=timeline_group)

if not timeline_df.empty:
    fig_timeline = px.bar(
        timeline_df, x="year", y="count", color="group",
        labels={"year": "Publication Year", "count": "Papers", "group": ""},
        height=350,
    )
    fig_timeline.update_layout(
        margin=dict(l=0, r=0, t=30, b=0),
        barmode="stack",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

# ---------------------------------------------------------------------------
# Top papers table
# ---------------------------------------------------------------------------
def build_top_table(nodes, node_to_comm, G, pr, in_deg):
    rows = []
    for n in nodes:
        data = G.nodes[n]
        rows.append({
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
# Layout
# ---------------------------------------------------------------------------
# Stats bar
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Papers shown", len(filtered_nodes))
c2.metric("Edges", filtered_G.number_of_edges())
c3.metric("Seeds", sum(1 for n in filtered_nodes if G.nodes[n].get("is_seed")))
c4.metric("Communities", n_communities)
c5.metric("Total in DB", stats["nodes"])

# Main panels
col_left, col_right = st.columns([3, 2])

with col_left:
    st.subheader("Citation Network")
    st.plotly_chart(fig_network, use_container_width=True)

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

# Table
st.subheader("Paper Details")
st.dataframe(
    table_df,
    use_container_width=True,
    height=400,
    hide_index=True,
)

# ---------------------------------------------------------------------------
# Community composition summary
# ---------------------------------------------------------------------------
with st.expander("Community Composition Details"):
    for comm_id in sorted(set(node_to_comm.values())):
        comm_nodes = [n for n in filtered_nodes if node_to_comm.get(n) == comm_id]
        seeds = [n for n in comm_nodes if G.nodes[n].get("is_seed")]
        cat_counts = Counter(G.nodes[n].get("seed_category", "") for n in seeds)
        top_by_deg = sorted(comm_nodes, key=lambda n: in_deg.get(n, 0), reverse=True)[:5]

        cat_str = ", ".join(f"{c} ({v})" for c, v in cat_counts.most_common(3) if c)
        st.markdown(f"**Community {comm_id}** — {len(comm_nodes)} papers, {len(seeds)} seeds")
        if cat_str:
            st.markdown(f"  Dominant categories: {cat_str}")
        for n in top_by_deg:
            d = G.nodes[n]
            tag = " **[SEED]**" if d.get("is_seed") else ""
            st.markdown(
                f"  - {d.get('first_author', '?')} ({d.get('publication_year', '?')}){tag} "
                f"— in-deg: {in_deg.get(n, 0)}"
            )
