# HPCCitationNetwork

A general-purpose citation network analysis tool, currently applied to a review on **hippocampal recurrent circuit computation** — what computation does intrinsic recurrence in the hippocampus perform?

## What it does

1. **Seed import**: Takes a curated list of papers (DOIs from a Google Doc) and resolves them via the OpenAlex API
2. **Snowball sampling**: Expands outward from seeds — collects all references and citing papers at each level
3. **Network construction**: Builds a directed citation graph (paper A → paper B = A cites B) with full metadata
4. **Community detection**: Runs Louvain community detection on the citation network
5. **Interactive dashboard**: Explore the network, compare detected communities against your own categories, find missing papers

## Quick start

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync --extra dev

# Run the dashboard
uv run streamlit run app.py

# Run tests
uv run pytest tests/ -v
```

## Pipeline

### 1. Import seeds and snowball

```python
from src.data.openalex_client import OpenAlexClient
from src.data.storage import CitationDB
from src.data.collector import SnowballCollector

client = OpenAlexClient(email="your@email.com")
db = CitationDB()
collector = SnowballCollector(client, db)

# Import seeds from Google Doc text (with DOIs organized by category headers)
doc_text = "..."  # text with DOIs under category headings
collector.import_seeds(doc_text)

# Snowball: collect references and citing papers for all seeds
collector.collect_level(level=1, max_cited_by=200)

# Fill in metadata for top papers (stubs from snowball only have IDs)
collector.fetch_metadata_for_stubs(batch_size=100)
```

### 2. Build and explore the graph

```python
from src.network.builder import build_citation_graph, graph_summary, find_top_cited_in_network

G = build_citation_graph(db)
print(graph_summary(G))
top_papers = find_top_cited_in_network(G, n=20)
```

### 3. Launch the dashboard

```bash
uv run streamlit run app.py
```

## Dashboard controls

| Control | What it does |
|---------|-------------|
| **Top N papers** | Show the N most-cited papers in the network |
| **Color by** | Color nodes by detected community, review category, or publication year |
| **Node size by** | Size nodes by in-degree, global citation count, or PageRank |
| **Louvain resolution** | Tune community granularity (lower = fewer/larger, higher = more/smaller) |
| **Year range** | Filter by publication year |
| **Year bin size** | Group timeline bars into 1, 2, 5, or 10-year bins |
| **Filter categories** | Show/hide specific review categories |
| **Search** | Filter by author name or title keyword |

## Project structure

```
src/
├── data/
│   ├── openalex_client.py  — OpenAlex API wrapper (resolve DOI, fetch refs/cited_by)
│   ├── storage.py          — SQLite database (papers + citations tables)
│   ├── collector.py        — SnowballCollector: seed import + multi-level snowball
│   ├── seed_import.py      — DOI parser for Google Doc (HPC-specific)
│   └── manual_seeds.py     — Manual DOI corrections and cross-listings (HPC-specific)
├── network/
│   └── builder.py          — Build networkx DiGraph, subgraphs, summary stats
├── analysis/               — (planned) Co-citation, hub analysis, temporal
└── viz/
    └── plots.py            — Interactive (pyvis) and static (matplotlib) network plots
app.py                      — Streamlit dashboard
tests/                      — 36 passing tests (pytest + responses for HTTP mocking)
data/raw/                   — Downloaded data (gitignored)
data/processed/             — citations.db and generated visualizations (gitignored)
```

## Adapting for a new topic

This tool is designed for reuse. To analyze a different set of papers:

1. **Replace the seeds**: Either modify `src/data/seed_import.py` to parse your document format, or call `collector.import_seeds()` with your own DOI-containing text
2. **Update categories**: Edit `CATEGORY_COLORS` in `src/viz/plots.py` and category labels in `seed_import.py`
3. **Keep the pipeline**: Everything in `collector.py`, `storage.py`, `openalex_client.py`, `builder.py`, and `app.py` is topic-agnostic

The HPC-specific files are: `seed_import.py`, `manual_seeds.py`, and the category color map.

## Current data (HPC analysis)

- **97 seed papers** imported from review document, organized into 8 categories:
  point attractors, continuous attractors, sequences, successor representation, BTSP, bespoke circuits, autonomous dynamics, general attractor theory
- **9,748 total papers** after level-1 snowball sampling
- **17,969 citation edges**
- **~1,000 papers** with full metadata (seeds + top-cited)
- Louvain community detection finds 7-8 communities that largely correspond to review categories

## Tech stack

- **Python 3.12** managed with `uv`
- **Data**: pandas, requests, SQLite
- **Graphs**: networkx, cdlib (Louvain)
- **Dashboard**: Streamlit + Plotly
- **Visualization**: pyvis (interactive HTML), matplotlib (static)
- **Testing**: pytest, responses (HTTP mocking)
