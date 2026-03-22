# Methods

## Data Collection

### Seed papers
Seed papers are imported from a curated Google Doc containing DOIs organized under category headings (e.g., "Point Attractors", "Sequences"). DOIs are extracted via regex, cleaned (handling parentheses, trailing punctuation, and URL-encoded characters), and resolved through the OpenAlex API to obtain canonical OpenAlex identifiers and metadata.

Papers appearing under multiple category headings are cross-listed (assigned to all relevant categories). Manual corrections handle edge cases: old-style DOIs with special characters, papers with non-standard URL formats, and missing entries.

**Current dataset**: 97 seed papers across 8 theoretical categories.

### Snowball sampling
Starting from the seed papers, citation neighbors are collected via the OpenAlex API:

- **References**: Papers cited by each seed (outgoing edges). These are available directly in the OpenAlex work record.
- **Cited-by**: Papers that cite each seed (incoming edges). Fetched via paginated API queries, capped at 200 per seed to manage network size.

This produces a level-1 snowball: all papers within one citation hop of any seed. The process can be extended to additional levels, though the network grows rapidly (level-1 alone yields ~9,700 papers from 97 seeds).

Level-1 papers are initially stored as stubs (OpenAlex ID only). Full metadata (title, authors, year, journal, abstract, citation count) is fetched separately for the most-connected papers to keep API usage manageable.

### Data storage
All data is stored in a local SQLite database with two tables:
- **papers**: One row per paper (OpenAlex ID, DOI, title, authors, year, journal, citation count, seed status, category, snowball level)
- **citations**: One row per directed citation edge (citing_id → cited_id)

Self-citations are filtered out at insertion time.

## Network Construction

The citation graph is a directed graph where:
- **Nodes** = papers (OpenAlex IDs)
- **Edges** = citation relationships (A → B means paper A cites paper B)

Node attributes include all available metadata plus derived fields (is_seed, seed_category, snowball_level). The graph is constructed from the SQLite database using networkx.

### Subgraph extraction
For visualization and analysis, subgraphs are extracted by:
- **Top-N by in-degree**: The N most-cited papers within the network (not global citation count — in-degree reflects importance within this specific literature)
- **Seed-only**: Just the seed papers and edges between them
- **Level-based**: Papers up to a given snowball level

## Metrics

### Node-level
- **In-degree** (within network): Number of papers in the collected network that cite this paper. Reflects importance within the specific literature being studied.
- **Global citation count**: Total citations reported by OpenAlex (includes citations from outside the collected network).
- **PageRank**: Recursive importance measure — papers cited by important papers receive higher scores. Computed on the full directed graph.

### Network-level
- **Density**: Fraction of possible edges that exist
- **Weakly connected components**: Number of disconnected subgraphs
- **Average in/out degree**: Mean citations per paper

## Community Detection

Communities are detected using the **Louvain algorithm** (Blondel et al., 2008) on an undirected version of the top-N citation subgraph. The resolution parameter controls granularity:
- Lower resolution → fewer, larger communities
- Higher resolution → more, smaller communities

The algorithm optimizes modularity — it finds groups of papers that cite each other more than expected by chance. Community assignments are used for:
1. **Validation**: Comparing detected communities against the manually assigned review categories (confusion matrix)
2. **Discovery**: Identifying papers that bridge communities or that cluster unexpectedly

### Validation approach
The confusion matrix cross-tabulates review categories (rows) against detected communities (columns) for seed papers only. Good correspondence (diagonal dominance) indicates that the citation structure reflects the theoretical groupings used in the review. Off-diagonal entries highlight papers that the citation network associates with a different community than their assigned category — these may represent cross-disciplinary work or categorization ambiguities.

Rows and columns are reordered so that the largest values fall along the diagonal, making the correspondence visually clear.

## Timeline Analysis

Publication years are visualized as stacked bar charts with community-colored segments, showing how the literature has evolved over time. Community trend lines are overlaid to highlight temporal patterns in each research area.

An adjustable bin size (1, 2, 5, or 10 years) allows smoothing of the temporal distribution, which is useful when per-year counts are small.

## Planned Analyses

### Hub analysis (Guimerà & Amaral framework)
Classification of papers by their structural role:
- **Within-module degree z-score (z_in)**: How connected a paper is within its own community, relative to other members
- **Participation coefficient (P)**: How evenly a paper's connections are distributed across communities

This yields a typology:
- **Provincial hubs** (high z_in, low P): Central within their community, few cross-community connections
- **Connector hubs** (high z_in, high P): Central locally AND bridge to other communities
- **Non-hub connectors** (low z_in, high P): Bridge papers that aren't locally important

### HITS algorithm
Distinguishes between:
- **Hubs**: Papers that comprehensively cite important work (e.g., review papers)
- **Authorities**: Papers cited by many hub papers (seminal works)

### LLM-based topic analysis
Using language models to analyze abstracts and identify differential topics across detected communities — moving beyond citation structure to semantic content.

### Co-citation and bibliographic coupling
- **Co-citation**: Papers frequently cited together (shared intellectual ancestry)
- **Bibliographic coupling**: Papers that cite the same sources (shared methodology or framework)

## References

- Blondel, V. D., Guillaume, J. L., Lambiotte, R., & Lefebvre, E. (2008). Fast unfolding of communities in large networks. *Journal of Statistical Mechanics*, P10008.
- Guimerà, R., & Amaral, L. A. N. (2005). Functional cartography of metabolic networks. *Nature*, 433, 895–900.
- Kleinberg, J. M. (1999). Authoritative sources in a hyperlinked environment. *Journal of the ACM*, 46(5), 604–632.
- OpenAlex API: https://docs.openalex.org/
