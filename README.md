# HPCCitationNetwork

Citation network analysis for a review on **hippocampal recurrent circuit computation** — what computation does intrinsic recurrence in the hippocampus perform?

## Features (planned)
- Import seed papers from curated review document via DOI → OpenAlex
- Build directed citation networks with snowball sampling
- Unsupervised community detection to test theoretical category structure
- Gap detection: find influential papers missing from the review
- Theory/model filtering via LLM-based abstract classification
- Interactive visualization of citation landscapes

## Setup

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync --extra dev

# Run tests
uv run pytest tests/ -v
```

## Project Structure

```
src/
├── data/       — API clients, data collection, SQLite storage
├── network/    — Graph construction and analysis metrics
├── analysis/   — Co-citation, bibliographic coupling, temporal
└── viz/        — Plotting and visualization functions
tests/          — Unit and integration tests
data/raw/       — Downloaded data (gitignored)
data/processed/ — Analysis outputs (gitignored)
notebooks/      — Ad hoc exploration
```
