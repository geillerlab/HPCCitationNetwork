# HPCCitationNetwork

Build, analyze, and explore citation networks around specialized scientific topics.

## Features (planned)
- Collect papers and citations via OpenAlex API
- Build directed citation networks with snowball sampling
- Analyze network structure: centrality, communities, co-citation, temporal evolution
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
