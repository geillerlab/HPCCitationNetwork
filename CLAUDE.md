# HPCCitationNetwork

## Project Overview
Citation network analysis for a review paper on **hippocampal recurrent circuit computation**.
The review (with Jon Rueckemann) asks: what computation does intrinsic recurrence in HPC perform?
Theories: point attractors, continuous attractors, sequences, SR, BTSP, bespoke circuits.

Collects papers via OpenAlex API, builds directed citation graphs, and analyzes
network structure (communities, centrality, co-citation, temporal evolution).
**Key goals**: (1) find missing papers, (2) test whether citation communities match review categories.

Google Doc seed source: "HPC Recurrent Circuit" (ID: 1QlF34RElUYZ8C6aEjisSuwyysOhPy6M6naVMcEduIiA)

## Tech Stack
- Python 3.12, managed with `uv`
- Data: pandas, requests, SQLite
- Graphs: networkx, cdlib
- Viz: matplotlib, pyvis
- Testing: pytest, responses (for mocking HTTP)

## Commands
```bash
uv sync                    # Install dependencies
uv sync --extra dev        # Install with dev dependencies
uv run pytest tests/ -v    # Run tests
uv run python <script>     # Run any script
```

## Code Conventions
- Type hints on all function signatures
- Google-style docstrings for public functions
- 4-space indentation (PEP 8)
- Module layout:
  - `src/data/` — API clients, data collection, storage
  - `src/network/` — Graph construction and metrics
  - `src/analysis/` — Co-citation, coupling, temporal analysis
  - `src/viz/` — Plotting and visualization functions
- Tests mirror src: `tests/test_<module>.py`
- Use `pathlib.Path` for all file paths

## Data Rules
- `data/raw/` — Downloaded data. NEVER modify manually.
- `data/processed/` — Analysis outputs. Regenerable from raw.
- Both directories are gitignored. Keep data reproducible via code.

## Testing
- Mock all HTTP calls in tests (use `responses` library)
- Test edge cases: empty networks, disconnected components, duplicate papers
- Run `uv run pytest tests/ -v` after changes to src/
