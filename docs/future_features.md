# Future Features

Ideas for dashboard and analysis improvements, saved for later implementation.

## High Priority — Paper Selection Improvements

### Seed cleanup
- Remove "general_attractor" papers — too broad, pull in unrelated work
- Review other seeds that are overly general (not specific to HPC recurrence)
- Consider: after snowball, check which seeds have low interconnectivity with the rest and flag for removal

### Network pruning: remove poorly-connected papers
- After multi-level snowball, papers with no connections to the core may be noise
- Research best practices: k-core decomposition, minimum citation overlap thresholds
- Restrict to papers within the giant connected component, or those connected to ≥N seeds
- Look into bibliometric methods for "relevance filtering" in snowball sampling

### LLM-based filtering: theory vs experimental papers
- Many snowballed papers are purely experimental with no model/theory
- Use LLM (Claude API) to classify papers by abstract: theory/model paper vs experimental
- Could also classify: does this paper propose or analyze a recurrent circuit mechanism?
- Needs abstracts — fetch from OpenAlex (available in API) for top papers
- Apply as a filter in the dashboard or as a pre-processing step

### Fix remaining "Unknown" papers
- Some papers in the dashboard still show "Unknown (no metadata)"
- These are stubs that weren't in the top-1000 metadata fetch
- Fix: either fetch metadata for all displayed papers, or run a broader metadata fill

## Medium Priority

### Hub analysis tab
- Guimerà & Amaral z-score vs participation coefficient scatter plot
- Provincial hubs (central within community) vs connector hubs (bridge communities)
- HITS algorithm (hubs vs authorities)
- Already planned as next major feature

### Missing papers panel
- Highly-cited papers in the network that aren't seeds
- Candidates to add to the review's reference list
- Sort by in-degree, filter out seeds, show community assignment

### Co-citation analysis
- Which papers are frequently cited together?
- Co-citation matrix → clustering reveals thematic groups
- Bibliographic coupling (shared references) as complementary signal

## Nice to Have

### Side-by-side comparison
- Two layouts or two parameter settings next to each other
- Useful for comparing community detection at different resolutions

### Seed coverage metric
- What fraction of the top-100 most-cited papers are seeds?
- Helps identify gaps in the review's coverage

### Abstract word clouds per community
- Requires fetching abstracts from OpenAlex (available in API)
- More informative than title-only keywords

## Infrastructure

### General-purpose package refactoring
- Separate HPC-specific logic (manual_seeds.py, seed_import.py) from reusable pipeline
- Package as installable tool: give it DOIs → get citation network + dashboard
- Config file instead of hardcoded category labels

### Deeper snowball
- Level-2 snowball on refined seed set
- Principled pruning: remove seeds with low connectivity to other seeds before expanding
