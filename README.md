# Anchor Memory

Anchor Memory is a local-first memory graph for long-running AI agents. This repository is a traceable, sanitized extraction of the evolved live implementation: core algorithms and routing policy retain source lineage, while private data, identities, paths, credentials, deployment topology, and private channel adapters are excluded or made configurable.

**This is not an independent implementation inspired by Anchor Memory.** It is the sanitized release form of the allowlisted live codebase, with narrow public-boundary refactors documented below.

## What is real here

The end-to-end implementation includes:

- SQLite authority, CRUD, CJK FTS5, chat-history retrieval, and durable outboxes;
- live recall v2: BM25/FTS, real Chroma vector candidates when enabled, optional Voyage rerank, RRF, bounded flow diffusion, temporal update resolution, confidence and quality gates;
- separate conductive `flow_edges` and non-conductive `semantic_edges`;
- candidate recall with zero activation writes, followed by idempotent heat confirmation only for memory IDs actually injected;
- agent-owned review for updates and EVOKES proposals;
- Belief Graph v2, isolated from recall ranking and heat flow;
- Wenku/Theseus shadow validation, repair, staleness checks, hydration, and independent recall;
- the full reflex routing and evidence-scoring pipeline;
- briefing, dream review, raw clustering, maintenance, and the real health-dashboard logic.

Chroma is not merely a write mirror: when installed, its collection is queried by the live recall seed path. Kuzu remains a rebuildable graph projection and does not rank recall results.

## Agent ownership

Memory curation, candidate review, understanding synthesis, belief maintenance, and approval decisions belong to the AI agent operating the system. “Review” in this repository does not mean a hidden human operator. Mechanical jobs may propose or validate; they do not author the agent's understanding.

Legacy `read_by_human` columns are comment-viewer read receipts retained for schema compatibility; they do not assign memory review or synthesis authority.

## Install

Base mode uses SQLite/FTS and an offline deterministic embedding fallback:

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e ".[dev]"
pytest
```

Enable real projections:

```bash
pip install -e ".[chroma,kuzu]"
```

Enable the HTTP/MCP surfaces or a local sentence-transformer:

```bash
pip install -e ".[server,local-embedding]"
```

The base mode does not silently replace the production vector architecture with a SQLite vector shadow. It simply degrades to the same FTS branch already present in live recall. Installing Chroma reconnects the real vector branch.

## Rebuild projections from SQLite

The rebuild command requires a new output directory. It opens the authority
database read-only, takes a consistent SQLite backup, rebuilds both Chroma and
Kuzu, and exits nonzero unless all three counts agree. The output metadata also
records source/snapshot hashes, embedding provider, model, and dimension.

```bash
python scripts/rebuild_projections.py --source-db ./data/memories.db --output-dir ./projection-rebuild
```

The output contains memory text because it is a recovery bundle. Treat it as
private data and do not commit it. The command never overwrites an existing
directory and never promotes the rebuilt bundle automatically.

## Minimal use

```python
from anchor_memory import AnchorMemory

memory = AnchorMemory("./data")
memory.integrate(
    "The fictional observatory recalibrated its blue sensor.",
    "raw",
    memory_id="demo-001",
    auto_link=False,
)
result = memory.recall("blue sensor calibration", budget=3)

# Candidate recall is read-only. Confirm heat only after final context injection.
memory.db.apply_heat(
    [item["id"] for item in result["results"]],
    0.12,
    event_id="conversation-0001",
    spread=True,
    source="recall",
)
```

## Authentication and credentials

Credentials come only from environment variables or an explicitly configured external secret file. Gateway and hook endpoints fail closed when their authentication keys are absent. Example deployments must bind to loopback unless the operator supplies a real authentication and reverse-proxy layer.

No production database, vector index, graph database, prompt trace, review queue, log, domain, credential, personal name, or private channel adapter is included.

## Verification

The release gates include:

```bash
python scripts/verify_release_tree.py .
pytest tests/test_live_contract.py tests/test_full_chain_contract.py
pytest tests/test_real_projections.py tests/test_server_surface.py  # with server/Chroma/Kuzu extras
```

[`docs/completeness.md`](docs/completeness.md) maps every live pipeline stage to retained modules and behavioral evidence.

`docs/lineage.tsv` records original and release hashes, line counts, retained symbols, and exact-code-line retention without publishing private source paths. The current extraction retains 758 of 760 source functions/classes; the two omitted symbols belonged to a private production-only push adapter. The table is reproducible with `scripts/generate_lineage.py` when the private audit mirror is available.

## Intentional public refactors

- production absolute paths and identity-bearing environment names are generic configuration;
- external service credentials are environment-only and fail closed;
- Chroma/Kuzu/model dependencies are optional, with an offline FTS fallback;
- `count()` reports authoritative SQLite rows instead of Chroma cardinality;
- graph health reports `ok=false` whenever consistency is false;
- the private production push adapter is excluded;
- the live dashboard is retained as a configurable example rather than a copy of production report paths.

These changes are narrow release-boundary refactors. Recall scoring, graph propagation, temporal policy, Theseus validation, clustering, briefing, belief semantics, and reflex routing are not reimplemented substitutes.

## Upstream

Anchor Memory began from [Anchor Memory by Limen](https://github.com/limen-threshold/anchor-memory), under the MIT License. The downstream implementation is traceable to upstream commit [`8dd2133`](https://github.com/limen-threshold/anchor-memory/commit/8dd21337c2b87ff6d876dc269b475242650ea56e). This clean release preserves attribution without publishing private downstream Git history.
