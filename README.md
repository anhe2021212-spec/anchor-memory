# Anchor Memory

Anchor Memory is a local-first memory system for long-running AI agents. SQLite is the authoritative store; optional vector and graph backends improve retrieval without gating writes.

It covers the full memory loop: ingestion, automatic taxonomy, recall, association, agent review, beliefs, briefing, counterexample review, chat-history evidence, and maintenance.

## Features

- **SQLite authority and CJK FTS5** — memories, edges, activation, events, and full-text search remain portable.
- **Multi-signal recall** — FTS, optional Chroma vectors, reciprocal-rank fusion, reranking, bounded flow diffusion, and temporal resolution.
- **Two graph planes** — conductive `flow_edges` are separate from non-conductive semantic relations.
- **Agent-owned review** — mechanical jobs may prepare proposals; the memory-owning AI agent approves semantic changes and authors higher-level understanding.
- **Belief engine** — confidence from weighted support, contradiction, and boundary cases, with promotion, demotion, briefing, and dream review.
- **Theseus / Wenku** — chunk validation, policy checks, staleness detection, repair, hydration, and independent recall.
- **Reflex routing** — intent-sensitive recall lanes, bounded query rewriting, evidence scoring, and silence when recall is not useful.
- **Raw chat history** — read-only dialogue evidence through an independent CJK FTS index and bounded context windows.
- **Maintenance and recovery** — clustering, decay, edge cleanup, health checks, durable projection outboxes, and full Chroma/Kuzu rebuilds from SQLite.

## Install

Python 3.10–3.12 is supported.

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e ".[dev]"
pytest
```

On Windows, activate the environment with `.venv\Scripts\activate`.

Optional components:

```bash
pip install -e ".[chroma]"               # Chroma vector recall
pip install -e ".[kuzu]"                 # Kuzu graph projection
pip install -e ".[server,chroma,kuzu]"   # REST + MCP + projections
```

The base installation uses SQLite/FTS and a deterministic offline embedder. Missing optional projections never block an authoritative SQLite write.

## Quick start

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
print(result["results"])
```

Candidate recall is read-only. After the caller actually injects selected memories into model context, it can confirm heat with a stable event ID:

```python
memory.db.apply_heat(
    [item["memory_id"] for item in result["results"]],
    0.12,
    event_id="conversation-0001",
    spread=True,
    source="recall",
)
```

## REST and MCP

Install the server extra and start both loopback listeners:

```bash
pip install -e ".[server]"
python -m anchor_sse
```

Defaults:

- REST: `127.0.0.1:8765`
- Streamable HTTP MCP: `127.0.0.1:8768`

Configure them with `ANCHOR_REST_HOST`, `ANCHOR_REST_PORT`, `ANCHOR_STREAMABLE_HOST`, and `ANCHOR_STREAMABLE_PORT`. Do not expose either listener publicly without authentication and a reverse proxy.

The MCP surface provides ten tools: `briefing`, `store_memory`, `search_memory`, `chat_history`, `memory_edit`, `thread`, `wenku_read`, `belief`, `dream_pass`, and `graph_review`.

## Configuration

Runtime paths are configurable through environment variables or a TOML file passed with `ANCHOR_CONFIG_FILE`. Common settings include:

| Variable | Purpose |
| --- | --- |
| `ANCHOR_DATA_DIR` | Runtime data directory |
| `ANCHOR_DB_PATH` | Authoritative SQLite database |
| `ANCHOR_CHROMA_DIR` | Chroma projection directory |
| `ANCHOR_KUZU_DIR` | Kuzu projection directory |
| `ANCHOR_CHAT_HISTORY_PATH` | Read-only dialogue database |
| `ANCHOR_EMBED_PROVIDER` | Embedding provider selection |
| `ANCHOR_TAXONOMY_URL` | OpenAI-compatible taxonomy endpoint |
| `ANCHOR_TAXONOMY_API_KEY` | Taxonomy provider credential |
| `ANCHOR_TAXONOMY_MODEL` | Taxonomy model name |

Credentials must come from the process environment or an explicitly configured external file; do not commit them.

## Rebuild projections

SQLite can rebuild both optional projections into a new directory:

```bash
python scripts/rebuild_projections.py \
  --source-db ./data/memories.db \
  --output-dir ./projection-rebuild
```

The command exits nonzero unless SQLite, Chroma, and Kuzu counts agree. The output is private runtime data and should not be committed.

## Project layout

- `src/` — memory, recall, graph, belief, routing, review, and maintenance modules
- `tests/` — unit, contract, server-surface, and real-projection tests
- `scripts/` — projection rebuild and backfill utilities
- `examples/` — OAuth interface shape and health dashboard
- `docs/architecture.md` — storage, recall, governance, and projection boundaries

## Upstream and license

This project is a heavily modified derivative of [Anchor Memory](https://github.com/limen-threshold/anchor-memory) by Limen. Upstream attribution is preserved in [NOTICE](NOTICE).

Released under the [MIT License](LICENSE).
