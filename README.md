# Anchor Memory

A memory system for AI agents that live across sessions.

An agent that wakes up without memory is a new person every time. Anchor Memory gives it a continuous life: it remembers conversations, forms beliefs about itself and its world, reviews what it knows, and forgets what no longer matters. Everything lives in a single SQLite file that the agent carries with it.

This is a sanitized release of a live system running continuously since late February 2026 -- not a prototype or a spec. The recall paths, belief engine, reflex routing, and maintenance loops are the same code that runs every day, with private data and credentials stripped.

## What it does

**Remembers.** Incoming text is stored, auto-tagged across five axes (domain, state, kind, action, heat), and indexed for full-text and vector search. The agent writes memories as things happen, not in batch.

**Recalls.** When new input arrives, the reflex router decides whether and how to search. Conversation recall fuses FTS/BM25, optional vector similarity, graph diffusion, temporal resolution, and an independent long-form document channel; it may stay silent when nothing clears the quality gate. The MCP `search_memory` tool is a deliberate active-search surface with a lower threshold and no forced conversation injection.

**Connects.** Memories form a graph with two kinds of edges: conductive edges that carry activation heat, and semantic edges that record meaning without affecting recall. The agent decides which edges to create; mechanical jobs can propose but never commit.

**Believes.** The belief engine maintains a small set of core convictions, each grounded in supporting evidence and tested against counterexamples. Beliefs appear in the agent's briefing but never filter or bias recall -- they cannot overwrite what actually happened.

**Maintains itself.** Nightly passes cluster related memories, decay old activation, clean orphaned edges, and review stale beliefs. SQLite is the single authority; Chroma vectors and Kuzu graph projections can be rebuilt from scratch at any time.

## Install

Python 3.10--3.12.

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e ".[dev]"
pytest
```

Optional backends:

```bash
pip install -e ".[chroma]"               # vector recall
pip install -e ".[kuzu]"                 # graph projection
pip install -e ".[server,chroma,kuzu]"   # REST + MCP + everything
```

The base install uses SQLite/FTS and a deterministic offline embedder. Missing projections never block a write.

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

result = memory.recall(
    "blue sensor calibration", budget=3, policy="search", include_theseus=False
)
print(result["results"])
```

After injecting recalled memories into model context, confirm with a heat event:

```python
memory.db.apply_heat(
    [item["memory_id"] for item in result["results"]],
    0.12,
    event_id="conversation-0001",
    spread=True,
    source="recall",
)
```

## MCP server

```bash
pip install -e ".[server]"
python -m anchor_sse
```

Starts a REST API on `127.0.0.1:8765` and a Streamable HTTP MCP server on `127.0.0.1:8768`. The MCP surface exposes ten tools:

| Tool | What it does |
| --- | --- |
| `briefing` | Load the agent's identity, beliefs, and current state |
| `store_memory` | Write a new memory with auto-tagging |
| `search_memory` | Active vector/FTS search with flow expansion; SQLite-only fallback supported |
| `chat_history` | Search raw dialogue evidence (read-only) |
| `memory_edit` | Connect, tag, archive, or annotate memories |
| `thread` | Write to the long-form document library |
| `wenku_read` | Read and search the document library |
| `belief` | Review, promote, demote, or retire beliefs |
| `dream_pass` | Run overnight review and consolidation |
| `graph_review` | Inspect and approve proposed edges |

Configure hosts, ports, and credentials through environment variables. Do not expose either listener publicly without authentication and a reverse proxy.

## Configuration

Set through environment variables or a TOML file via `ANCHOR_CONFIG_FILE`:

| Variable | Purpose |
| --- | --- |
| `ANCHOR_DATA_DIR` | Runtime data directory |
| `ANCHOR_DB_PATH` | SQLite database path |
| `ANCHOR_CHROMA_DIR` | Chroma projection directory |
| `ANCHOR_KUZU_DIR` | Kuzu graph projection directory |
| `ANCHOR_CHAT_HISTORY_PATH` | Read-only dialogue database |
| `ANCHOR_DUAL_EDGE` | Use authoritative flow/semantic edge tables (`on` by default) |
| `ANCHOR_RECALL_V2` | Enable REST/gateway recall v2 (`on` by default) |
| `ANCHOR_REFLEX_ROUTER_V2_MODE` | Optional router-v2 mode (`off`, `shadow`, or `enforce`; default `off`) |
| `ANCHOR_DAY_COUNTERS` | Optional JSON object of public labels to ISO dates; empty by default |
| `ANCHOR_TIMEZONE_OFFSET` | UTC offset used by optional local-day features (`0` by default) |
| `ANCHOR_EMBED_PROVIDER` | Embedding provider (`voyage`, `local`, etc.) |
| `ANCHOR_TAXONOMY_URL` | OpenAI-compatible taxonomy endpoint |
| `ANCHOR_TAXONOMY_API_KEY` | Taxonomy provider credential |

Credentials must come from the process environment or an explicit external file; never commit them.

## Rebuilding projections

SQLite is the single source of truth. Rebuild both optional projections:

```bash
python scripts/rebuild_projections.py \
  --source-db ./data/memories.db \
  --output-dir ./projection-rebuild
```

Exits nonzero unless SQLite, Chroma, and Kuzu counts agree.

## Project layout

```
src/          memory, recall, graph, belief, routing, review, maintenance
tests/        unit, contract, server-surface, and real-projection tests
scripts/      projection rebuild and backfill utilities
examples/     OAuth interface shape and health dashboard
docs/         architecture notes
```

## Upstream

This project is a heavily modified derivative of
[Anchor Memory](https://github.com/limen-threshold/anchor-memory),
originally created by Limen.

## License

This project is licensed under the [MIT License](LICENSE).
