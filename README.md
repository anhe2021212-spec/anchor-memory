# Anchor Memory

A local-first memory system for AI agents. SQLite is the single source of truth
— every node, edge, belief, activation score, and maintenance event lives in one
portable file. Optional vector stores and graph projections improve retrieval
but never gate a write.

This is not a RAG wrapper. Anchor Memory is a full cognition chain: from the
moment a message arrives, through reflex routing, multi-signal recall, belief
reasoning, and overnight maintenance — the complete loop that turns
conversations into durable, retrievable, evolving memory.

## Who authors and reviews the memory

The **memory-owning AI agent** is the subject, semantic author, and intended
reviewer of this system. The human operator supplies infrastructure; they are
not the curator of the agent's memories.

Throughout the code and documentation, `manual`, `review`, `annotate`,
`approve`, `reject`, `promote`, and `demote` mean deliberate actions performed
by the AI agent itself. Automated components may preserve evidence, retrieve
candidates, or prepare bounded proposals, but they do not transfer semantic
authorship to a human administrator.

## What's inside

```
message in
    │
    ▼
┌─────────┐    ┌───────────┐    ┌──────────┐    ┌──────────┐
│  Reflex  │───▶│   Recall  │───▶│  Inject  │───▶│   Heat   │
│  Router  │    │  Pipeline │    │   Gate    │    │  Confirm │
└─────────┘    └───────────┘    └──────────┘    └──────────┘
  15 intent      FTS5 + vec       adapter         idempotent
  classes        + RRF + flow     decides         activation
                 + rerank

              ┌───────────┐    ┌──────────┐    ┌──────────┐
              │  Beliefs  │    │ Theseus  │    │   Chat   │
              │  Engine   │    │  Wenku   │    │ History  │
              └───────────┘    └──────────┘    └──────────┘
              confidence +      chunk valid.    FTS5 + CJK
              dream review      + hydration     read-only

              ┌───────────┐    ┌──────────┐
              │ Taxonomy  │    │  Night   │
              │ (LLM)     │    │  Batch   │
              └───────────┘    └──────────┘
              closed 5-axis     decay, merge
              with retry        cluster, heal
```

**Reflex router** — 15 intent classes with bilingual (EN + CJK) pattern
matching. Each class maps to allowed recall lanes, query rewriting bounds, and
answer mode. Deterministic decision IDs for tracing.

**Recall pipeline** — FTS5 full-text, vector similarity, reciprocal-rank fusion,
flow-edge diffusion, temporal resolution, and reranking. Multiple signals fuse
into one ranked candidate list.

**Belief system** — Beliefs are hypotheses with confidence scores from
emotion-weighted, time-decayed cases. Support, contradict, or bound a belief.
Promote and demote with audit trails. `dream_pass` runs counterexample review
overnight.

**Theseus / Wenku** — Lossless chunk validation (consecutive numbering, source
reconstruction, label bounds, enum membership). Deterministic policy
enforcement. Neighboring-chunk hydration under character budget.

**Taxonomy** — LLM-backed 5-axis closed vocabulary (`state`, `domain`, `action`,
`kind`, `heat`). Durable retry outbox for failed classifications.

**Maintenance** — Activation decay, anti-island cleanup, LLM-driven memory
clustering, health checks. Every command supports `--dry-run` and `--json`.

**Recall evaluation** — Trace → extract → annotate → score. Measures
`precision_strict`, `precision_loose`, pool-hit split, and silence pass rate.
The memory-owning AI agent annotates captured cases in a separate reflective
pass; the decision pipeline being evaluated cannot assign its own verdict.

## Quick start

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e .

# generate a synthetic demo database
python scripts/generate_demo_db.py --data-dir ./data --json

# recall
anchor-demo --config config/anchor.example.toml recall "violet coolant"

# health check
anchor-maintenance health --config config/anchor.example.toml --json
```

No API keys needed for the default local embedding and rerank providers.

On Windows: `.venv\Scripts\activate`.

### Optional projections

```bash
pip install -e ".[chroma]"    # vector store
pip install -e ".[kuzu]"      # graph projection
pip install -e ".[full]"      # both + jieba CJK segmentation
```

Set `ANCHOR_CHROMA_DIR` or `ANCHOR_KUZU_DIR` to enable. Missing packages never
block SQLite writes — projection work is queued in a durable outbox and drained
by `shadow-refresh` in the background.

## MCP server

```bash
pip install -e ".[server,mcp]"
python -m anchor_memory.api.mcp
```

Ten tools: `briefing`, `store_memory`, `search_memory`, `chat_history`,
`memory_edit`, `thread`, `wenku_read`, `belief`, `dream_pass`, `graph_review`.

Mechanical maintenance and injection confirmation are internal operations, not
daily tools.

## REST API

```bash
uvicorn anchor_memory.api.rest:create_app --factory --host localhost --port 8080
```

Candidate recall is read-only. A channel adapter calls the injection
confirmation endpoint only after selecting the exact IDs inserted into model
context. Stable event IDs make heat idempotent.

## Tests

```bash
pip install -e ".[dev]"
pytest
```

The suite includes unit, integration, and contract tests. Optional projection
tests run when Chroma/Kuzu are installed.

To reproduce the clean Python matrix:

```bash
python scripts/run_python_matrix.py --output evidence.json
```

Builds managed Python 3.10/3.11/3.12 environments, runs the full suite,
validates dependency licenses, and emits path-free JSON evidence.

## Documentation

- [Architecture](docs/architecture.md)
- [Data model](docs/data-model.md)
- [Recall pipeline](docs/recall-pipeline.md)
- [Full cognition chain](docs/full-chain.md)
- [Maintenance](docs/maintenance.md)
- [Security](SECURITY.md)

## Privacy

The clean-tree builder uses a versioned allowlist, verifies local import
closure, rejects symlinks and runtime artifacts, and writes a SHA-256 manifest.
The release verifier reports only rule names, file names, and line numbers — it
never echoes a suspected secret.

This repository contains only synthetic examples. Do not create fixtures by
exporting and renaming production memories.

## Lineage

This project is a heavily modified derivative of
[Anchor Memory](https://github.com/limen-threshold/anchor-memory) by Limen.
The downstream fork shares Git history with upstream through
[`8dd2133`](https://github.com/limen-threshold/anchor-memory/commit/8dd21337c2b87ff6d876dc269b475242650ea56e)
(v1.3.2, April 2026), then evolved independently with substantial architectural
changes to storage, recall, belief, routing, maintenance, and evaluation.

This public repository is rebuilt from a sanitized source tree. It preserves the
upstream MIT license but does not include private downstream Git history.

## License

[MIT](LICENSE)
