# Architecture

## Write path

`AnchorMemory.integrate()` validates endpoints, commits the memory and FTS state to SQLite, then handles explicit flow/semantic relations and bounded proposals. Chroma and Kuzu are projections. Projection failure must not roll back the SQLite authority record.

## Recall path

The live recall path combines FTS/BM25 and, when configured, Chroma vector seeds. It applies RRF, optional external rerank, bounded `flow_edges` diffusion, temporal update resolution, confidence/activation signals, and the independent Theseus shadow channel. `semantic_edges` never conduct heat.

Recall is candidate selection and therefore read-only. The caller confirms only final injected IDs through an idempotent heat event.

## Governance

Updates and EVOKES proposals remain outside authoritative semantic edges until the AI agent approves them. Raw clustering produces overlapping candidate groups and never writes an understanding. Beliefs have no outgoing conductive edges and cannot filter or bias recall.

## Projection boundary

Chroma stores text and vectors because it participates in vector recall. Kuzu stores graph identifiers and edge properties, not memory text. SQLite can rebuild the Kuzu node, FlowEdge, and SemanticEdge projection.
