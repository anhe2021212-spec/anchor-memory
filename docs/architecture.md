# Architecture

## How a memory gets in

`AnchorMemory.integrate()` validates the input, commits the memory and its FTS index to SQLite, then handles explicit edges and bounded link proposals. Chroma vectors and Kuzu graph entries are projections -- if either fails, the SQLite record stands. Nothing downstream can prevent a write from landing.

## How a memory comes back

Recall starts with the reflex router deciding whether the incoming message needs memory at all. If it does, the router picks a lane (full recall, cold-store lookup, person-entity search, or silence) and rewrites the query for precision.

The recall path itself fuses multiple signals: FTS/BM25 keyword matching, optional Chroma vector similarity, reciprocal-rank fusion, external reranking, bounded `flow_edges` heat diffusion, temporal update resolution, and confidence/activation weighting. A separate Theseus shadow channel searches the long-form document library independently.

Recall is candidate selection -- read-only. The caller decides which candidates to actually inject into context, then confirms those with an idempotent heat event. `semantic_edges` never conduct heat.

## Who decides what

The agent owns all semantic decisions. Mechanical jobs (clustering, taxonomy tagging, edge proposals) can suggest but never commit. EVOKES proposals sit in a review queue until the agent approves them. Understanding summaries are authored by the agent, not generated automatically.

Beliefs have no outgoing conductive edges. They appear in the briefing to inform the agent's thinking, but they cannot filter recall results or bias what gets surfaced. What actually happened (memories) and what the agent currently believes (beliefs) are structurally separated.

## Where data lives

SQLite is the single authority. Everything else is a projection:

- **Chroma** stores text and vectors because it participates in vector recall. Rebuildable from SQLite.
- **Kuzu** stores graph identifiers and edge properties, not memory text. Rebuildable from SQLite.
- **FTS5** index lives inside the SQLite file. Rebuilt automatically.

If Chroma or Kuzu disappear, the system degrades gracefully to SQLite-only recall. If SQLite disappears, everything is gone.
