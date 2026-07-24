# Live-chain completeness evidence

This release is an allowlisted, sanitized extraction of the live Anchor Memory codebase. It is not a clean-room or design-principles reference implementation.

| Live stage | Retained implementation | Behavioral evidence |
| --- | --- | --- |
| Ingestion and authority | `anchor_memory.py`, `integrate_v2.py`, `anchor_db.py` | SQLite-first commit, FTS indexing, projection outbox, explicit relations, and no-Chroma degradation tests |
| Automatic taxonomy | `taxonomy_tagger.py`, `model_routes.py` | configured LLM route, tag validation, SQLite writeback, and `store_memory` trigger tests |
| Recall and temporal resolution | `recall_v2.py`, `recall_trace.py`, `shadow_index.py` | FTS/BM25, real Chroma seed, RRF, optional rerank, flow diffusion, temporal policy, read-only candidate tests |
| Association and clustering | `propose_links.py`, `cluster_raw.py`, `dual_edge.py` | real Chroma auto-link, bounded clustering, flow/semantic isolation, and 11 live cluster regressions |
| Agent approval | `update_review.py`, typed graph code | proposal remains non-authoritative until approval; approval changes the authoritative level and leaves a durable projection task |
| Beliefs, briefing, and dream review | `belief.py`, `belief_graph.py`, `anchor_sse.py` | confidence/case graph live regressions plus callable briefing and counterexample-review tests |
| Wenku/Theseus | `theseus_shadow_index.py` and backfill tool | validation, repair, staleness, hydration, and independent-recall regressions |
| Raw chat history | `cold_store.py` | read-only relay access, independent CJK FTS index, age gate, and bounded dialogue-window test |
| Reflex routing | `reflex_router.py`, `reflex_router_v2.py`, `reflex_router_v2_runtime.py`, `gateway.py` | full retained routing/evidence-scoring symbols and live gateway regression suite |
| Maintenance and observability | `maintenance.py`, dashboard example, rebuild scripts | dry-run and failure exit-code tests, sanitized review dashboard tests, and SQLite-to-Chroma/Kuzu rebuild count proof |
| Runtime surface | `anchor_sse.py` | exact ten-tool registration test: store, search, chat history, thread, Wenku, briefing, dream, edit, graph review, belief |

The public lineage covers 30 current live source files and retains 758 of 760 live functions/classes. The two omitted symbols are the private production push adapter. Private databases, identities, credentials, service topology, and channel-specific transport are deliberately outside the public release boundary; their exclusion does not replace or remove a memory-system stage.
