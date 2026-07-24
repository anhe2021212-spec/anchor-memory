"""Unified durable write orchestration for Anchor phase C."""
from __future__ import annotations

import json
import uuid
from pathlib import Path

import dual_edge
from propose_links import propose_links


def _ids(value):
    if isinstance(value, str):
        value = value.split(",")
    return list(dict.fromkeys(str(x).strip() for x in (value or []) if str(x).strip()))


def _review(memory, source_id, target_id, role, strength=1.0, reason=""):
    import update_review
    if role == "EVOKES":
        return update_review.propose_evokes(source_id, target_id, reason or "integrate explicit edge")
    if role == "updates":
        row_new, row_old = memory.db.get(source_id), memory.db.get(target_id)
        item = {"id": update_review.pid("supersede", source_id, target_id), "kind": "supersede",
                "new_id": source_id, "old_id": target_id, "status": "pending",
                "source": "integrate_explicit_edge", "reason": update_review.summary(reason, 500),
                "new_summary": update_review.summary((row_new or {}).get("text", "")),
                "old_summary": update_review.summary((row_old or {}).get("text", ""))}
        return update_review.enqueue([item])
    raise ValueError("role does not use review repository")


def _validate_before_commit(memory, mid, level, collection, connect_to,
                            supported_by, updates, explicit_edges):
    """Validate every caller-supplied endpoint before SQLite becomes visible."""
    target_ids = set(_ids(connect_to)) | set(_ids(supported_by))
    if updates:
        target_ids.add(str(updates).strip())
    for edge in explicit_edges or []:
        target = str(edge.get("target_id", "")).strip()
        if not target:
            raise ValueError("explicit edge missing target_id")
        target_ids.add(target)
    if mid in target_ids:
        raise ValueError("self edge is not allowed")
    rows = {target: memory.db.get(target) for target in target_ids}
    missing = sorted(target for target, row in rows.items() if not row)
    if missing:
        raise ValueError("missing relation target ids: " + ",".join(missing))
    support = _ids(supported_by)
    if support and (level != "understanding" or collection == "wenku"):
        raise ValueError("supported_by requires a non-wenku understanding")
    invalid_support = [target for target in support if
        (rows[target].get("level") or "raw") != "raw" or
        (rows[target].get("collection") or "") == "wenku"]
    if invalid_support:
        raise ValueError("supported_by targets must be non-wenku raw: " +
                         ",".join(invalid_support))
    allowed = {"derived_from", "SUPPORTED_BY", "constellates", "EVOKES",
               "updates", "belief_member", "GROUNDED_IN"}
    for edge in explicit_edges or []:
        role = str(edge.get("role", "")).strip()
        if role not in allowed:
            raise ValueError("unknown explicit role: " + role)
        target = rows[str(edge.get("target_id", "")).strip()]
        target_level = target.get("level") or "raw"
        target_collection = target.get("collection") or ""
        if role == "SUPPORTED_BY" and not (
                level == "understanding" and collection != "wenku"
                and target_level == "raw" and target_collection != "wenku"):
            raise ValueError("invalid explicit SUPPORTED_BY endpoints")
        if role == "EVOKES" and not (
                collection != "wenku" and target_collection == "wenku"):
            raise ValueError("invalid explicit EVOKES endpoints")
        if role == "constellates" and not (
                level == "cognition" and collection != "wenku"
                and target_level == "cognition" and target_collection != "wenku"):
            raise ValueError("invalid explicit constellates endpoints")


def integrate(memory, content: str, level: str, *, tier: str = "long",
              emotion_score: float = 0.5, context: str = "", source_ref: str = "",
              tag: str = "", updates: str = "", connect_to="", supported_by="",
              grounded_in="", explicit_edges=None, auto_link: bool = True,
              link_budget: int = 20, memory_id: str | None = None,
              collection: str = "") -> dict:
    content = (content or "").strip()
    level = (level or "").strip().lower()
    if not content:
        raise ValueError("content must not be empty")
    if level not in {"raw", "understanding", "cognition"}:
        raise ValueError("level must be raw, understanding, or cognition")
    if not 1 <= int(link_budget) <= 500:
        raise ValueError("link_budget must be in 1..500")
    mid = memory_id or str(uuid.uuid4())[:8]
    collection = (collection or "").strip()
    _validate_before_commit(memory, mid, level, collection, connect_to,
                            supported_by, updates, explicit_edges)
    warnings, embedding_pending = [], False
    flow_count = semantic_count = review_count = 0
    memory.store(mid, content, tag=tag or "general", tier=tier,
                 emotion_score=emotion_score, level=level, collection=collection)
    with memory.db._conn() as conn:
        embedding_pending = conn.execute(
            "SELECT 1 FROM embedding_outbox WHERE memory_id=?", (mid,)
        ).fetchone() is not None
        fts_pending = conn.execute(
            "SELECT 1 FROM fts_map WHERE memory_id=?", (mid,)
        ).fetchone() is None
    if fts_pending:
        warnings.append("FTS projection missing; maintenance repair pending")
    if context:
        with memory.db._conn() as conn:
            conn.execute("UPDATE memories SET context=? WHERE memory_id=?", (context, mid))
            conn.commit()
    if source_ref:
        memory.db.annotate(mid, "source_ref:" + source_ref)

    manual_pairs = set()
    for target in _ids(connect_to):
        if not memory.db.connect(mid, target):
            warnings.append("connect_to ignored: " + target)
        else:
            flow_count += 2
            manual_pairs.add(tuple(sorted((mid, target))))

    support = _ids(supported_by)
    for target in support:
        result = memory.db.write_semantic_edge(mid, target, "SUPPORTED_BY", strength=1.0,
            conductance=0.0, confidence=1.0, provenance="structured", review_state="auto",
            created_by="machine", audit_note="integrate:" + mid)
        warnings.extend(result.get("warnings") or [])
        semantic_count += int(result.get("created", False))
    if _ids(grounded_in):
        warnings.append("GROUNDED_IN is read-only in phase C; ignored")

    if updates:
        _review(memory, mid, str(updates).strip(), "updates", reason="integrate updates parameter")
        review_count += 1

    for edge in explicit_edges or []:
        target, role = str(edge.get("target_id", "")).strip(), str(edge.get("role", "")).strip()
        strength = float(edge.get("strength", 1.0))
        note = str(edge.get("audit_note", ""))
        if role in {"derived_from", "SUPPORTED_BY", "constellates"}:
            result = memory.db.write_semantic_edge(mid, target, role, strength=strength,
                conductance=0.0, confidence=1.0, provenance="manual", review_state="approved",
                created_by="agent", audit_note=note)
            warnings.extend(result.get("warnings") or [])
            semantic_count += int(result.get("created", False))
        elif role in {"EVOKES", "updates"}:
            _review(memory, mid, target, role, strength, note)
            review_count += 1
        elif role == "belief_member":
            warnings.append("belief_member has no phase-C endpoint table; ignored")
        elif role == "GROUNDED_IN":
            warnings.append("GROUNDED_IN is read-only; ignored")

    if auto_link and not embedding_pending:
        for proposal in propose_links(
            [mid],
            relation_policy="all",
            budget=int(link_budget),
            db_path=Path(memory.db.db_path),
            chroma_path=Path(memory.db.db_path).parent / "chroma",
            collection=memory._collection_name,
        ):
            if proposal["disposition"] == "auto_flow":
                pair = tuple(sorted((proposal["source_id"], proposal["target_id"])))
                if pair in manual_pairs:
                    continue
                for direction in proposal["directions"]:
                    edge_result = memory.db.write_flow_edge(
                        direction["source_id"], direction["target_id"],
                        direction["weight"], direction["conductance"],
                        "auto_integrate", mode="auto")
                    warnings.extend(edge_result.get("warnings") or [])
                    flow_count += 1
            elif proposal["disposition"] == "auto_semantic":
                result = memory.db.write_semantic_edge(proposal["source_id"], proposal["target_id"],
                    proposal["proposed_role"], strength=1.0, conductance=0.0,
                    confidence=proposal["confidence"], provenance=proposal["provenance"],
                    review_state="auto", created_by="machine",
                    audit_note="integrate:" + mid)
                semantic_count += int(result.get("created", False))
            elif proposal["disposition"] == "review":
                review_count += 1

    summary = {"memory_id": mid, "level": level, "tier": tier,
        "flow_edges_created": flow_count, "semantic_edges_created": semantic_count,
        "review_proposals_created": review_count,
        "outbox_pending": {"kuzu": 0, "kuzu_node": 0, "kuzu_edge_legacy": 0,
                           "kuzu_edge_v2": 0, "embedding": int(embedding_pending)},
        "warnings": list(dict.fromkeys(warnings)), "embedding_pending": embedding_pending,
        "fts_pending": fts_pending}
    with memory.db._conn() as conn:
        for key, table in (("kuzu_node", "kuzu_node_outbox"),
                           ("kuzu_edge_legacy", "kuzu_edge_outbox"),
                           ("kuzu_edge_v2", "kuzu_edge_outbox_v2")):
            summary["outbox_pending"][key] = conn.execute(
                f"SELECT count(*) FROM {table}").fetchone()[0]
        summary["outbox_pending"]["kuzu"] = sum(
            summary["outbox_pending"][key] for key in
            ("kuzu_node", "kuzu_edge_legacy", "kuzu_edge_v2")
        )
    event_detail = {
        "node_committed": True, "embedding_ok": not embedding_pending,
        "fts_ok": not fts_pending, "flow_edges_created": flow_count,
        "semantic_edges_created": semantic_count,
        "review_proposals_created": review_count,
        "outbox_pending": summary["outbox_pending"], "warnings": summary["warnings"],
    }
    memory.db.log_event(mid, "integrated", json.dumps(
        event_detail, ensure_ascii=False, separators=(",", ":"))[:1000])
    summary["ok"] = "partial" if summary["warnings"] or embedding_pending or summary["outbox_pending"]["kuzu"] else True
    return summary


__all__ = ["integrate"]
