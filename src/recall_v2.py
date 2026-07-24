"""Side-effect-free Anchor recall v2 (phase C)."""
from __future__ import annotations

import math
import os
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


POLICIES = {
    "conversation": {"budget": 5, "theseus_budget": 2, "min_score": 0.25},
    "briefing": {"budget": 5, "theseus_budget": 1, "min_score": 0.15},
    "reflex": {"budget": 3, "theseus_budget": 1, "min_score": 0.25},
    "search": {"budget": 10, "theseus_budget": 0, "min_score": 0.05},
}
MAX_VECTOR_DISTANCE = 0.8
MAX_THESEUS_DISTANCE = 0.58
_HISTORY_QUERY_RE = re.compile(
    r"(以前|过去|当时|那时|那会|曾经|旧|老版本|历史|回顾|回忆|上次|之前|"
    r"timeline|archive|复盘|沿革|怎么变|为什么改)", re.I
)


def _temporal_mode(query: str, requested: str | None = None) -> str:
    """Current is the safe default; only explicit retrospective intent opens history."""
    requested = (requested or "").strip().lower()
    if requested in {"current", "historical"}:
        return requested
    return "historical" if _HISTORY_QUERY_RE.search(query or "") else "current"


def _state_tags(value: str) -> set[str]:
    return {part.strip().lower() for part in (value or "").split(",") if part.strip()}


def _temporal_label(item: dict) -> str | None:
    tags = _state_tags(item.get("tag", ""))
    return "past" if (
        "state:past" in tags or "state:obsolete" in tags or item.get("superseded_by")
    ) else None


def _voyage_rerank(query, documents):
    """Use the phase-B reranker contract; fail closed to the RRF scores."""
    if not documents:
        return None, "unavailable"
    key = os.environ.get("VOYAGE_API_KEY", "").strip()
    key_file = os.environ.get("VOYAGE_KEY_FILE", "").strip()
    if not key and key_file:
        secret = Path(key_file).expanduser()
        if secret.is_file():
            for line in secret.read_text(encoding="utf-8").splitlines():
                if line.startswith("VOYAGE_API_KEY="):
                    key = line.split("=", 1)[1].strip().strip('"').strip("'")
                    break
    if not key:
        return None, "unavailable"
    try:
        import httpx
        with httpx.Client(timeout=8.0, trust_env=False) as client:
            response = client.post("https://api.voyageai.com/v1/rerank",
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                json={"model": "rerank-2.5-lite", "query": query,
                      "documents": [x[:1200] for x in documents], "top_k": len(documents)})
        response.raise_for_status()
        score_map = {int(x["index"]): float(x["relevance_score"])
                     for x in response.json().get("data", [])}
        return [max(0.0, min(1.0, score_map.get(i, 0.0)))
                for i in range(len(documents))], "ok"
    except Exception:
        return None, "failed_fallback"


def _parse_time(value):
    try:
        return datetime.fromisoformat((value or "").replace("Z", "+00:00")).astimezone(timezone.utc)
    except Exception:
        return None


def _freshness(value, now):
    dt = _parse_time(value)
    if dt is None:
        return 0.4
    days = max(0.0, (now - dt).total_seconds() / 86400)
    return 1.0 if days <= 7 else 0.8 if days <= 30 else 0.6 if days <= 90 else 0.4


def _vector_rows(memory, embedding, n, where=None):
    count = memory._collection.count()
    if not count:
        return []
    raw = memory._collection.query(
        query_embeddings=[embedding], n_results=min(int(n), count), where=where,
        include=["documents", "metadatas", "distances"],
    )
    rows = []
    for cid, doc, meta, dist in zip(raw["ids"][0], raw["documents"][0], raw["metadatas"][0], raw["distances"][0]):
        meta = meta or {}
        rows.append({"memory_id": meta.get("memory_id", cid), "text": doc or "",
                     "meta": meta, "distance": float(dist)})
    return rows


def _seed_candidates(memory, query, embedding, fetch_n, level_filter):
    vector = []
    for row in _vector_rows(memory, embedding, fetch_n):
        if row["distance"] > MAX_VECTOR_DISTANCE or row["meta"].get("collection") == "wenku":
            continue
        if level_filter and row["meta"].get("level", "raw") not in level_filter:
            continue
        vector.append(row)
    bm25 = memory.db.bm25_search(query, limit=fetch_n, collection="exclude")
    rrf, cache = defaultdict(float), {}
    for rank, row in enumerate(vector):
        mid = row["memory_id"]
        rrf[mid] += 1.0 / (61 + rank)
        cache[mid] = row
    vector_ids = set(cache)
    for rank, row in enumerate(bm25):
        mid = row["memory_id"]
        rrf[mid] += (1.5 if mid not in vector_ids else 1.0) / (61 + rank)
        if mid not in cache:
            item = memory.db.get(mid)
            if item and (not level_filter or item.get("level", "raw") in level_filter):
                cache[mid] = {"memory_id": mid, "text": item.get("text", ""),
                              "meta": item, "distance": None}
    values = [rrf[x] for x in rrf if x in cache]
    low, spread = (min(values), max(values) - min(values)) if values else (0.0, 0.0)
    rows = []
    for mid in sorted(rrf, key=rrf.get, reverse=True):
        if mid not in cache:
            continue
        similarity = 0.5 if spread <= 1e-12 else (rrf[mid] - low) / spread
        item = dict(cache[mid])
        item["query_similarity"] = similarity
        rows.append(item)
    scores, rerank_status = _voyage_rerank(query, [x["text"] for x in rows])
    if scores is not None:
        for item, score in zip(rows, scores):
            item["rrf_similarity"] = item["query_similarity"]
            item["query_similarity"] = score
        rows.sort(key=lambda x: (-x["query_similarity"], x["memory_id"]))
    return rows[:fetch_n], rerank_status


def _diffuse(db, seeds, depth=3, total_budget=25):
    now = datetime.now(timezone.utc)
    # Seed graph score is deliberately zero: the graph did not discover it.
    result = {mid: {"score": 0.0, "mass": max(0.0, float(score)), "depth": 0, "path": [mid]}
              for mid, score in seeds.items()}
    frontier, visited = list(seeds), set(seeds)
    for level in range(1, depth + 1):
        nxt = []
        for source in frontier:
            rows = db.get_flow_neighbors(source, direction="outgoing", min_weight=0.0, limit=500)
            weighted = [(r, max(0.0, float(r["weight"]) * float(r["conductance"]) *
                                  _freshness(r.get("last_fired"), now))) for r in rows]
            denom = sum(x[1] for x in weighted)
            if denom <= 0:
                continue
            for edge, effective in sorted(weighted, key=lambda x: (-x[1], x[0]["memory_id"])):
                target = edge["memory_id"]
                if target in visited or len(visited) >= total_budget:
                    continue
                score = result[source]["mass"] * 0.85 * effective / denom
                visited.add(target)
                result[target] = {"score": score, "mass": score, "depth": level,
                                  "path": result[source]["path"] + [target]}
                nxt.append(target)
        frontier = nxt
        if not frontier or len(visited) >= total_budget:
            break
    return result


def _superseded_by(db, mid):
    seen, current = {mid}, mid
    for _ in range(5):
        rows = db.get_semantic_neighbors(current, direction="incoming", roles=["updates"],
                                         review_state=None, limit=20)
        rows = [x for x in rows if x.get("review_state") in {"approved", "auto"}]
        if not rows:
            return current if current != mid else None
        newest = sorted(rows, key=lambda x: x.get("created", ""), reverse=True)[0]["memory_id"]
        if newest in seen:
            return None
        current = newest
        seen.add(current)
    return current if current != mid else None


def _apply_temporal_policy(memory, ranked: list[dict], mode: str) -> list[dict]:
    """Collapse approved update chains only for current answers; keep the wide pool intact."""
    mode = _temporal_mode("", mode)
    if mode == "historical":
        out = []
        for source in ranked:
            item = dict(source)
            label = _temporal_label(item)
            if label:
                item["temporal_label"] = label
            out.append(item)
        return out

    collapsed: dict[str, dict] = {}
    order: list[str] = []
    for source in ranked:
        item = dict(source)
        old_id = item.get("memory_id")
        newest_id = item.get("superseded_by")
        if newest_id:
            row = memory.db.get(newest_id)
            if row and (row.get("collection") or "") != "wenku":
                item.update({
                    "memory_id": newest_id,
                    "text": row.get("text", ""),
                    "snippet": row.get("text", ""),
                    "tag": row.get("tag", ""),
                    "tier": row.get("tier", "long"),
                    "level": row.get("level", "raw"),
                    "timestamp": row.get("timestamp", ""),
                    "resolved_from": old_id,
                    "superseded_by": None,
                })
                breakdown = item.get("score_breakdown") or {}
                penalty = float(breakdown.get("temporal_penalty") or 0.0)
                if penalty < 0:
                    item["score"] = round(float(item.get("score") or 0.0) - penalty * .05, 6)
                    item["score_breakdown"] = {**breakdown, "temporal_penalty": 0.0}

        tags = _state_tags(item.get("tag", ""))
        if "state:obsolete" in tags:
            continue
        label = _temporal_label(item)
        if label:
            item["temporal_label"] = label

        mid = item.get("memory_id")
        if not mid:
            continue
        previous = collapsed.get(mid)
        if previous is None:
            collapsed[mid] = item
            order.append(mid)
        elif float(item.get("score") or 0.0) > float(previous.get("score") or 0.0):
            if previous.get("resolved_from") and not item.get("resolved_from"):
                item["resolved_from"] = previous["resolved_from"]
            collapsed[mid] = item
        elif item.get("resolved_from") and not previous.get("resolved_from"):
            previous["resolved_from"] = item["resolved_from"]
    return sorted((collapsed[mid] for mid in order),
                  key=lambda x: (-float(x.get("score") or 0.0), x["memory_id"]))


def _confidence(db, mid):
    with db._conn() as conn:
        row = conn.execute(
            "SELECT max(confidence) FROM semantic_edges WHERE review_state IN ('approved','auto') "
            "AND (source_id=? OR target_id=?)", (mid, mid)).fetchone()
    return float(row[0] or 0.0)


def _anchor_item(memory, mid, query_similarity, graph_score, source_channel, path):
    row = memory.db.get(mid)
    if not row or (row.get("collection") or "") == "wenku":
        return None
    activation = min(max(float(row.get("activation_score") or 0.0) / 8.0, 0.0), 1.0)
    confidence = _confidence(memory.db, mid)
    superseded = _superseded_by(memory.db, mid)
    temporal = 1.0 if superseded else 0.0
    score = .45 * query_similarity + .15 * activation + .15 * graph_score + .10 * confidence - .05 * temporal
    return {"memory_id": mid, "text": row.get("text", ""), "snippet": row.get("text", ""),
            "tag": row.get("tag", ""), "tier": row.get("tier", "long"),
            "level": row.get("level", "raw"), "timestamp": row.get("timestamp", ""),
            "score": round(score, 6), "source_channel": source_channel, "evoked_by": None,
            "superseded_by": superseded, "path": path,
            "score_breakdown": {"query_similarity": round(query_similarity, 6),
                "base_activation": round(activation, 6), "graph_diffusion": round(graph_score, 6),
                "confidence_bonus": round(confidence, 6), "temporal_penalty": -round(temporal, 6),
                "novelty_bonus": 0.0}}


def _theseus_shadow_rows(memory, embedding, n, parent_id=None):
    """Query the dedicated Theseus shadow collection and hydrate safe chunks.

    Chroma ids are tsh_<hash>, never Wenku memory ids. Every result therefore
    maps through metadata.parent_memory_id and is rejected if its source hash
    no longer matches the current parent text.
    """
    try:
        import theseus_shadow_index
        collection = memory._client.get_collection(
            name=theseus_shadow_index.COLLECTION_NAME)
        count = collection.count()
        if not count:
            return []
        kwargs = {
            "query_embeddings": [embedding],
            "n_results": min(max(1, int(n)), count),
            "include": ["documents", "metadatas", "distances"],
        }
        if parent_id:
            kwargs["where"] = {"parent_memory_id": str(parent_id)}
        raw = collection.query(**kwargs)
    except Exception:
        return []

    rows, seen = [], set()
    ids = (raw.get("ids") or [[]])[0]
    docs = (raw.get("documents") or [[]])[0]
    metas = (raw.get("metadatas") or [[]])[0]
    distances = (raw.get("distances") or [[]])[0]
    for sid, index_doc, meta, distance in zip(ids, docs, metas, distances):
        meta = meta or {}
        pid = str(meta.get("parent_memory_id") or "")
        if not pid or pid in seen or (parent_id and pid != str(parent_id)):
            continue
        parent = memory.db.get(pid)
        if not parent or (parent.get("collection") or "") != "wenku":
            continue
        current_hash = theseus_shadow_index.source_hash(parent.get("text") or "")
        if meta.get("source_hash") != current_hash:
            continue
        try:
            hydrated = theseus_shadow_index.hydrate_hit(
                memory.db.db_path, pid, int(meta["chunk_no"]), char_budget=1200)
        except Exception:
            hydrated = None
        hit = (hydrated or {}).get("hit") or {}
        if not hydrated or hit.get("source_hash") != current_hash:
            continue
        text = hydrated.get("primary_text") or hit.get("text") or index_doc or ""
        seen.add(pid)
        rows.append({"memory_id": pid, "shadow_id": sid, "text": text,
                     "meta": meta, "distance": float(distance), "parent": parent})
    return rows


def _theseus(memory, embedding, anchor_rows, budget, min_score, free_embedding=None):
    candidates = {}
    for anchor in anchor_rows:
        for edge in memory.db.get_semantic_neighbors(anchor["memory_id"], direction="outgoing",
                                                     roles=["EVOKES"], review_state="approved", limit=50):
            target = edge["memory_id"]
            shadow_rows = _theseus_shadow_rows(memory, embedding, 8, parent_id=target)
            shadow_id = None
            if shadow_rows:
                shadow_row = shadow_rows[0]
                relevance = max(0.0, min(1.0, 1.0 - shadow_row["distance"]))
                text = shadow_row["text"]
                shadow_id = shadow_row["shadow_id"]
            else:
                got = memory._collection.get(ids=[target], include=["embeddings", "documents", "metadatas"])
                embeddings = got.get("embeddings")
                if not got.get("ids") or embeddings is None or len(embeddings) == 0:
                    continue
                vec = embeddings[0]
                dot = sum(float(a) * float(b) for a, b in zip(embedding, vec))
                norm = math.sqrt(sum(float(a) ** 2 for a in embedding) * sum(float(b) ** 2 for b in vec)) or 1.0
                relevance = max(0.0, min(1.0, dot / norm))
                text = (got.get("documents") or [""])[0]
            strength = float(edge.get("strength") or 1.0)
            score = min(1.0, anchor["score"] * strength * relevance + .10 * float(edge.get("confidence") or 0.0))
            candidates[target] = {"wenku_id": target, "memory_id": target,
                "shadow_id": shadow_id, "text": text, "score": round(score, 6),
                "source_channel": "evokes", "evoked_by": anchor["memory_id"]}
    free_embedding = free_embedding or embedding
    for row in _theseus_shadow_rows(memory, free_embedding, max(10, budget * 5)):
        if row["distance"] > MAX_THESEUS_DISTANCE:
            continue
        mid = row["memory_id"]
        stored = row["parent"]
        relevance = max(0.0, 1.0 - row["distance"])
        activation = min(max(float(stored.get("activation_score") or 0.0) / 8.0, 0.0), 1.0)
        score = .45 * relevance + .15 * activation
        item = {"wenku_id": mid, "memory_id": mid, "shadow_id": row["shadow_id"],
                "text": row["text"], "score": round(score, 6),
                "source_channel": "free", "evoked_by": None}
        if mid not in candidates or score > candidates[mid]["score"]:
            candidates[mid] = item
    return sorted((x for x in candidates.values() if x["score"] >= min_score),
                  key=lambda x: (-x["score"], x["memory_id"]))[:budget]


def recall(memory, query: str, *, budget: int | None = None, allow_empty: bool = True,
           policy: str = "conversation", include_theseus: bool = True,
           theseus_budget: int | None = None, level_filter: list[str] | None = None,
           min_score: float | None = None, context: str = "",
           temporal_mode: str | None = None) -> dict:
    if policy not in POLICIES:
        raise ValueError("unknown recall policy")
    preset = POLICIES[policy]
    budget = preset["budget"] if budget is None else max(1, int(budget))
    theseus_budget = preset["theseus_budget"] if theseus_budget is None else max(0, int(theseus_budget))
    min_score = preset["min_score"] if min_score is None or float(min_score) <= 0 else float(min_score)
    if not include_theseus:
        theseus_budget = 0
    theseus_budget = min(theseus_budget, budget)
    query = (query or "").strip()
    if not query:
        return {"results": [], "theseus_results": [], "total_candidates_scored": 0,
                "empty_reason": "query empty"}
    embedding = memory._encode_query(query)
    seeds, rerank_status = _seed_candidates(memory, query, embedding, budget * 3, level_filter)
    graph = _diffuse(memory.db, {x["memory_id"]: x["query_similarity"] for x in seeds},
                     depth=3, total_budget=budget * 5)
    anchor = {}
    for seed in seeds:
        mid = seed["memory_id"]
        item = _anchor_item(memory, mid, seed["query_similarity"], 0.0,
                            "main", [mid])
        if item:
            anchor[mid] = item
    for mid, info in graph.items():
        if info["depth"] == 0 or mid in anchor:
            continue
        item = _anchor_item(memory, mid, 0.0, info["score"],
                            "graph_diffusion", info["path"])
        if item and (not level_filter or item["level"] in level_filter):
            anchor[mid] = item
    ranked = sorted(anchor.values(), key=lambda x: (-x["score"], x["memory_id"]))
    temporal_mode = _temporal_mode(query, temporal_mode)
    ranked = _apply_temporal_policy(memory, ranked, temporal_mode)
    eligible_main = [x for x in ranked if x["score"] >= min_score]
    # Theseus is an attachment to the precise main route, not a static
    # reservation.  Keep one main slot whenever a qualified main result exists;
    # otherwise Theseus may use the whole budget.  Only slots actually filled
    # by Theseus are deducted, so unused association quota returns to main.
    main_floor = 1 if eligible_main else 0
    effective_theseus_budget = min(theseus_budget, budget - main_floor)
    free_embedding = memory._encode_query(query + "\n" + context) if context else embedding
    theseus = (_theseus(memory, embedding, ranked, effective_theseus_budget,
                        min_score, free_embedding)
               if effective_theseus_budget else [])
    main = eligible_main[:budget - len(theseus)]
    if not allow_empty and not main and not theseus and ranked:
        main = ranked[:1]
    return {"results": main, "theseus_results": theseus,
            "total_candidates_scored": len(anchor) + len(theseus),
            "empty_reason": None if main or theseus else "no candidates above threshold",
            "policy": policy, "temporal_mode": temporal_mode, "min_score": min_score,
            "rerank_model": "rerank-2.5-lite", "rerank_status": rerank_status,
            "read_only": {"activation": False, "last_fired": False, "hebbian": False, "review_state": False}}


__all__ = ["recall", "POLICIES"]
