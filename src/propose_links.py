"""Read-only link proposal engine for Anchor Memory phase C."""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, TypedDict

import numpy as np

import cluster_raw
from release_config import AnchorConfig

RelationPolicy = Literal["flow", "semantic", "all"]
POLICY_VERSION = os.environ.get("ANCHOR_LINK_POLICY_VERSION", "phase-c-v1")
PRIMARY = float(os.environ.get("ANCHOR_LINK_PRIMARY", "0.82"))
DEGREE_CAP = int(os.environ.get("ANCHOR_LINK_DEGREE_CAP", "12"))
MAX_PER_SOURCE = int(os.environ.get("ANCHOR_LINK_MAX_PER_SOURCE", "2"))
AUTO_WEIGHT_MIN = float(os.environ.get("ANCHOR_LINK_WEIGHT_MIN", "0.12"))
AUTO_WEIGHT_MAX = float(os.environ.get("ANCHOR_LINK_WEIGHT_MAX", "0.24"))
AUTO_CONDUCTANCE = float(os.environ.get("ANCHOR_LINK_CONDUCTANCE", "0.25"))
_CONFIG = AnchorConfig.load()
DB_PATH = _CONFIG.db_path
CHROMA_PATH = _CONFIG.chroma_dir or (_CONFIG.data_dir / "chroma")


def _default_collection() -> str:
    provider = os.environ.get("ANCHOR_EMBED_PROVIDER", "local").strip().lower()
    if provider == "voyage":
        suffix = os.environ.get("VOYAGE_COLLECTION_SUFFIX", "voyage4_1024")
        return f"memories_{suffix}"
    return "memories"


COLLECTION = os.environ.get("ANCHOR_CHROMA_COLLECTION", _default_collection())
REVIEW_PATH = Path(os.environ.get("ANCHOR_UPDATE_REVIEW_QUEUE", _CONFIG.review_dir / "update_review_queue.json"))


class Direction(TypedDict):
    source_id: str
    target_id: str
    weight: float
    conductance: float


class LinkEvidence(TypedDict, total=False):
    provider: str
    score: float
    detail: dict


class LinkProposal(TypedDict):
    proposal_id: str
    source_id: str
    target_id: str
    edge_family: Literal["flow", "semantic"]
    proposed_role: str | None
    directions: list[Direction]
    confidence: float
    priority_score: float
    provenance: str
    disposition: Literal["auto_flow", "auto_semantic", "review", "drop"]
    review_state: Literal["auto", "pending"] | None
    evidence: list[LinkEvidence]
    policy_version: str
    created_at: str
    expires_at: str | None


def normalize_embeddings(matrix: np.ndarray) -> np.ndarray:
    """Normalize rows; kept public so providers can share the exact kernel."""
    return cluster_raw._normalize_matrix(np.asarray(matrix, dtype=np.float32))


def cosine_pairs(source: np.ndarray, targets: np.ndarray) -> np.ndarray:
    if len(targets) == 0:
        return np.asarray([], dtype=np.float32)
    left = normalize_embeddings(np.asarray([source], dtype=np.float32))[0]
    right = normalize_embeddings(np.asarray(targets, dtype=np.float32))
    return right @ left


def bounded_clusters(items: list[dict], primary: float = PRIMARY,
                     secondary: float = 0.65, max_hops: int = 2) -> list[list[dict]]:
    if len(items) < 2:
        return []
    matrix = normalize_embeddings(np.stack([item["embedding"] for item in items]))
    similarity = matrix @ matrix.T
    adjacency = [set() for _ in items]
    for left in range(len(items)):
        for right in range(left + 1, len(items)):
            if float(similarity[left, right]) >= primary:
                adjacency[left].add(right)
                adjacency[right].add(left)
    remaining = {idx for idx, neighbors in enumerate(adjacency) if neighbors}
    clusters: list[list[dict]] = []
    while remaining:
        seed = max(remaining, key=lambda idx: (len(adjacency[idx]), -idx))
        candidate = {seed}
        frontier = {seed}
        for _ in range(max_hops):
            next_frontier = {
                neighbor
                for idx in frontier
                for neighbor in adjacency[idx]
                if neighbor in remaining and neighbor not in candidate
            }
            if not next_frontier:
                break
            candidate |= next_frontier
            frontier = next_frontier
        candidate = cluster_raw._baseline_enforce_secondary(
            candidate, similarity, secondary
        )
        if len(candidate) >= 2:
            members = [
                items[idx]
                for idx in sorted(
                    candidate,
                    key=lambda idx: (items[idx].get("timestamp") or "", idx),
                )
            ]
            clusters.append(members)
            remaining -= candidate
        else:
            remaining.discard(seed)
    return clusters


def fingerprint_proposal(edge_family: str, role: str | None, source_id: str,
                         target_id: str, evidence: list[LinkEvidence]) -> str:
    if edge_family == "flow":
        source_id, target_id = sorted((source_id, target_id))
    keys = sorted(
        f"{x.get('provider','')}:{json.dumps(x.get('detail', {}), ensure_ascii=False, sort_keys=True, separators=(',', ':'))}"
        for x in evidence
    )
    payload = "\x1f".join((POLICY_VERSION, edge_family, role or "", source_id, target_id, *keys))
    return "link_" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def rank_proposals(rows: list[LinkProposal], budget: int) -> list[LinkProposal]:
    return sorted(rows, key=lambda x: (-x["priority_score"], x["proposal_id"]))[:budget]


def _connection(db_path: Path = DB_PATH) -> sqlite3.Connection:
    c = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=10)
    c.row_factory = sqlite3.Row
    c.execute("PRAGMA query_only=ON")
    return c


def _load_pool(c: sqlite3.Connection, chroma_path: Path = CHROMA_PATH,
               collection: str = COLLECTION) -> list[dict]:
    """Load the shared embedding pool, then attach authoritative SQLite levels."""
    try:
        import chromadb
        col = chromadb.PersistentClient(path=str(chroma_path)).get_collection(collection)
    except Exception:
        return []
    got = col.get(include=["embeddings"])
    embeddings = got.get("embeddings")
    if embeddings is None:
        return []
    vectors = dict(zip(got.get("ids") or [], embeddings))
    if not vectors:
        return []
    rows = c.execute(
        "SELECT memory_id,COALESCE(level,'raw') level,"
        "COALESCE(collection,'') collection,timestamp FROM memories"
    ).fetchall()
    return [{**dict(row), "id": row["memory_id"],
             "embedding": np.asarray(vectors[row["memory_id"]], dtype=np.float32)}
            for row in rows if row["memory_id"] in vectors]


def _load_specific_embeddings(node_ids: list[str], chroma_path: Path = CHROMA_PATH,
                              collection: str = COLLECTION) -> dict[str, np.ndarray]:
    try:
        import chromadb
        client = chromadb.PersistentClient(path=str(chroma_path))
        col = client.get_collection(collection)
    except Exception:
        return {}
    out: dict[str, np.ndarray] = {}
    for start in range(0, len(node_ids), 500):
        got = col.get(ids=node_ids[start:start + 500], include=["embeddings"])
        embeddings = got.get("embeddings")
        if embeddings is None:
            continue
        for mid, emb in zip(got.get("ids") or [], embeddings):
            if emb is not None:
                out[mid] = np.asarray(emb, dtype=np.float32)
    return out


def _suppressed_ids() -> set[str]:
    out: set[str] = set()
    if REVIEW_PATH.exists():
        try:
            data = json.loads(REVIEW_PATH.read_text(encoding="utf-8"))
            for row in data.get("proposals", []):
                if row.get("status") in {"rejected", "skipped", "expired"}:
                    out.add(str(row.get("id") or row.get("proposal_id") or ""))
                    if row.get("fingerprint"):
                        out.add(str(row["fingerprint"]))
        except (OSError, ValueError, TypeError):
            pass
    return out


def _node_rows(c: sqlite3.Connection, node_ids: list[str]) -> dict[str, dict]:
    marks = ",".join("?" for _ in node_ids)
    rows = c.execute(
        f"SELECT memory_id,COALESCE(level,'raw') level,COALESCE(collection,'') collection,"
        f"timestamp FROM memories WHERE memory_id IN ({marks})", node_ids,
    ).fetchall()
    result = {r["memory_id"]: dict(r) for r in rows}
    missing = [x for x in node_ids if x not in result]
    if missing:
        raise ValueError("missing node ids: " + ",".join(missing[:10]))
    return result


def _degree(c: sqlite3.Connection, mid: str) -> int:
    return int(c.execute(
        "SELECT COUNT(DISTINCT neighbor) FROM ("
        "SELECT target_id neighbor FROM flow_edges WHERE source_id=? "
        "UNION SELECT source_id neighbor FROM flow_edges WHERE target_id=?"
        ")", (mid, mid)
    ).fetchone()[0])


def _exists_flow(c: sqlite3.Connection, a: str, b: str) -> bool:
    return c.execute(
        "SELECT 1 FROM flow_edges WHERE (source_id=? AND target_id=?) OR "
        "(source_id=? AND target_id=?) LIMIT 1", (a, b, b, a),
    ).fetchone() is not None


def _exists_semantic(c: sqlite3.Connection, a: str, b: str, role: str) -> bool:
    return c.execute(
        "SELECT 1 FROM semantic_edges WHERE source_id=? AND target_id=? AND role=? LIMIT 1",
        (a, b, role),
    ).fetchone() is not None


def _created_at(meta: dict, target_meta: dict | None = None) -> str:
    values = [meta.get("timestamp") or ""]
    if target_meta:
        values.append(target_meta.get("timestamp") or "")
    value = max(values)
    return value or datetime.now(timezone.utc).isoformat()


def _embedding_provider(c: sqlite3.Connection, node_ids: list[str], meta: dict[str, dict],
                        skip_existing: bool, chroma_path: Path = CHROMA_PATH,
                        collection: str = COLLECTION) -> list[LinkProposal]:
    if Path(chroma_path) == Path(CHROMA_PATH) and collection == COLLECTION:
        # Preserve the live helper contract for callers/tests that patch the
        # module defaults while allowing release instances to pass isolated paths.
        pool = _load_pool(c)
        source_embeddings = _load_specific_embeddings(node_ids)
    else:
        pool = _load_pool(c, chroma_path, collection)
        source_embeddings = _load_specific_embeddings(node_ids, chroma_path, collection)
    targets = {x["id"]: x for x in pool}
    rows: list[LinkProposal] = []
    suppressed = _suppressed_ids()
    for source_id in node_ids:
        source_meta = meta[source_id]
        if source_meta["collection"] == "wenku":
            # Phase C never auto-bridges Wenku to Anchor.
            continue
        source_vec = source_embeddings.get(source_id)
        if source_vec is None:
            continue
        source_level = source_meta["level"]
        allowed_levels = ({"raw"} if source_level == "raw" else
                          {"understanding"} if source_level == "understanding" else
                          {"raw", "understanding"} if source_level == "cognition" else set())
        available = [x for x in pool if x["id"] != source_id
                     and x["collection"] != "wenku" and x["level"] in allowed_levels]
        scores = cosine_pairs(source_vec, np.stack([x["embedding"] for x in available])) if available else []
        ranked = sorted(zip(available, scores), key=lambda item: (-float(item[1]), item[0]["id"]))
        accepted = 0
        source_degree = _degree(c, source_id)
        for target, score_raw in ranked:
            score = float(score_raw)
            target_id = target["id"]
            if score < PRIMARY or source_degree + accepted >= DEGREE_CAP or _degree(c, target_id) >= DEGREE_CAP:
                continue
            if skip_existing and _exists_flow(c, source_id, target_id):
                continue
            scaled = (score - PRIMARY) / max(1e-9, 1.0 - PRIMARY)
            weight = AUTO_WEIGHT_MIN + max(0.0, min(1.0, scaled)) * (AUTO_WEIGHT_MAX - AUTO_WEIGHT_MIN)
            evidence: list[LinkEvidence] = [{"provider": "embedding", "score": score,
                "detail": {"threshold": PRIMARY, "cosine": round(score, 6)}}]
            pid = fingerprint_proposal("flow", None, source_id, target_id, evidence)
            if pid in suppressed:
                continue
            a, b = sorted((source_id, target_id))
            rows.append({"proposal_id": pid, "source_id": a, "target_id": b,
                "edge_family": "flow", "proposed_role": None,
                "directions": [{"source_id": a, "target_id": b, "weight": round(weight, 6), "conductance": AUTO_CONDUCTANCE},
                               {"source_id": b, "target_id": a, "weight": round(weight, 6), "conductance": AUTO_CONDUCTANCE}],
                "confidence": round(score, 6), "priority_score": round(score, 6),
                "provenance": "cluster", "disposition": "auto_flow", "review_state": None,
                "evidence": evidence, "policy_version": POLICY_VERSION,
                "created_at": _created_at(source_meta, targets.get(target_id)), "expires_at": None})
            accepted += 1
            if accepted >= max(1, min(MAX_PER_SOURCE, 2)):
                break
    return rows


def _structured_provider(c: sqlite3.Connection, node_ids: list[str], meta: dict[str, dict],
                         skip_existing: bool, chroma_path: Path = CHROMA_PATH,
                        collection: str = COLLECTION) -> list[LinkProposal]:
    rows: list[LinkProposal] = []
    suppressed = _suppressed_ids()
    for source_id in node_ids:
        if meta[source_id]["level"] != "understanding" or meta[source_id]["collection"] == "wenku":
            continue
        annotations = c.execute("SELECT text FROM annotations WHERE memory_id=?", (source_id,)).fetchall()
        raw_ids: list[str] = []
        for annotation in annotations:
            raw_ids.extend(cluster_raw._parse_source_raw_ids(annotation[0]))
        for target_id in sorted(set(raw_ids)):
            target = c.execute(
                "SELECT memory_id,COALESCE(level,'raw') level,COALESCE(collection,'') collection,timestamp "
                "FROM memories WHERE memory_id=?", (target_id,)).fetchone()
            if not target or target["level"] != "raw" or target["collection"] == "wenku":
                continue
            if skip_existing and _exists_semantic(c, source_id, target_id, "SUPPORTED_BY"):
                continue
            evidence: list[LinkEvidence] = [{"provider": "structured_source", "score": 1.0,
                "detail": {"annotation_key": "source_raw_ids"}}]
            pid = fingerprint_proposal("semantic", "SUPPORTED_BY", source_id, target_id, evidence)
            if pid in suppressed:
                continue
            rows.append({"proposal_id": pid, "source_id": source_id, "target_id": target_id,
                "edge_family": "semantic", "proposed_role": "SUPPORTED_BY",
                "directions": [{"source_id": source_id, "target_id": target_id,
                                "weight": 1.0, "conductance": 0.0}],
                "confidence": 1.0, "priority_score": 1.0, "provenance": "structured",
                "disposition": "auto_semantic", "review_state": "auto", "evidence": evidence,
                "policy_version": POLICY_VERSION, "created_at": _created_at(meta[source_id], dict(target)),
                "expires_at": None})
    return rows


def propose_links(node_ids: Sequence[str], *, relation_policy: RelationPolicy = "flow",
                  budget: int = 50, skip_existing: bool = True,
                  db_path: Path | None = None, chroma_path: Path | None = None,
                  collection: str | None = None) -> list[LinkProposal]:
    ids = list(dict.fromkeys(str(x).strip() for x in node_ids if str(x).strip()))
    if not ids:
        raise ValueError("node_ids must not be empty")
    if relation_policy not in {"flow", "semantic", "all"}:
        raise ValueError("relation_policy must be flow, semantic, or all")
    if not 1 <= int(budget) <= 500:
        raise ValueError("budget must be in 1..500")
    resolved_db = Path(db_path) if db_path is not None else DB_PATH
    resolved_chroma = Path(chroma_path) if chroma_path is not None else CHROMA_PATH
    resolved_collection = collection or COLLECTION
    with _connection(resolved_db) as c:
        meta = _node_rows(c, ids)
        rows: list[LinkProposal] = []
        if relation_policy in {"flow", "all"}:
            rows.extend(_embedding_provider(c, ids, meta, skip_existing, resolved_chroma, resolved_collection))
        if relation_policy in {"semantic", "all"}:
            rows.extend(_structured_provider(c, ids, meta, skip_existing))
    # Canonical proposal id dedupe before budget.
    deduped = {x["proposal_id"]: x for x in rows}
    return rank_proposals(list(deduped.values()), int(budget))


__all__ = ["propose_links", "normalize_embeddings", "cosine_pairs", "bounded_clusters",
           "fingerprint_proposal", "rank_proposals", "LinkProposal"]
