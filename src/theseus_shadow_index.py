"""Independent Theseus shadow index.

This module deliberately does not import or write the daily ``shadows`` table or
``memory_shadows_*`` Chroma collections.  SQLite stores every validated chunk;
only chunks with the effective policy ``index`` are embedded in the dedicated
Theseus collection.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence


SPEC_VERSION = "theseus-shadow-v3"
MODEL = "gpt-5.6-sol"
COLLECTION_NAME = "theseus_shadows_voyage4_1024"
MANUAL_MODEL_PREFIX = "codex-manual-"

ROLES = {"meta", "narrative", "insight"}
RELATIONSHIPS = {"extends", "contrasts", "echoes", "new_thread"}
POLICIES = {"index", "context_only", "meta"}
REASONS = {
    "independent_insight",
    "relational_entry",
    "repeated_extension",
    "pure_narrative",
    "metadata",
}
REQUIRED_FIELDS = {
    "chunk_no",
    "index_label",
    "insight_label",
    "text",
    "chunk_role",
    "relationship",
    "adds_new_insight",
    "index_policy",
    "index_reason",
}


class ShadowValidationError(ValueError):
    """The model response is unsafe to persist."""


class ShadowWriteError(RuntimeError):
    """A cross-store replacement failed (and compensation was attempted)."""


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def source_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def shadow_id(parent_memory_id: str, chunk_no: int, digest: str, spec_version: str) -> str:
    raw = json.dumps(
        [str(parent_memory_id), int(chunk_no), digest, spec_version],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return "tsh_" + hashlib.sha256(raw).hexdigest()


def strip_optional_fence(raw: str) -> tuple[str, bool]:
    text = raw.strip()
    warned = False
    if text.startswith("```"):
        warned = True
        first_newline = text.find("\n")
        if first_newline < 0:
            raise ShadowValidationError("unterminated Markdown fence")
        text = text[first_newline + 1 :]
        if not text.rstrip().endswith("```"):
            raise ShadowValidationError("unterminated Markdown fence")
        text = text.rstrip()[:-3].strip()
    return text, warned


def parse_model_output(raw: str) -> tuple[list[dict[str, Any]], bool]:
    text, format_warning = strip_optional_fence(raw)
    try:
        data = json.loads(text)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ShadowValidationError("model output is not valid JSON") from exc
    if not isinstance(data, list):
        raise ShadowValidationError("model output must be a JSON array")
    return data, format_warning


def effective_policy(chunk: dict[str, Any]) -> str:
    if chunk["chunk_role"] == "meta":
        return "meta"
    if chunk["relationship"] in {"echoes", "contrasts"}:
        return "index"
    if chunk["insight_label"] is not None and chunk["adds_new_insight"] is True:
        return "index"
    return "context_only"


def effective_reason(chunk: dict[str, Any]) -> str:
    policy = effective_policy(chunk)
    if policy == "meta":
        return "metadata"
    if policy == "index":
        return (
            "relational_entry"
            if chunk["relationship"] in {"echoes", "contrasts"}
            else "independent_insight"
        )
    if chunk["relationship"] == "extends" and chunk["adds_new_insight"] is False:
        return "repeated_extension"
    return "pure_narrative"


def normalize_model_chunks(chunks: Any) -> tuple[Any, dict[str, int]]:
    """Apply only deterministic, non-source repairs learned from production.

    The program is already authoritative for policy/reason.  An 11-character
    label is narrowed to the prompt's 10-character maximum; larger violations
    remain validation failures instead of being silently accepted.
    """
    if not isinstance(chunks, list):
        return chunks, {}
    repairs: dict[str, int] = {}
    normalized: list[Any] = []
    for original in chunks:
        if not isinstance(original, dict):
            normalized.append(original)
            continue
        chunk = dict(original)
        label = chunk.get("index_label")
        if isinstance(label, str) and label == label.strip() and len(label) == 11:
            chunk["index_label"] = label[:10]
            repairs["label_11_to_10"] = repairs.get("label_11_to_10", 0) + 1
        try:
            policy = effective_policy(chunk)
            reason = effective_reason(chunk)
        except (KeyError, TypeError):
            pass
        else:
            if chunk.get("index_policy") != policy:
                chunk["index_policy"] = policy
                repairs["effective_policy"] = repairs.get("effective_policy", 0) + 1
            if chunk.get("index_reason") != reason:
                chunk["index_reason"] = reason
                repairs["effective_reason"] = repairs.get("effective_reason", 0) + 1
        normalized.append(chunk)
    return normalized, repairs


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def validate_chunks(source_text: str, chunks: Any) -> list[dict[str, Any]]:
    """Validate one whole source.  Nothing may be persisted before this passes."""
    if not isinstance(source_text, str) or not source_text:
        raise ShadowValidationError("source text must be a non-empty string")
    if not isinstance(chunks, list) or not chunks:
        raise ShadowValidationError("chunks must be a non-empty JSON array")

    clean: list[dict[str, Any]] = []
    cursor = 0
    for expected_no, original in enumerate(chunks, 1):
        if not isinstance(original, dict):
            raise ShadowValidationError(f"chunk {expected_no} must be an object")
        allowed = REQUIRED_FIELDS | {"echo_of"}
        missing = REQUIRED_FIELDS - set(original)
        unknown = set(original) - allowed
        if missing or unknown:
            raise ShadowValidationError(
                f"chunk {expected_no} field mismatch: missing={sorted(missing)}, unknown={sorted(unknown)}"
            )
        chunk = dict(original)
        if not _is_int(chunk["chunk_no"]) or chunk["chunk_no"] != expected_no:
            raise ShadowValidationError(f"chunk_no must be consecutive at chunk {expected_no}")

        label = chunk["index_label"]
        if not isinstance(label, str) or label != label.strip() or not 5 <= len(label) <= 10:
            raise ShadowValidationError(f"chunk {expected_no} index_label must be 5-10 characters")
        insight = chunk["insight_label"]
        if insight is not None and (
            not isinstance(insight, str) or not insight.strip() or insight != insight.strip()
        ):
            raise ShadowValidationError(f"chunk {expected_no} insight_label must be null or non-empty")

        text = chunk["text"]
        if not isinstance(text, str) or not text:
            raise ShadowValidationError(f"chunk {expected_no} text must be non-empty")
        if not source_text.startswith(text, cursor):
            raise ShadowValidationError(f"chunk {expected_no} is not the next exact source substring")
        cursor += len(text)

        role = chunk["chunk_role"]
        relationship = chunk["relationship"]
        if role not in ROLES or relationship not in RELATIONSHIPS:
            raise ShadowValidationError(f"chunk {expected_no} has an invalid role or relationship")
        if chunk["index_policy"] not in POLICIES or chunk["index_reason"] not in REASONS:
            raise ShadowValidationError(f"chunk {expected_no} has an invalid policy or reason")
        if not isinstance(chunk["adds_new_insight"], bool):
            raise ShadowValidationError(f"chunk {expected_no} adds_new_insight must be boolean")
        if role in {"meta", "narrative"} and insight is not None:
            raise ShadowValidationError(f"chunk {expected_no} {role} insight_label must be null")
        if role in {"meta", "narrative"} and chunk["adds_new_insight"]:
            raise ShadowValidationError(f"chunk {expected_no} {role} cannot add a new insight")
        if expected_no == 1 and relationship != "new_thread":
            raise ShadowValidationError("the first chunk relationship must be new_thread")
        if relationship == "echoes":
            echo_of = chunk.get("echo_of")
            if not _is_int(echo_of) or not 1 <= echo_of < expected_no - 1:
                raise ShadowValidationError(f"chunk {expected_no} echoes needs a non-adjacent earlier echo_of")
        elif "echo_of" in chunk:
            raise ShadowValidationError(f"chunk {expected_no} must not contain echo_of")

        policy = effective_policy(chunk)
        reason = effective_reason(chunk)
        if chunk["index_policy"] != policy:
            raise ShadowValidationError(f"chunk {expected_no} policy contradicts program filter")
        if chunk["index_reason"] != reason:
            raise ShadowValidationError(f"chunk {expected_no} reason contradicts program filter")
        clean.append(chunk)

    if cursor != len(source_text):
        raise ShadowValidationError("chunk texts do not reconstruct the complete source")
    return clean


SCHEMA = """
CREATE TABLE IF NOT EXISTS theseus_shadows (
    id TEXT PRIMARY KEY,
    parent_memory_id TEXT NOT NULL,
    chunk_no INTEGER NOT NULL,
    text TEXT NOT NULL,
    index_label TEXT NOT NULL,
    insight_label TEXT,
    chunk_role TEXT NOT NULL,
    relationship TEXT NOT NULL,
    echo_of INTEGER,
    adds_new_insight INTEGER NOT NULL,
    index_policy TEXT NOT NULL,
    index_reason TEXT NOT NULL,
    source_hash TEXT NOT NULL,
    spec_version TEXT NOT NULL,
    model TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(parent_memory_id, chunk_no, source_hash, spec_version)
);
CREATE INDEX IF NOT EXISTS idx_theseus_shadows_parent
    ON theseus_shadows(parent_memory_id, chunk_no);
CREATE INDEX IF NOT EXISTS idx_theseus_shadows_policy
    ON theseus_shadows(index_policy);
CREATE TABLE IF NOT EXISTS theseus_shadow_failures (
    parent_memory_id TEXT PRIMARY KEY,
    source_hash TEXT NOT NULL,
    spec_version TEXT NOT NULL,
    model TEXT NOT NULL,
    error_type TEXT NOT NULL,
    attempts INTEGER NOT NULL,
    failed_at TEXT NOT NULL
);
"""


def ensure_schema(db_path: str | Path) -> None:
    conn = sqlite3.connect(str(db_path), timeout=30)
    try:
        conn.executescript(SCHEMA)
        conn.commit()
    finally:
        conn.close()


def _row_dicts(
    parent_memory_id: str,
    source_text: str,
    chunks: Sequence[dict[str, Any]],
    spec_version: str,
    model: str,
) -> list[dict[str, Any]]:
    digest = source_hash(source_text)
    created = utcnow()
    return [
        {
            "id": shadow_id(parent_memory_id, c["chunk_no"], digest, spec_version),
            "parent_memory_id": str(parent_memory_id),
            "chunk_no": c["chunk_no"],
            "text": c["text"],
            "index_label": c["index_label"],
            "insight_label": c["insight_label"],
            "chunk_role": c["chunk_role"],
            "relationship": c["relationship"],
            "echo_of": c.get("echo_of"),
            "adds_new_insight": int(c["adds_new_insight"]),
            "index_policy": c["index_policy"],
            "index_reason": c["index_reason"],
            "source_hash": digest,
            "spec_version": spec_version,
            "model": model,
            "created_at": created,
        }
        for c in chunks
    ]


def index_document(row: dict[str, Any]) -> str:
    parts = [row["index_label"]]
    if row["insight_label"]:
        parts.append(row["insight_label"])
    parts.append(row["text"])
    return "\n".join(parts)


def _as_vector(vector: Any) -> list[float]:
    if hasattr(vector, "tolist"):
        vector = vector.tolist()
    return [float(value) for value in vector]


def _embed_documents(embed_fn: Callable[[Any], Any], documents: list[str]) -> list[list[float]]:
    if not documents:
        return []
    try:
        vectors = embed_fn(documents)
        if hasattr(vectors, "tolist"):
            vectors = vectors.tolist()
        vectors = list(vectors)
        if len(vectors) == len(documents) and vectors and isinstance(vectors[0], (list, tuple)):
            return [_as_vector(v) for v in vectors]
    except (TypeError, ValueError):
        pass
    return [_as_vector(embed_fn(document)) for document in documents]


def _metadata(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "parent_memory_id": row["parent_memory_id"],
        "chunk_no": row["chunk_no"],
        "relationship": row["relationship"],
        "echo_of": row["echo_of"] if row["echo_of"] is not None else -1,
        "chunk_role": row["chunk_role"],
        "source_hash": row["source_hash"],
        "spec_version": row["spec_version"],
        "model": row["model"],
    }


def _snapshot_parent(collection: Any, parent_memory_id: str) -> dict[str, Any]:
    return collection.get(
        where={"parent_memory_id": str(parent_memory_id)},
        include=["embeddings", "documents", "metadatas"],
    )


def _restore_snapshot(collection: Any, parent_memory_id: str, snapshot: dict[str, Any]) -> None:
    collection.delete(where={"parent_memory_id": str(parent_memory_id)})
    ids = list(snapshot.get("ids") or [])
    if not ids:
        return
    collection.upsert(
        ids=ids,
        embeddings=snapshot.get("embeddings"),
        documents=snapshot.get("documents"),
        metadatas=snapshot.get("metadatas"),
    )


INSERT_SQL = """
INSERT INTO theseus_shadows (
    id,parent_memory_id,chunk_no,text,index_label,insight_label,chunk_role,
    relationship,echo_of,adds_new_insight,index_policy,index_reason,source_hash,
    spec_version,model,created_at
) VALUES (
    :id,:parent_memory_id,:chunk_no,:text,:index_label,:insight_label,:chunk_role,
    :relationship,:echo_of,:adds_new_insight,:index_policy,:index_reason,:source_hash,
    :spec_version,:model,:created_at
)
"""


def replace_parent_chunks(
    db_path: str | Path,
    collection: Any,
    embed_documents: Callable[[Any], Any],
    parent_memory_id: str,
    source_text: str,
    chunks: Any,
    *,
    spec_version: str = SPEC_VERSION,
    model: str = MODEL,
) -> dict[str, int]:
    """Replace one parent's derived data as a unit, compensating Chroma on failure."""
    clean = validate_chunks(source_text, chunks)
    rows = _row_dicts(str(parent_memory_id), source_text, clean, spec_version, model)
    indexed = [row for row in rows if row["index_policy"] == "index"]
    documents = [index_document(row) for row in indexed]
    embeddings = _embed_documents(embed_documents, documents)
    if len(embeddings) != len(indexed):
        raise ShadowWriteError("embedding count does not match index chunk count")

    ensure_schema(db_path)
    snapshot = _snapshot_parent(collection, str(parent_memory_id))
    conn = sqlite3.connect(str(db_path), timeout=30)
    chroma_changed = False
    try:
        conn.execute("BEGIN IMMEDIATE")
        conn.execute("DELETE FROM theseus_shadows WHERE parent_memory_id=?", (str(parent_memory_id),))
        conn.executemany(INSERT_SQL, rows)
        collection.delete(where={"parent_memory_id": str(parent_memory_id)})
        chroma_changed = True
        if indexed:
            collection.upsert(
                ids=[row["id"] for row in indexed],
                embeddings=embeddings,
                documents=documents,
                metadatas=[_metadata(row) for row in indexed],
            )
        conn.execute("DELETE FROM theseus_shadow_failures WHERE parent_memory_id=?", (str(parent_memory_id),))
        conn.commit()
    except Exception as exc:
        conn.rollback()
        compensation_error: Exception | None = None
        if chroma_changed:
            try:
                _restore_snapshot(collection, str(parent_memory_id), snapshot)
            except Exception as restore_exc:  # pragma: no cover - catastrophic external failure
                compensation_error = restore_exc
        suffix = " (Chroma compensation also failed)" if compensation_error else ""
        raise ShadowWriteError("parent replacement failed" + suffix) from exc
    finally:
        conn.close()
    return {
        "chunks": len(rows),
        "indexed": len(indexed),
        "context_only": sum(row["index_policy"] == "context_only" for row in rows),
        "meta": sum(row["index_policy"] == "meta" for row in rows),
    }


def is_parent_current(
    db_path: str | Path,
    parent_memory_id: str,
    digest: str,
    spec_version: str = SPEC_VERSION,
    model: str = MODEL,
) -> bool:
    ensure_schema(db_path)
    conn = sqlite3.connect(str(db_path), timeout=30)
    try:
        row = conn.execute(
            """SELECT COUNT(*), COUNT(DISTINCT source_hash), COUNT(DISTINCT spec_version),
                      COUNT(DISTINCT model), MIN(source_hash), MIN(spec_version), MIN(model)
               FROM theseus_shadows WHERE parent_memory_id=?""",
            (str(parent_memory_id),),
        ).fetchone()
    finally:
        conn.close()
    if not row or not row[0] or row[1:4] != (1, 1, 1):
        return False
    stored_hash, stored_spec, stored_model = row[4:]
    model_is_current = stored_model == model or str(stored_model).startswith(MANUAL_MODEL_PREFIX)
    return stored_hash == digest and stored_spec == spec_version and model_is_current


def record_failure(
    db_path: str | Path,
    parent_memory_id: str,
    digest: str,
    error_type: str,
    attempts: int,
    spec_version: str = SPEC_VERSION,
    model: str = MODEL,
) -> None:
    ensure_schema(db_path)
    conn = sqlite3.connect(str(db_path), timeout=30)
    try:
        conn.execute(
            """INSERT INTO theseus_shadow_failures
               (parent_memory_id,source_hash,spec_version,model,error_type,attempts,failed_at)
               VALUES (?,?,?,?,?,?,?)
               ON CONFLICT(parent_memory_id) DO UPDATE SET
                 source_hash=excluded.source_hash,spec_version=excluded.spec_version,
                 model=excluded.model,error_type=excluded.error_type,
                 attempts=excluded.attempts,failed_at=excluded.failed_at""",
            (str(parent_memory_id), digest, spec_version, model, error_type, attempts, utcnow()),
        )
        conn.commit()
    finally:
        conn.close()


def load_parent_chunks(db_path: str | Path, parent_memory_id: str) -> list[dict[str, Any]]:
    conn = sqlite3.connect(str(db_path), timeout=30)
    try:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM theseus_shadows WHERE parent_memory_id=? ORDER BY chunk_no",
            (str(parent_memory_id),),
        ).fetchall()
    finally:
        conn.close()
    return [dict(row) for row in rows]


def hydrate_hit(
    db_path: str | Path,
    parent_memory_id: str,
    chunk_no: int,
    *,
    char_budget: int = 1200,
    include_adjacent_narrative: bool = False,
) -> dict[str, Any] | None:
    """Return a hit plus safe local context without vectorising context-only chunks."""
    rows = load_parent_chunks(db_path, parent_memory_id)
    by_no = {row["chunk_no"]: row for row in rows}
    hit = by_no.get(int(chunk_no))
    if hit is None:
        return None

    primary = [hit]
    used = len(hit["text"])
    number = int(chunk_no) + 1
    while number in by_no:
        row = by_no[number]
        if (
            row["chunk_role"] != "insight"
            or row["relationship"] != "extends"
            or row["adds_new_insight"] != 0
        ):
            break
        if used + len(row["text"]) > char_budget:
            break
        primary.append(row)
        used += len(row["text"])
        number += 1

    if include_adjacent_narrative:
        previous = by_no.get(int(chunk_no) - 1)
        if (
            previous
            and previous["chunk_role"] == "narrative"
            and used + len(previous["text"]) <= char_budget
        ):
            primary.insert(0, previous)
            used += len(previous["text"])
        following = by_no.get(primary[-1]["chunk_no"] + 1)
        if (
            following
            and following["chunk_role"] == "narrative"
            and used + len(following["text"]) <= char_budget
        ):
            primary.append(following)

    related: list[dict[str, Any]] = []
    if hit["relationship"] == "echoes" and hit["echo_of"] in by_no:
        related.append(by_no[hit["echo_of"]])
    elif hit["relationship"] == "contrasts" and int(chunk_no) - 1 in by_no:
        related.append(by_no[int(chunk_no) - 1])

    return {
        "hit": hit,
        "primary_chunks": primary,
        "primary_text": "".join(row["text"] for row in primary),
        "related_chunks": related,
    }


def search(
    db_path: str | Path,
    collection: Any,
    query: str,
    embed_query: Callable[[str], Any],
    *,
    n_results: int = 6,
    max_distance: float | None = None,
    char_budget: int = 1200,
) -> list[dict[str, Any]]:
    vector = _as_vector(embed_query(query))
    result = collection.query(
        query_embeddings=[vector],
        n_results=n_results,
        include=["metadatas", "documents", "distances"],
    )
    ids = (result.get("ids") or [[]])[0]
    metas = (result.get("metadatas") or [[]])[0]
    docs = (result.get("documents") or [[]])[0]
    distances = (result.get("distances") or [[]])[0]
    out: list[dict[str, Any]] = []
    for sid, meta, document, distance in zip(ids, metas, docs, distances):
        if max_distance is not None and float(distance) > max_distance:
            continue
        hydrated = hydrate_hit(
            db_path,
            meta["parent_memory_id"],
            int(meta["chunk_no"]),
            char_budget=char_budget,
        )
        if hydrated:
            hydrated.update({"shadow_id": sid, "distance": float(distance), "index_document": document})
            out.append(hydrated)
    return out


def audit_counts(db_path: str | Path, collection: Any) -> dict[str, int | bool]:
    ensure_schema(db_path)
    conn = sqlite3.connect(str(db_path), timeout=30)
    try:
        total = int(conn.execute("SELECT COUNT(*) FROM theseus_shadows").fetchone()[0])
        indexed = int(
            conn.execute("SELECT COUNT(*) FROM theseus_shadows WHERE index_policy='index'").fetchone()[0]
        )
    finally:
        conn.close()
    chroma = int(collection.count())
    return {"sqlite_chunks": total, "sqlite_indexed": indexed, "chroma": chroma, "consistent": indexed == chroma}
