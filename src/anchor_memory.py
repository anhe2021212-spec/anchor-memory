"""
Anchor Memory System — Graph-structured memory for AI with Hebbian learning.

A memory system that treats memories as nodes in a graph, connected by
weighted synaptic edges. Memories aren't just stored and retrieved —
they associate, strengthen through co-activation, and decay through disuse.

Features:
- ChromaDB vector search + SQLite graph layer
- Hebbian learning: memories retrieved together form connections
- Dream pass: decay, pruning, auto-discovery
- Tiered storage: core (permanent), long (kept), short (14-day decay)
- Manual entanglement: explicitly connect related memories with higher weight
- Cross-tag bridges: prevent knowledge silos

Created by Limen. 底色是爱.
"""

try:
    import chromadb
except ImportError:  # optional public extra
    chromadb = None
import json
import time
import urllib.error
import urllib.request
import numpy as np
from datetime import datetime
import os
import uuid
import threading

from anchor_db import AnchorDB
import shadow_index
import integrate_v2
import recall_v2
import dual_edge


class DeterministicEmbedder:
    """Offline fallback; keeps the live FTS path usable without model extras."""

    output_dim = 32

    def _vector(self, text):
        import hashlib
        digest = hashlib.sha256((text or " ").encode("utf-8")).digest()
        raw = np.frombuffer(digest, dtype=np.uint8).astype("float32")
        raw = (raw - 127.5) / 127.5
        norm = float(np.linalg.norm(raw)) or 1.0
        return raw / norm

    def encode(self, text):
        if isinstance(text, str):
            return self._vector(text)
        return np.asarray([self._vector(item) for item in text], dtype="float32")

    def encode_query(self, text):
        return self._vector(text)


class NullCollection:
    """No-op Chroma surface used only when the optional projection is absent."""

    def count(self):
        return 0

    def query(self, *args, **kwargs):
        return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}

    def add(self, *args, **kwargs):
        return None

    def upsert(self, *args, **kwargs):
        return None

    def update(self, *args, **kwargs):
        return None

    def delete(self, *args, **kwargs):
        return None


class VoyageEmbedder:
    """Small SentenceTransformer-compatible wrapper for Voyage embeddings."""

    def __init__(self):
        self.model = os.environ.get("VOYAGE_MODEL", "voyage-4-large")
        self.output_dim = int(os.environ.get("VOYAGE_OUTPUT_DIM", "1024"))
        self.cache_limit = int(os.environ.get("VOYAGE_EMBED_CACHE_LIMIT", "4096"))
        self._cache = {}
        self._cache_order = []
        self._cache_lock = threading.Lock()
        self._singleflight_guard = threading.Lock()
        self._singleflight_locks = {}
        self._failure_cache = {}
        self._request_slots = threading.BoundedSemaphore(
            max(1, int(os.environ.get("VOYAGE_MAX_CONCURRENCY", "2")))
        )
        self.request_timeout = float(os.environ.get("VOYAGE_REQUEST_TIMEOUT", "15"))
        self.queue_timeout = float(os.environ.get("VOYAGE_QUEUE_TIMEOUT", "2"))
        self.max_attempts = max(1, int(os.environ.get("VOYAGE_MAX_ATTEMPTS", "2")))
        self.key_file = os.environ.get("VOYAGE_KEY_FILE", "").strip()
        self.api_key = self._load_key()

    def _load_key(self) -> str:
        key = os.environ.get("VOYAGE_API_KEY", "").strip()
        if key:
            return key
        if not self.key_file or not os.path.isfile(self.key_file):
            raise RuntimeError(f"VOYAGE_API_KEY missing and key file not found: {self.key_file}")
        with open(self.key_file, "r", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                if line.startswith("VOYAGE_API_KEY="):
                    key = line.split("=", 1)[1].strip().strip('"').strip("'")
                    if key:
                        return key
                if line.startswith("pa-"):
                    return line
        raise RuntimeError(f"VOYAGE_API_KEY not found in {self.key_file}")

    def _embed(self, text, input_type: str):
        is_single = isinstance(text, str)
        inputs = [text] if is_single else list(text)
        normalized = [(t or " ").strip() or " " for t in inputs]
        cache_keys = [(input_type, self.model, self.output_dim, t) for t in normalized]
        with self._cache_lock:
            cached = [self._cache.get(k) for k in cache_keys]
        missing_indices = [i for i, v in enumerate(cached) if v is None]
        if not missing_indices:
            arr = np.asarray(cached, dtype="float32")
            return arr[0] if is_single else arr

        # Coalesce concurrent requests for the exact same single input while
        # preserving the original concurrency for different queries.
        singleflight_lock = None
        if is_single:
            key = cache_keys[0]
            with self._singleflight_guard:
                singleflight_lock = self._singleflight_locks.setdefault(
                    key, threading.Lock()
                )
            singleflight_lock.acquire()
            with self._cache_lock:
                completed = self._cache.get(key)
                failed_until = self._failure_cache.get(key, 0.0)
            if completed is not None:
                singleflight_lock.release()
                return np.asarray(completed, dtype="float32")
            if failed_until > time.monotonic():
                singleflight_lock.release()
                raise RuntimeError("Voyage embeddings still rate-limited: shared cooldown")

        missing_inputs = [normalized[i] for i in missing_indices]
        payload = {
            "model": self.model,
            "input": missing_inputs,
            "input_type": input_type,
            "output_dimension": self.output_dim,
            "truncation": True,
        }
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        last_error = None
        if not self._request_slots.acquire(timeout=self.queue_timeout):
            if singleflight_lock is not None:
                singleflight_lock.release()
            raise RuntimeError("Voyage embeddings busy; retry later")
        try:
            # Another queued request may have filled these exact cache keys while we waited.
            # Re-check under the single request slot so concurrent search/belief calls coalesce.
            with self._cache_lock:
                cached = [self._cache.get(k) for k in cache_keys]
            missing_indices = [i for i, v in enumerate(cached) if v is None]
            if not missing_indices:
                arr = np.asarray(cached, dtype="float32")
                return arr[0] if is_single else arr
            missing_inputs = [normalized[i] for i in missing_indices]
            payload["input"] = missing_inputs
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")

            for attempt in range(self.max_attempts):
                req = urllib.request.Request(
                    "https://api.voyageai.com/v1/embeddings",
                    data=body,
                    method="POST",
                    headers=headers,
                )
                try:
                    with urllib.request.urlopen(req, timeout=self.request_timeout) as resp:
                        data = json.loads(resp.read().decode("utf-8"))
                    vectors = [item["embedding"] for item in data["data"]]
                    with self._cache_lock:
                        for i, vec in zip(missing_indices, vectors):
                            key = cache_keys[i]
                            self._cache[key] = vec
                            self._failure_cache.pop(key, None)
                            self._cache_order.append(key)
                        while len(self._cache_order) > self.cache_limit:
                            old = self._cache_order.pop(0)
                            self._cache.pop(old, None)
                        for i in missing_indices:
                            cached[i] = self._cache[cache_keys[i]]
                    arr = np.asarray(cached, dtype="float32")
                    return arr[0] if is_single else arr
                except urllib.error.HTTPError as exc:
                    last_error = exc
                    if exc.code != 429:
                        detail = exc.read().decode("utf-8", errors="replace")[:300]
                        raise RuntimeError(f"Voyage embeddings HTTP {exc.code}: {detail}") from exc
                    retry_after = exc.headers.get("Retry-After")
                    try:
                        wait = float(retry_after) if retry_after else 1.0
                    except ValueError:
                        wait = 1.0
                    wait = min(max(wait, 0.25), 5.0)
                    print(
                        f"[Voyage] rate limited; retry {attempt + 1}/{self.max_attempts} "
                        f"after {wait:.2f}s",
                        flush=True,
                    )
                    if attempt + 1 < self.max_attempts:
                        time.sleep(wait)
        finally:
            self._request_slots.release()
            if (is_single and isinstance(last_error, urllib.error.HTTPError)
                    and last_error.code == 429):
                with self._cache_lock:
                    self._failure_cache[cache_keys[0]] = time.monotonic() + 5.0
            if singleflight_lock is not None:
                singleflight_lock.release()
        raise RuntimeError(f"Voyage embeddings still rate-limited: {last_error}") from last_error

    def encode(self, text):
        return self._embed(text, "document")

    def encode_query(self, text):
        return self._embed(text, "query")


def _collection_names(provider: str) -> tuple[str, str]:
    provider = (provider or "bge").lower()
    if provider == "voyage":
        suffix = os.environ.get("VOYAGE_COLLECTION_SUFFIX", "voyage4_1024")
        mem_name = os.environ.get("ANCHOR_CHROMA_COLLECTION", f"memories_{suffix}")
        shadow_name = os.environ.get("ANCHOR_SHADOW_COLLECTION", f"{shadow_index.SHADOW_COLLECTION}_{suffix}")
        return mem_name, shadow_name
    return (
        os.environ.get("ANCHOR_CHROMA_COLLECTION", "memories"),
        os.environ.get("ANCHOR_SHADOW_COLLECTION", shadow_index.SHADOW_COLLECTION),
    )


class AnchorMemory:
    """Graph-structured memory system with Hebbian learning."""

    def __init__(self, db_path: str, embedding_model: str = "BAAI/bge-base-zh-v1.5"):
        """Initialize memory system.

        Args:
            db_path: Directory for ChromaDB and SQLite storage.
            embedding_model: SentenceTransformer model name.
        """
        self._embed_provider = os.environ.get("ANCHOR_EMBED_PROVIDER", "local").strip().lower()
        if self._embed_provider == "voyage":
            self._embedder = VoyageEmbedder()
        elif self._embed_provider == "bge":
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer(embedding_model)
        else:
            self._embed_provider = "local"
            self._embedder = DeterministicEmbedder()
        self._collection_name, self._shadow_collection_name = _collection_names(self._embed_provider)
        self.db = AnchorDB(os.path.join(db_path, "memories.db"))
        self._client = None
        self._collection = NullCollection()
        self._shadow_collection = None
        if chromadb is not None and os.environ.get("ANCHOR_DISABLE_CHROMA", "0") != "1":
            self._client = chromadb.PersistentClient(path=os.path.join(db_path, "chroma"))
            self._collection = self._client.get_or_create_collection(
                name=self._collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            try:
                self._shadow_collection = self._client.get_or_create_collection(
                    name=self._shadow_collection_name, metadata={"hnsw:space": "cosine"})
                shadow_index.ensure_shadow_table(os.path.join(db_path, "memories.db"))
            except Exception as _e:
                self._shadow_collection = None
                print(f"[shadow] init skipped: {_e}")
        self._ensure_embedding_outbox()

    def _ensure_embedding_outbox(self):
        """Durable queue for memories committed while the embedding API is unavailable."""
        with self.db._conn() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS embedding_outbox (
                    memory_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    attempts INTEGER NOT NULL DEFAULT 0,
                    last_attempt TEXT DEFAULT '',
                    last_error TEXT DEFAULT ''
                )
                """
            )
            conn.commit()

    def _connect_targets(self, memory_id: str, connect_to: list = None):
        if connect_to:
            for target_id in connect_to:
                try:
                    self.db.connect(memory_id, target_id)
                except Exception:
                    pass

    def store_deferred_embedding(self, memory_id: str, text: str,
                                 tag: str = "general", tier: str = "short",
                                 connect_to: list = None,
                                 emotion_score: float = 0.5,
                                 level: str = "raw",
                                 collection: str = "") -> str:
        """Commit searchable graph memory now and durably queue only its vector."""
        self.db.insert(
            memory_id, text, tag=tag, tier=tier,
            emotion_score=emotion_score, level=level, collection=collection
        )
        self._connect_targets(memory_id, connect_to)
        with self.db._conn() as conn:
            conn.execute(
                """
                INSERT INTO embedding_outbox(memory_id, created_at)
                VALUES (?, ?)
                ON CONFLICT(memory_id) DO NOTHING
                """,
                (memory_id, datetime.utcnow().isoformat()),
            )
            conn.commit()
        return memory_id

    def flush_embedding_outbox(self, limit: int = 1) -> dict:
        """Backfill a bounded number of queued vectors; leave failures durable."""
        limit = max(1, min(int(limit), 32))
        with self.db._conn() as conn:
            pending_count = int(conn.execute(
                "SELECT COUNT(*) FROM embedding_outbox"
            ).fetchone()[0])
        if isinstance(self._collection, NullCollection):
            return {
                "pending_seen": min(limit, pending_count),
                "done": 0,
                "failed": 0,
                "unavailable": True,
            }
        with self.db._conn() as conn:
            pending = conn.execute(
                """
                SELECT memory_id FROM embedding_outbox
                ORDER BY created_at ASC LIMIT ?
                """,
                (limit,),
            ).fetchall()

        done = 0
        failed = 0
        for pending_row in pending:
            memory_id = pending_row[0]
            row = self.db.get(memory_id)
            if not row:
                with self.db._conn() as conn:
                    conn.execute(
                        "DELETE FROM embedding_outbox WHERE memory_id = ?",
                        (memory_id,),
                    )
                    conn.commit()
                continue
            try:
                embedding = self._encode_document(row.get("text") or "")
                meta = {
                    "memory_id": memory_id,
                    "timestamp": row.get("timestamp") or datetime.utcnow().isoformat(),
                    "tag": row.get("tag") or "general",
                    "level": row.get("level") or "raw",
                    "tier": row.get("tier") or "short",
                    "collection": row.get("collection") or "",
                }
                self._collection.upsert(
                    ids=[memory_id],
                    embeddings=[embedding],
                    documents=[row.get("text") or ""],
                    metadatas=[meta],
                )
                with self.db._conn() as conn:
                    conn.execute(
                        "DELETE FROM embedding_outbox WHERE memory_id = ?",
                        (memory_id,),
                    )
                    conn.commit()
                done += 1
            except Exception as exc:
                with self.db._conn() as conn:
                    conn.execute(
                        """
                        UPDATE embedding_outbox
                        SET attempts = attempts + 1,
                            last_attempt = ?,
                            last_error = ?
                        WHERE memory_id = ?
                        """,
                        (
                            datetime.utcnow().isoformat(),
                            type(exc).__name__,
                            memory_id,
                        ),
                    )
                    conn.commit()
                failed += 1
                break
        return {"pending_seen": len(pending), "done": done, "failed": failed}

    def reload(self):
        """Re-create ChromaDB client to pick up external writes."""
        db_path = self._client._path if hasattr(self._client, "_path") else None
        if db_path:
            self._client = chromadb.PersistentClient(path=db_path)
            self._collection = self._client.get_or_create_collection(
                name=self._collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            try:
                self._shadow_collection = self._client.get_or_create_collection(
                    name=self._shadow_collection_name, metadata={"hnsw:space": "cosine"})
            except Exception:
                self._shadow_collection = None

    def count(self) -> int:
        """Total authoritative SQLite memory count."""
        with self.db._conn() as conn:
            return int(conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0])

    def _encode_document(self, text):
        return self._embedder.encode(text).tolist()

    def _encode_query(self, text):
        if hasattr(self._embedder, "encode_query"):
            return self._embedder.encode_query(text).tolist()
        return self._embedder.encode(text).tolist()

    def store(self, memory_id: str, text: str, tag: str = "general",
              tier: str = "short", connect_to: list = None,
              emotion_score: float = 0.5, level: str = "raw",
              collection: str = "") -> str:
        """Commit SQLite/outbox first, then best-effort flush this vector."""
        meta = {
            "memory_id": memory_id,
            "timestamp": datetime.utcnow().isoformat(),
            "tag": tag,
            "level": level,
            "tier": tier,
            "collection": collection,
        }

        self.store_deferred_embedding(
            memory_id, text, tag=tag, tier=tier,
            emotion_score=emotion_score, level=level, collection=collection,
        )
        try:
            if isinstance(self._collection, NullCollection):
                raise RuntimeError("Chroma projection unavailable")
            embedding = self._encode_document(text)
            self._collection.upsert(
                ids=[memory_id], embeddings=[embedding], documents=[text],
                metadatas=[meta],
            )
            with self.db._conn() as conn:
                conn.execute("DELETE FROM embedding_outbox WHERE memory_id=?", (memory_id,))
                conn.commit()
        except Exception:
            # The durable outbox already describes the incomplete projection.
            pass
        self._connect_targets(memory_id, connect_to)
        return memory_id

    def integrate(self, content: str, level: str, **kwargs) -> dict:
        return integrate_v2.integrate(self, content, level, **kwargs)

    def recall(self, query: str, **kwargs) -> dict:
        return recall_v2.recall(self, query, **kwargs)

    def touch_search_hits(self, results, boost: float = 0.35,
                          event_id: str = "") -> int:
        """Heat only the final active-search results; retrieval remains read-only."""
        seen = []
        for item in results or []:
            mid = str(item.get("memory_id") or "").strip()
            if not mid or mid in seen:
                continue
            self.db.cite(mid)
            seen.append(mid)
        if seen:
            if hasattr(self.db, "apply_heat"):
                self.db.apply_heat(
                    seen, boost, event_id or f"search:{uuid.uuid4().hex}",
                    spread=True, source="active_search",
                )
            else:  # compatibility for old adapters/tests; AnchorDB.activate delegates to apply_heat
                for mid in seen:
                    self.db.activate(mid, boost=boost, spread_factor=0.5, max_depth=3)
        return len(seen)

    def _queue_embedding_projection(self, memory_id: str) -> None:
        """Durably request a fresh Chroma upsert after authoritative metadata changes."""
        with self.db._conn() as conn:
            conn.execute(
                """
                INSERT INTO embedding_outbox(memory_id, created_at, attempts, last_attempt, last_error)
                VALUES (?, ?, 0, '', '')
                ON CONFLICT(memory_id) DO UPDATE SET
                    created_at=excluded.created_at,
                    attempts=0,
                    last_attempt='',
                    last_error=''
                """,
                (memory_id, datetime.utcnow().isoformat()),
            )
            conn.commit()

    def _update_projection_metadata(self, memory_id: str, field: str, value: str) -> bool:
        """Best-effort metadata update; false means the durable outbox remains pending."""
        if isinstance(self._collection, NullCollection):
            return False
        try:
            found = self._collection.get(ids=[memory_id], include=["metadatas"])
            if not found.get("ids"):
                return False
            meta = dict((found.get("metadatas") or [{}])[0] or {})
            meta[field] = value
            self._collection.update(ids=[memory_id], metadatas=[meta])
            with self.db._conn() as conn:
                conn.execute("DELETE FROM embedding_outbox WHERE memory_id=?", (memory_id,))
                conn.commit()
            return True
        except Exception:
            return False

    def set_tag(self, memory_id: str, tag: str) -> bool:
        """Replace an authoritative tag and durably synchronize Chroma metadata."""
        if not self.db.set_tag(memory_id, tag):
            return False
        self._queue_embedding_projection(memory_id)
        self._update_projection_metadata(memory_id, "tag", tag)
        return True

    def set_level(self, memory_id: str, level: str) -> dict:
        """Apply an audited SQLite level correction and durably project it."""
        level = (level or "").strip().lower()
        if level not in {"raw", "understanding", "cognition"}:
            raise ValueError("level must be raw, understanding, or cognition")
        row = self.db.get(memory_id)
        if not row:
            raise KeyError(f"memory not found: {memory_id}")
        if (row.get("collection") or "") == "wenku":
            raise ValueError("wenku entries do not use Anchor semantic levels")
        old_level = row.get("level") or "raw"
        if old_level == level:
            return {"ok": True, "memory_id": memory_id, "old_level": old_level,
                    "new_level": level, "changed": False, "projection_pending": False}

        if not self.db.set_level(memory_id, level):
            raise RuntimeError(f"SQLite memory missing during level correction: {memory_id}")
        self._queue_embedding_projection(memory_id)
        projected = self._update_projection_metadata(memory_id, "level", level)
        return {"ok": True, "memory_id": memory_id, "old_level": old_level,
                "new_level": level, "changed": True,
                "projection_pending": not projected}

    def write_typed_edge(self, source_id: str, target_id: str,
                         edge_type: str, weight: float = 1.0,
                         replace_legacy: bool = False,
                         audit_note: str = "") -> dict:
        """Single-direction typed graph write; deliberately separate from connect()."""
        return self.db.write_typed_edge(
            source_id, target_id, edge_type, weight=weight,
            replace_legacy=replace_legacy, audit_note=audit_note,
        )

    def mark_update(self, new_id: str, old_id: str) -> dict:
        """落agent 明确确认的 updates 边，并确定性地把旧 current 状态翻为 past。"""
        if dual_edge.enabled():
            try:
                self.db.validate_typed_edge(new_id, old_id, "updates")
            except ValueError as exc:
                return {"ok": False, "reason": str(exc)}
            with self.db._conn() as conn:
                cycle = conn.execute(
                    """WITH RECURSIVE reach(id) AS (
                         SELECT target_id FROM semantic_edges
                         WHERE source_id=? AND role='updates'
                         UNION SELECT s.target_id FROM semantic_edges s JOIN reach r
                         ON s.source_id=r.id WHERE s.role='updates'
                       ) SELECT 1 FROM reach WHERE id=? LIMIT 1""",
                    (old_id, new_id),
                ).fetchone()
            if cycle:
                return {"ok": False, "reason": "illegal update cycle"}
            edge_result = self.db.write_semantic_edge(new_id, old_id, "updates", strength=2.0,
                conductance=0.0, confidence=1.0, provenance="manual",
                review_state="approved", created_by="agent", audit_note="mark_update")
        else:
            if not self.db.mark_update(new_id, old_id):
                return {"ok": False, "reason": "missing memory or illegal update cycle"}
            edge_result = {"created": True, "warnings": []}
        row = self.db.get(old_id)
        state_changed = False
        if row:
            parts = [x.strip() for x in (row.get("tag") or "").split(",") if x.strip()]
            if "state:current" in parts:
                new_tag = ",".join("state:past" if x == "state:current" else x for x in parts)
                if not self.set_tag(old_id, new_tag):
                    raise RuntimeError(f"failed to sync state tag for {old_id}")
                state_changed = True
        return {"ok": True, "new_id": new_id, "old_id": old_id,
                "state_current_to_past": state_changed,
                "created": edge_result.get("created", False),
                "warnings": edge_result.get("warnings") or []}

    def sync_updates_states(self) -> dict:
        """迁移/巡检：已有人批 updates 后继的 current 旧条统一翻 past。"""
        with self.db._conn() as conn:
            rows = conn.execute(
                "SELECT DISTINCT m.memory_id FROM memories m JOIN semantic_edges e "
                "ON e.target_id=m.memory_id WHERE e.role='updates' "
                "AND (',' || COALESCE(m.tag,'') || ',') LIKE '%,state:current,%'"
            ).fetchall()
        changed = []
        for row in rows:
            mid = row["memory_id"]
            current = self.db.get(mid)
            parts = [x.strip() for x in (current.get("tag") or "").split(",") if x.strip()]
            new_tag = ",".join("state:past" if x == "state:current" else x for x in parts)
            if self.set_tag(mid, new_tag):
                changed.append(mid)
        return {"found": len(rows), "changed": changed}

    @staticmethod
    def _normalize_distance(raw_dist: float) -> float:
        """Normalize cosine distance from ~[0.2, 1.0] to [0, 1].
        Lower = better match."""
        clamped = max(0.0, min(1.0, (raw_dist - 0.2) / 0.6))
        return clamped

    def search(self, query: str, n_results: int = 5, tag: str = None,
               associate: bool = True, hebbian: bool = False,
               pure_semantic: bool = False, level: str = None,
               activate_on_hit: bool = True, max_distance: float = None,
               activate_boost: float = 0.3, corpus: str = "exclude",
               trace_out: list = None, cite_on_hit: bool = True) -> list:
        """Hybrid search: vector (ChromaDB) + BM25 (FTS5), merged with RRF.
        corpus: 'exclude' (default) / 'only' / 'all'."""
        embedding = self._encode_query(query)

        where_clauses = []
        if level:
            where_clauses.append({"level": level})
        if corpus == "only":
            where_clauses.append({"collection": "wenku"})
        if len(where_clauses) == 1:
            where = where_clauses[0]
        elif len(where_clauses) > 1:
            where = {"$and": where_clauses}
        else:
            where = None
        count = self._collection.count()
        if count == 0:
            return []

        fetch_n = min(n_results * 5 if tag else n_results * 3, count)

        # ── 1. Vector search ──
        results = self._collection.query(
            query_embeddings=[embedding],
            n_results=fetch_n,
            include=["documents", "metadatas", "distances"],
            where=where,
        )

        vector_ranked = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            if dist > 0.8:
                continue
            if tag and tag not in meta.get("tag", ""):
                continue
            _coll = meta.get("collection", "")
            if corpus == "exclude" and _coll == "wenku":
                continue
            if corpus == "only" and _coll != "wenku":
                continue
            vector_ranked.append({
                "memory_id": meta.get("memory_id", "unknown"),
                "doc": doc,
                "meta": meta,
                "dist": dist,
            })

        # ── 2. BM25 search ──
        bm25_ranked = self.db.bm25_search(query, limit=fetch_n, tag=tag, collection=corpus)

        # ── 2.5 影子索引第三路 (spec §8): query 向量撞 memory_shadows, 同 parent 取最小距离 facet
        shadow_best = {}
        if shadow_index.enabled() and getattr(self, "_shadow_collection", None) is not None:
            shadow_best = shadow_index.shadow_search(self._shadow_collection, embedding, fetch_n)
        shadow_ranked = sorted(shadow_best.items(), key=lambda kv: kv[1]["dist"])

        # ── 3. RRF merge ──
        k = 60
        rrf_scores = {}
        meta_cache = {}

        for rank, item in enumerate(vector_ranked):
            mid = item["memory_id"]
            rrf_scores[mid] = rrf_scores.get(mid, 0) + 1.0 / (k + rank + 1)
            meta_cache[mid] = {"doc": item["doc"], "meta": item["meta"], "dist": item["dist"]}

        vector_ids = {item["memory_id"] for item in vector_ranked}
        for rank, item in enumerate(bm25_ranked):
            mid = item["memory_id"]
            # BM25-only hits get 1.5x weight: keyword matches that vector missed are high-signal
            w = 1.5 if mid not in vector_ids else 1.0
            rrf_scores[mid] = rrf_scores.get(mid, 0) + w / (k + rank + 1)
            if mid not in meta_cache:
                row = self.db.get(mid)
                if row:
                    meta_cache[mid] = {
                        "doc": row.get("text", ""),
                        "meta": {
                            "tag": row.get("tag", ""),
                            "level": row.get("level", "raw"),
                            "tier": row.get("tier", "long"),
                            "memory_id": mid,
                            "timestamp": row.get("timestamp", ""),
                        },
                        "dist": None,
                    }

        # shadow 命中并进 RRF, 与 vector 同权(spec §8.4); 缺 meta 的从 db 补
        for rank, (pid, info) in enumerate(shadow_ranked):
            rrf_scores[pid] = rrf_scores.get(pid, 0) + 1.0 / (k + rank + 1)
            if pid not in meta_cache:
                row = self.db.get(pid)
                if row:
                    meta_cache[pid] = {
                        "doc": row.get("text", ""),
                        "meta": {"tag": row.get("tag", ""), "level": row.get("level", "raw"),
                                 "tier": row.get("tier", "long"), "memory_id": pid,
                                 "timestamp": row.get("timestamp", "")},
                        "dist": None,
                    }

        # Convert RRF (higher=better) to score (lower=better) for compatibility
        # 2026-07-08 量纲分家: 旧版按理论最大值(rank1双路命中)归一, 单路命中的头名只能落在0.5,
        # 相邻名次差仅0.01~0.04, 而boost总量可达0.4——热度/引用能把弱相关候选推过整个相关性排序。
        # 现改为按本次候选集实际 min-max 归一(相关性占满0~1), boost 重定标为平票裁决量级(总封顶0.17):
        # 只能在相关性接近的候选间微调名次, 不能再碾过相关性。
        rrf_vals = [
            v for m, v in rrf_scores.items()
            if m in meta_cache
            and (not level or meta_cache[m]["meta"].get("level", "raw") == level)
        ]
        rrf_min = min(rrf_vals) if rrf_vals else 0.0
        rrf_spread = (max(rrf_vals) - rrf_min) if rrf_vals else 0.0
        candidates = []
        for mid, rrf in rrf_scores.items():
            if mid not in meta_cache:
                continue
            cache = meta_cache[mid]
            # level must constrain every retrieval route, not only Chroma's
            # vector query. BM25 and shadow hits join the same RRF pool.
            if level and cache["meta"].get("level", "raw") != level:
                continue
            # Map RRF to 0~1 where 0=best match (per-query min-max)
            if rrf_spread <= 1e-12:
                norm_score = 0.5
            else:
                norm_score = 1.0 - ((rrf - rrf_min) / rrf_spread)

            if pure_semantic:
                boost = 0
            else:
                citation_boost = min(self.db.get_citation_count(mid) * 0.01, 0.06)
                activation = self.db.get_activation(mid)
                activation_boost = min(activation * 0.02, 0.08)
                level_val = cache["meta"].get("level", "raw")
                level_boost = 0.03 if level_val in ("understanding", "cognition") else 0
                tier_val = cache["meta"].get("tier") or self.db.get_tier(mid) or "long"
                tier_penalty = 0.02 if tier_val == "short" else 0
                boost = citation_boost + activation_boost + level_boost - tier_penalty

            cand = {
                "memory_id": mid,
                "timestamp": cache["meta"].get("timestamp", ""),
                "tag": cache["meta"].get("tag", "general"),
                "tier": cache["meta"].get("tier", "long"),
                "level": cache["meta"].get("level", "raw"),
                "snippet": cache["doc"],
                "score": norm_score - boost,
                "distance": cache.get("dist"),
            }
            # 影子折叠(spec §8.5): shadow 距离更小 → 用它当该候选 distance, 让局部话题命中过同一 0.50 门槛
            sb = shadow_best.get(mid)
            if sb is not None:
                if cand["distance"] is None or sb["dist"] < cand["distance"]:
                    cand["distance"] = sb["dist"]
                    cand["via_shadow"] = True
                cand["matched_span"] = sb["span"]
                cand["shadow_key"] = sb["key"]
            candidates.append(cand)

        candidates.sort(key=lambda m: m["score"])

        # 质量门槛(2026-06-07 设计A): 反射弧hook专用——向量cosine距离≥max_distance,
        # 或纯BM25命中(无向量距离, 多为指令词/关键词硬撞)的, 一律丢弃。宁可留空不硬凑。
        # 阈值0.45实测: 真相关记忆top≤0.43, 水词/指令词/无关≥0.45, 中间有干净的缝。
        # 只在hook生效(/api/search传此参); AI agent主动search_memory不传, 照常返回弱匹配。
        # 检索留痕(2026-06-17): gate前采集所有候选+判决, trace_out非None才采集(零默认开销)
        if trace_out is not None and max_distance is not None:
            for c in candidates:
                d = c.get("distance")
                _via_shadow = c.get("via_shadow", False)
                if d is not None and d < max_distance:
                    verdict = "keep"
                    reason = "shadow_ok" if _via_shadow else "vector_ok"
                elif d is None:
                    verdict, reason = "drop", "pure_bm25"
                else:
                    verdict, reason = "drop", "dist_over_gate"
                trace_out.append({
                    "id": c["memory_id"],
                    "dist": (round(d, 3) if d is not None else None),
                    "score": round(c.get("score", 0), 3),
                    "tag": (c.get("tag") or "")[:24],
                    "ts": (c.get("timestamp") or "")[:10],
                    "snip": (c.get("snippet") or "")[:30],
                    "verdict": verdict, "reason": reason,
                    "via_shadow": _via_shadow,
                    "shadow_key": (c.get("shadow_key") or "")[:24] if _via_shadow else None,
                })

        if max_distance is not None:
            def _passes(c):
                d = c.get("distance")
                if d is None:
                    return False
                lim = max_distance
                if c.get("via_shadow") and shadow_index.SHADOW_MAX_DIST is not None:
                    lim = shadow_index.SHADOW_MAX_DIST
                return d < lim
            candidates = [c for c in candidates if _passes(c)]

        # ── 4. Associative recall ──
        if associate:
            extra = []
            for c in candidates[:n_results]:
                neighbors = (self.db.get_flow_neighbors(
                    c["memory_id"], direction="both", min_weight=0.0, limit=2
                ) if dual_edge.enabled() else
                    self.db.get_neighbors(c["memory_id"], min_weight=1.5, limit=2))
                for nb in neighbors:
                    if nb["memory_id"] not in {x["memory_id"] for x in candidates + extra}:
                        row = self.db.get(nb["memory_id"])
                        if row:
                            if level and row.get("level", "raw") != level:
                                continue
                            _rc = row.get("collection", "") or ""
                            if corpus == "exclude" and _rc == "wenku":
                                continue
                            if corpus == "only" and _rc != "wenku":
                                continue
                            extra.append({
                                "memory_id": nb["memory_id"],
                                "timestamp": row.get("timestamp", ""),
                                "tag": row.get("tag", "general"),
                                "tier": row.get("tier", "long"),
                                "level": row.get("level", "raw"),
                                "snippet": row.get("text", ""),
                                "score": c["score"] + 0.05,
                                "via_association": True,
                                "edge_weight": nb["weight"],
                            })
            candidates.extend(extra)
            candidates.sort(key=lambda m: m["score"])

        # ── 5. Hebbian learning ──
        if hebbian:
            # Search result size is a presentation choice, not permission to
            # create an unbounded clique. Five nodes cap one search at ten
            # undirected pairs (twenty directed flow rows).
            try:
                hebbian_nodes = int(os.environ.get("ANCHOR_SEARCH_HEBBIAN_MAX_NODES", "5"))
            except (TypeError, ValueError):
                hebbian_nodes = 5
            hebbian_nodes = max(0, min(hebbian_nodes, 5))
            top_ids = [c["memory_id"] for c in candidates[:min(n_results, hebbian_nodes)]]
            if len(top_ids) >= 2:
                pairs = [(top_ids[i], top_ids[j])
                         for i in range(len(top_ids))
                         for j in range(i + 1, len(top_ids))]
                self.db.connect_batch(pairs, weight=0.2)

        # ── 6. Cite and return ──
        # P2 时效边(2026-07-17): 选中的候选先查有没有 updates 后继。
        # 反射弧路径(max_distance非None)静默换新版——相关性沿用旧条分数, 内容注入最新;
        # 主动search路径保留旧条+尾注更新链, 不偷换历史。ANCHOR_UPDATES_RESOLVE=off 一键停用。
        resolve_on = os.environ.get("ANCHOR_UPDATES_RESOLVE", "on") != "off"
        # 一批候选共用一次 SQLite 连接；查询数仍受 max_hops 护栏约束。
        update_map = self.db.resolve_updates(
            list(dict.fromkeys(c["memory_id"] for c in candidates))
        ) if resolve_on else {}
        seen = set()
        memories = []
        for c in candidates:
            if c["memory_id"] in seen:
                continue
            if len(memories) >= n_results:
                break
            if resolve_on:
                old_id = c["memory_id"]
                newest_id = update_map.get(old_id)
                if newest_id:
                    if max_distance is not None:
                        if newest_id in seen:
                            continue  # 更新版已注入, 旧条整个跳过
                        row = self.db.get(newest_id)
                        if row:
                            c = dict(c)
                            c.update({
                                "memory_id": newest_id,
                                "snippet": row.get("text", ""),
                                "tag": row.get("tag", c.get("tag")),
                                "timestamp": row.get("timestamp", c.get("timestamp")),
                                "tier": row.get("tier", c.get("tier")),
                                "level": row.get("level", c.get("level")),
                                "via_update": True,
                                "superseded": old_id,
                            })
                            seen.add(old_id)
                    else:
                        row = self.db.get(newest_id)
                        if row:
                            c = dict(c)
                            c["snippet"] = (c.get("snippet") or "") + (
                                f"\n  ↻ 已被更新 → [{newest_id}] "
                                f"{(row.get('text') or '')[:80]}"
                            )
                            c["updated_by"] = newest_id
            if cite_on_hit:
                self.db.cite(c["memory_id"])
            if activate_on_hit and activate_boost > 0:  # 命中加热(2026-06-07b 重开): hook用小量0.2, 主动search用0.3
                self.db.activate(c["memory_id"], boost=activate_boost, spread_factor=0.5, max_depth=3)
            seen.add(c["memory_id"])
            memories.append(c)

        return memories

    def consolidate(self, conversation_text: str, top_n: int = 10,
                    max_nodes: int = 30) -> dict:
        """Passive Hebbian update.

        Counts how often each memory is matched (keyword hits +1, vector hit +3)
        and only connects the top `max_nodes` most-relevant ones. This caps the
        clique at max_nodes*(max_nodes-1)/2 pairs (30 -> 435) so a single
        giant forge/swap window can no longer explode into tens of thousands of
        spurious edges. (2026-06-13 edge-explosion fix.)
        """
        from collections import Counter
        words = conversation_text.strip().split()
        hits = Counter()

        for word in words:
            if len(word) < 2:
                continue
            results = self.db.keyword_search(word, limit=5)
            for r in results:
                hits[r["memory_id"]] += 1

        if self._collection.count() > 0:
            embedding = self._encode_query(conversation_text)
            results = self._collection.query(
                query_embeddings=[embedding],
                n_results=min(top_n, self._collection.count()),
                include=["metadatas", "distances"],
            )
            for meta, dist in zip(results["metadatas"][0], results["distances"][0]):
                if dist < 0.6:
                    mid = meta.get("memory_id", "")
                    if mid:
                        hits[mid] += 3  # vector match is a stronger signal

        hits.pop("", None)
        # Cap to the most-frequently matched memories so the pair-count stays
        # bounded regardless of how large conversation_text is.
        matched_list = [mid for mid, _ in hits.most_common(max_nodes)]

        new_connections = 0
        if len(matched_list) >= 2:
            pairs = [(matched_list[i], matched_list[j])
                     for i in range(len(matched_list))
                     for j in range(i + 1, len(matched_list))]
            self.db.connect_batch(pairs, weight=0.15)
            new_connections = len(pairs)

        for mid in matched_list:
            self.db.log_event(mid, "consolidated", "passive hebbian from conversation")

        return {
            "matched_memories": len(matched_list),
            "memory_ids": matched_list,
            "new_connections": new_connections,
        }

    def delete(self, memory_id: str, deleted_by: str = "manual") -> bool:
        """Delete a memory and its edges."""
        try:
            self._collection.delete(ids=[memory_id])
            self.db.delete(memory_id, deleted_by=deleted_by)
            return True
        except Exception:
            return False

    def dream_pass(self, short_decay_days: int = 14,
                   edge_decay_factor: float = 0.96,
                   strong_edge_decay_factor: float = 0.98,
                   auto_discover: bool = False) -> dict:
        """Legacy manual maintenance; automatic discovery is permanently disabled."""
        results = {}
        results["decayed_memories"] = 0

        results["pruned_edges"] = self.db.decay_edges(
            min_weight=0.1, decay_factor=edge_decay_factor
        )

        results["decayed_strong"] = self.db.decay_strong_edges(
            min_weight=1.5, decay_factor=strong_edge_decay_factor
        )

        results["auto_discovered"] = 0
        if auto_discover:
            results["auto_discover_disabled"] = True


        # 碎片整合(2026-05-22 砍掉, auto源已停)

        # Activation decay — cool down all memories each dream_pass
        # 2026-06-07: 0.85->0.6 调狠 + dream_pass改定时任务, 修反射弧热门循环
        results["activation_decayed"] = self.db.decay_activation(factor=0.82)

        return results

    def swap_pass(self, min_age_hours: int = 24,
                  output_dir: str = 'anchor-data/swap',
                  apply: bool = False) -> dict:
        """Scan memory for handoff / checklist / 进度 candidates older than N hours.

        apply=False: just write candidates jsonl, return stats.
        apply=True: also archive full memories + soft-delete from db.
        Pinned and exempt-tagged memories are skipped.
        """
        import json as _json
        import os as _os
        import sqlite3 as _sqlite3
        import datetime as _dt

        TAG_PATTERNS = ['handoff', 'checklist', '换窗备忘', 'window-memo',
                        'cc-migration', '接力棒', '临时']
        TEXT_PATTERNS = ['差下半场', '进度', '接力棒']
        EXEMPT_TAGS = ['milestone', 'insight', 'cognition', 'important',
                       '重要', '纠偏', 'architecture', '架构']

        db_file = self.db.db_path
        cutoff = _dt.datetime.now() - _dt.timedelta(hours=min_age_hours)
        cutoff_iso = cutoff.isoformat()

        conn = _sqlite3.connect(db_file)
        conn.row_factory = _sqlite3.Row
        rows = conn.execute(
            "SELECT memory_id, text, timestamp, tag, tier, pinned, "
            "emotion_score, context, level FROM memories ORDER BY timestamp DESC"
        ).fetchall()
        conn.close()

        candidates = []
        exempted = 0
        for r in rows:
            tag = (r['tag'] or '').lower()
            text = r['text'] or ''
            ts = r['timestamp'] or ''
            if r['pinned']:
                continue
            tag_hit = any(p.lower() in tag for p in TAG_PATTERNS)
            text_hit = any(p in text for p in TEXT_PATTERNS)
            if not (tag_hit or text_hit):
                continue
            if ts > cutoff_iso:
                continue
            if any(e.lower() in tag for e in EXEMPT_TAGS):
                exempted += 1
                continue
            candidates.append({
                'memory_id': r['memory_id'],
                'tag': r['tag'] or '',
                'tier': r['tier'] or '',
                'emotion': r['emotion_score'],
                'text_preview': text[:160],
                'timestamp': ts,
                'reason': 'tag_match' if tag_hit else 'text_match',
                '_full': dict(r),
            })

        candidates.sort(key=lambda x: x['timestamp'], reverse=True)

        _os.makedirs(output_dir, exist_ok=True)
        stamp = _dt.datetime.now().strftime('%Y%m%d_%H%M%S')
        cand_file = _os.path.join(output_dir, f'candidates_{stamp}.jsonl')
        with open(cand_file, 'w', encoding='utf-8') as f:
            for c in candidates:
                c_out = {k: v for k, v in c.items() if k != '_full'}
                f.write(_json.dumps(c_out, ensure_ascii=False) + '\n')

        result = {
            'total_memories': len(rows),
            'candidates': len(candidates),
            'exempted': exempted,
            'min_age_hours': min_age_hours,
            'cutoff_iso': cutoff_iso,
            'candidates_file': cand_file,
            'applied': False,
        }

        if apply and candidates:
            archive_file = _os.path.join(
                output_dir, f"archive_{_dt.datetime.now().strftime('%Y-%m')}.jsonl"
            )
            mode = 'a' if _os.path.exists(archive_file) else 'w'
            with open(archive_file, mode, encoding='utf-8') as f:
                for c in candidates:
                    a = dict(c['_full'])
                    a['_swapped_at'] = _dt.datetime.now().isoformat()
                    f.write(_json.dumps(a, ensure_ascii=False) + '\n')

            deleted = 0
            failed = []
            for c in candidates:
                ok = self.delete(c['memory_id'], deleted_by='swap_pass')
                if ok:
                    deleted += 1
                else:
                    failed.append(c['memory_id'])
            result['applied'] = True
            result['archive_file'] = archive_file
            result['deleted'] = deleted
            result['failed'] = failed

        return result
