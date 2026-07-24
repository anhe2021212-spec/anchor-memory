"""
Anchor Memory System — SQLite layer for graph-structured memory.

Handles: memory storage, tiered decay, synaptic edges (Hebbian learning),
emotion scoring, citation tracking, and graph operations.
"""

import sqlite3
import os
import random
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
import re
import dual_edge
import belief_graph
from time_utils import jst_day_bounds_utc, jst_range_bounds_utc


class AnchorDB:
    """SQLite storage with graph layer for memory synapses."""

    MAX_EDGE_WEIGHT = 10.0  # Synaptic saturation
    UPDATES_ACTIVATION_NUDGE = 0.05
    PLANE_LEVELS = frozenset({"raw", "understanding", "cognition"})
    EDGE_TYPES = frozenset({
        "lateral", "temporal", "derived_from", "updates",
        "SUPPORTED_BY", "GROUNDED_IN", "EVOKES", "backfill",
    })
    EDGE_TYPE_ALIASES = {
        "supported_by": "SUPPORTED_BY",
        "grounded_in": "GROUNDED_IN",
        "evokes": "EVOKES",
    }
    HEAT_EDGE_TYPES = frozenset({"lateral", "temporal", "derived_from"})
    READ_ONLY_EDGE_TYPES = frozenset({"backfill"})

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.kuzu_path = os.environ.get(
            "ANCHOR_KUZU_PATH",
            os.path.join(os.path.dirname(os.path.abspath(db_path)), "kuzu_db"),
        )
        self._kuzu_db = None
        self._kuzu_conn = None
        self._kuzu_executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="anchor-kuzu"
        )
        self._kuzu_outbox_lock = threading.Lock()
        self._init_tables()
        self._init_kuzu()
        if self._kuzu_conn:
            dual_edge.bootstrap_kuzu(self)
            belief_graph.bootstrap_kuzu(self)

    def _conn(self):
        conn = sqlite3.connect(self.db_path, timeout=5.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA busy_timeout = 5000")
        conn.execute("PRAGMA synchronous = NORMAL")
        return conn

    def _init_kuzu(self):
        """Open Kuzu on its dedicated thread; fall back to SQLite if it fails."""
        def open_graph():
            import kuzu

            os.makedirs(os.path.dirname(self.kuzu_path), exist_ok=True)
            database = kuzu.Database(self.kuzu_path)
            connection = kuzu.Connection(database)
            connection.execute(
                "CREATE NODE TABLE IF NOT EXISTS "
                "Memory(memory_id STRING, PRIMARY KEY(memory_id))"
            )
            connection.execute(
                "CREATE REL TABLE IF NOT EXISTS EDGE("
                "FROM Memory TO Memory, weight DOUBLE, edge_type STRING, "
                "created STRING, last_fired STRING)"
            )
            return database, connection

        try:
            self._kuzu_db, self._kuzu_conn = (
                self._kuzu_executor.submit(open_graph).result()
            )
        except Exception as exc:
            self._kuzu_db = None
            self._kuzu_conn = None
            print(f"[Kuzu] unavailable; using SQLite edge fallback: {exc}")

    @property
    def kuzu_available(self) -> bool:
        return self._kuzu_conn is not None

    @staticmethod
    def _consume_kuzu_rows(connection, query: str, params: dict) -> list:
        result = connection.execute(query, params)
        rows = []
        while result.has_next():
            rows.append(result.get_next())
        return rows

    def _kuzu_rows(self, query: str, params: dict = None) -> list:
        """Drain legacy writes, then read on Kuzu's dedicated thread."""
        if not self._kuzu_conn:
            return []
        self._drain_kuzu_outbox()
        dual_edge.drain(self)
        belief_graph.drain(self)
        return self._kuzu_executor.submit(
            self._consume_kuzu_rows,
            self._kuzu_conn,
            query,
            params or {},
        ).result()

    @staticmethod
    def _execute_kuzu_write(connection, query: str, params: dict):
        connection.execute(query, params)

    def _kuzu_write(self, query: str, params: dict = None):
        if not self._kuzu_conn:
            return
        self._kuzu_executor.submit(
            self._execute_kuzu_write,
            self._kuzu_conn,
            query,
            params or {},
        ).result()

    def _drain_kuzu_outbox(self):
        """Apply trigger-captured SQLite graph changes as absolute Kuzu state."""
        if not self._kuzu_conn:
            return
        with self._kuzu_outbox_lock:
            with self._conn() as conn:
                node_rows = conn.execute(
                    "SELECT memory_id, op, revision FROM kuzu_node_outbox"
                ).fetchall()
                edge_rows = conn.execute(
                    "SELECT source_id, target_id, op, weight, edge_type, "
                    "created, last_fired, revision FROM kuzu_edge_outbox"
                ).fetchall()
                if not node_rows and not edge_rows:
                    return
                valid_ids = {
                    row[0] for row in conn.execute(
                        "SELECT memory_id FROM memories"
                    ).fetchall()
                }

            node_upserts = [
                {"memory_id": row["memory_id"]}
                for row in node_rows
                if row["op"] == "upsert" and row["memory_id"] in valid_ids
            ]
            node_deletes = [
                {"memory_id": row["memory_id"]}
                for row in node_rows
                if row["op"] == "delete" or row["memory_id"] not in valid_ids
            ]
            edge_deletes = []
            edge_upserts = []
            for row in edge_rows:
                endpoints_exist = (
                    row["source_id"] in valid_ids
                    and row["target_id"] in valid_ids
                )
                if row["op"] == "delete" or not endpoints_exist:
                    edge_deletes.append({
                        "source_id": row["source_id"],
                        "target_id": row["target_id"],
                    })
                else:
                    edge_upserts.append({
                        "source_id": row["source_id"],
                        "target_id": row["target_id"],
                        "weight": float(row["weight"]),
                        "edge_type": row["edge_type"] or "lateral",
                        "created": row["created"] or "",
                        "last_fired": row["last_fired"] or "",
                    })

            if node_upserts:
                self._kuzu_write(
                    "UNWIND $rows AS row "
                    "MERGE (:Memory {memory_id: row.memory_id})",
                    {"rows": node_upserts},
                )
            if edge_deletes:
                self._kuzu_write(
                    """
                    UNWIND $rows AS row
                    MATCH (a:Memory)-[e:EDGE]->(b:Memory)
                    WHERE a.memory_id = row.source_id
                      AND b.memory_id = row.target_id
                    DELETE e
                    """,
                    {"rows": edge_deletes},
                )
            if edge_upserts:
                self._kuzu_write(
                    """
                    UNWIND $rows AS row
                    MATCH (a:Memory {memory_id: row.source_id}),
                          (b:Memory {memory_id: row.target_id})
                    MERGE (a)-[e:EDGE]->(b)
                    ON CREATE SET e.weight = row.weight,
                                  e.edge_type = row.edge_type,
                                  e.created = row.created,
                                  e.last_fired = row.last_fired
                    ON MATCH SET e.weight = row.weight,
                                 e.edge_type = row.edge_type,
                                 e.created = row.created,
                                 e.last_fired = row.last_fired
                    """,
                    {"rows": edge_upserts},
                )
            if node_deletes:
                self._kuzu_write(
                    """
                    UNWIND $rows AS row
                    MATCH (m:Memory {memory_id: row.memory_id})
                    DETACH DELETE m
                    """,
                    {"rows": node_deletes},
                )

            with self._conn() as conn:
                conn.executemany(
                    "DELETE FROM kuzu_node_outbox "
                    "WHERE memory_id=? AND revision=?",
                    [
                        (row["memory_id"], row["revision"])
                        for row in node_rows
                    ],
                )
                conn.executemany(
                    "DELETE FROM kuzu_edge_outbox "
                    "WHERE source_id=? AND target_id=? AND revision=?",
                    [
                        (
                            row["source_id"], row["target_id"],
                            row["revision"],
                        )
                        for row in edge_rows
                    ],
                )
                conn.commit()

    def _upsert_kuzu_memory(self, memory_id: str):
        if self._kuzu_conn and memory_id:
            self._kuzu_write(
                "MERGE (:Memory {memory_id: $memory_id})",
                {"memory_id": memory_id},
            )

    def _delete_kuzu_memory(self, memory_id: str):
        if self._kuzu_conn and memory_id:
            self._kuzu_write(
                "MATCH (m:Memory {memory_id: $memory_id}) DETACH DELETE m",
                {"memory_id": memory_id},
            )

    def _init_tables(self):
        with self._conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS memories (
                    memory_id   TEXT PRIMARY KEY,
                    text        TEXT NOT NULL,
                    timestamp   TEXT NOT NULL,
                    usage_count INTEGER DEFAULT 0,
                    last_used   TEXT,
                    tag         TEXT DEFAULT 'general',
                    tier        TEXT DEFAULT 'short',
                    pinned      INTEGER DEFAULT 0,
                    emotion_score REAL DEFAULT 0.5
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS edges (
                    source_id   TEXT NOT NULL,
                    target_id   TEXT NOT NULL,
                    weight      REAL DEFAULT 1.0,
                    created     TEXT NOT NULL,
                    last_fired  TEXT NOT NULL,
                    PRIMARY KEY (source_id, target_id),
                    FOREIGN KEY (source_id) REFERENCES memories(memory_id) ON DELETE CASCADE,
                    FOREIGN KEY (target_id) REFERENCES memories(memory_id) ON DELETE CASCADE
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(source_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(target_id)")
            # Comments table — memory as conversation space (design: Veille & 吱吱)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS comments (
                    comment_id  TEXT PRIMARY KEY,
                    memory_id   TEXT NOT NULL REFERENCES memories(memory_id) ON DELETE CASCADE,
                    content     TEXT NOT NULL,
                    author      TEXT DEFAULT 'ai',
                    reply_to    TEXT REFERENCES comments(comment_id),
                    read_by_ai  INTEGER DEFAULT 0,
                    read_by_human INTEGER DEFAULT 0,
                    created_at  TEXT NOT NULL
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_comments_memory ON comments(memory_id)")
            # Annotations — append-only notes on memories (design: Altair)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS annotations (
                    annotation_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    memory_id   TEXT NOT NULL REFERENCES memories(memory_id) ON DELETE CASCADE,
                    text        TEXT NOT NULL,
                    created_at  TEXT NOT NULL
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_annotations_memory ON annotations(memory_id)")
            # Event log — immutable record of all operations (inspired by event sourcing)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS events (
                    event_id    INTEGER PRIMARY KEY AUTOINCREMENT,
                    memory_id   TEXT,
                    event_type  TEXT NOT NULL,
                    detail      TEXT DEFAULT '',
                    created_at  TEXT NOT NULL
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_events_memory ON events(memory_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_events_type ON events(event_type)")
            # Trash — soft-delete buffer, auto-purge after 7 days
            conn.execute("""
                CREATE TABLE IF NOT EXISTS trash (
                    memory_id   TEXT PRIMARY KEY,
                    text        TEXT NOT NULL,
                    context     TEXT DEFAULT '',
                    timestamp   TEXT NOT NULL,
                    tag         TEXT DEFAULT '',
                    tier        TEXT DEFAULT '',
                    emotion_score REAL DEFAULT 0.5,
                    level       TEXT DEFAULT 'raw',
                    deleted_at  TEXT NOT NULL,
                    deleted_by  TEXT DEFAULT 'unknown'
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_trash_deleted ON trash(deleted_at)")
            conn.commit()
        self._ensure_context_column()
        self._ensure_visual_column()
        self._ensure_level_column()
        self._ensure_edge_type_column()
        self._ensure_kuzu_outbox()
        dual_edge.ensure_schema(self)
        belief_graph.ensure_schema(self)
        self._ensure_fts_tables()
        self._ensure_activation_column()

    def _ensure_context_column(self):
        """Add context column if missing. text = search summary, context = full original."""
        with self._conn() as conn:
            try:
                conn.execute("SELECT context FROM memories LIMIT 1")
            except sqlite3.OperationalError:
                conn.execute("ALTER TABLE memories ADD COLUMN context TEXT DEFAULT ''")
                conn.commit()

    def _ensure_visual_column(self):
        """Add visual_embedding column if missing. For Anchor Vision integration."""
        with self._conn() as conn:
            try:
                conn.execute("SELECT visual_embedding FROM memories LIMIT 1")
            except sqlite3.OperationalError:
                conn.execute("ALTER TABLE memories ADD COLUMN visual_embedding TEXT DEFAULT ''")
                conn.commit()


    def _ensure_level_column(self):
        """Add level column if missing. raw/understanding/cognition."""
        with self._conn() as conn:
            try:
                conn.execute("SELECT level FROM memories LIMIT 1")
            except sqlite3.OperationalError:
                conn.execute("ALTER TABLE memories ADD COLUMN level TEXT DEFAULT 'raw'")
                conn.commit()

    def _ensure_edge_type_column(self):
        """Add edge_type column if missing. lateral (横向关联) / vertical (纵向提炼)."""
        with self._conn() as conn:
            try:
                conn.execute("SELECT edge_type FROM edges LIMIT 1")
            except sqlite3.OperationalError:
                conn.execute("ALTER TABLE edges ADD COLUMN edge_type TEXT DEFAULT 'lateral'")
                conn.commit()


    def _ensure_kuzu_outbox(self):
        """Capture graph changes made by legacy SQLite-only processes."""
        with self._conn() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS kuzu_node_outbox (
                    memory_id TEXT PRIMARY KEY,
                    op TEXT NOT NULL,
                    revision INTEGER NOT NULL DEFAULT 1
                );
                CREATE TABLE IF NOT EXISTS kuzu_edge_outbox (
                    source_id TEXT NOT NULL,
                    target_id TEXT NOT NULL,
                    op TEXT NOT NULL,
                    weight REAL,
                    edge_type TEXT,
                    created TEXT,
                    last_fired TEXT,
                    revision INTEGER NOT NULL DEFAULT 1,
                    PRIMARY KEY (source_id, target_id)
                );

                CREATE TRIGGER IF NOT EXISTS trg_kuzu_memory_insert
                AFTER INSERT ON memories BEGIN
                    INSERT INTO kuzu_node_outbox(memory_id, op, revision)
                    VALUES (NEW.memory_id, 'upsert', 1)
                    ON CONFLICT(memory_id) DO UPDATE SET
                        op = 'upsert',
                        revision = kuzu_node_outbox.revision + 1;
                END;

                CREATE TRIGGER IF NOT EXISTS trg_kuzu_memory_delete
                AFTER DELETE ON memories BEGIN
                    INSERT INTO kuzu_node_outbox(memory_id, op, revision)
                    VALUES (OLD.memory_id, 'delete', 1)
                    ON CONFLICT(memory_id) DO UPDATE SET
                        op = 'delete',
                        revision = kuzu_node_outbox.revision + 1;
                END;

                CREATE TRIGGER IF NOT EXISTS trg_kuzu_edge_insert
                AFTER INSERT ON edges BEGIN
                    INSERT INTO kuzu_edge_outbox(
                        source_id, target_id, op, weight, edge_type,
                        created, last_fired, revision
                    ) VALUES (
                        NEW.source_id, NEW.target_id, 'upsert', NEW.weight,
                        COALESCE(NEW.edge_type, 'lateral'),
                        NEW.created, NEW.last_fired, 1
                    )
                    ON CONFLICT(source_id, target_id) DO UPDATE SET
                        op = 'upsert',
                        weight = excluded.weight,
                        edge_type = excluded.edge_type,
                        created = excluded.created,
                        last_fired = excluded.last_fired,
                        revision = kuzu_edge_outbox.revision + 1;
                END;

                CREATE TRIGGER IF NOT EXISTS trg_kuzu_edge_update
                AFTER UPDATE ON edges BEGIN
                    INSERT INTO kuzu_edge_outbox(
                        source_id, target_id, op, weight, edge_type,
                        created, last_fired, revision
                    ) VALUES (
                        NEW.source_id, NEW.target_id, 'upsert', NEW.weight,
                        COALESCE(NEW.edge_type, 'lateral'),
                        NEW.created, NEW.last_fired, 1
                    )
                    ON CONFLICT(source_id, target_id) DO UPDATE SET
                        op = 'upsert',
                        weight = excluded.weight,
                        edge_type = excluded.edge_type,
                        created = excluded.created,
                        last_fired = excluded.last_fired,
                        revision = kuzu_edge_outbox.revision + 1;
                END;

                CREATE TRIGGER IF NOT EXISTS trg_kuzu_edge_delete
                AFTER DELETE ON edges BEGIN
                    INSERT INTO kuzu_edge_outbox(
                        source_id, target_id, op, revision
                    ) VALUES (
                        OLD.source_id, OLD.target_id, 'delete', 1
                    )
                    ON CONFLICT(source_id, target_id) DO UPDATE SET
                        op = 'delete',
                        weight = NULL,
                        edge_type = NULL,
                        created = NULL,
                        last_fired = NULL,
                        revision = kuzu_edge_outbox.revision + 1;
                END;
            """)
            conn.commit()

    def _ensure_activation_column(self):
        """Ensure activation and heat audit schema exists before serving requests."""
        with self._conn() as conn:
            columns = {row[1] for row in conn.execute("PRAGMA table_info(memories)")}
            if "activation_score" not in columns:
                conn.execute("ALTER TABLE memories ADD COLUMN activation_score REAL DEFAULT 0.0")
            if "last_heated_at" not in columns:
                conn.execute("ALTER TABLE memories ADD COLUMN last_heated_at TEXT")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS heat_events (
                    event_id TEXT PRIMARY KEY,
                    source TEXT NOT NULL DEFAULT '',
                    memory_ids TEXT NOT NULL DEFAULT '',
                    boost REAL NOT NULL,
                    created_at TEXT NOT NULL
                )
            """)
            conn.commit()

    # ── Event Log (immutable) ──

    def log_event(self, memory_id: str, event_type: str, detail: str = ""):
        """Log an immutable event. Types: created, updated, searched, connected, annotated, deleted, visual_stored."""
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO events (memory_id, event_type, detail, created_at) VALUES (?, ?, ?, ?)",
                (memory_id, event_type, detail, datetime.utcnow().isoformat()),
            )
            conn.commit()

    def get_events(self, memory_id: str, limit: int = 50) -> list:
        """Get event history for a memory."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT event_id, event_type, detail, created_at FROM events "
                "WHERE memory_id = ? ORDER BY created_at DESC LIMIT ?",
                (memory_id, limit),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_recent_events(self, limit: int = 20, event_type: str = None) -> list:
        """Get recent events across all memories."""
        with self._conn() as conn:
            if event_type:
                rows = conn.execute(
                    "SELECT event_id, memory_id, event_type, detail, created_at FROM events "
                    "WHERE event_type = ? ORDER BY created_at DESC LIMIT ?",
                    (event_type, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT event_id, memory_id, event_type, detail, created_at FROM events "
                    "ORDER BY created_at DESC LIMIT ?",
                    (limit,),
                ).fetchall()
        return [dict(r) for r in rows]

    # ── Annotations (append-only) ──

    def annotate(self, memory_id: str, text: str) -> int:
        """Add an annotation to a memory. Append-only — never delete or edit."""
        with self._conn() as conn:
            exists = conn.execute("SELECT 1 FROM memories WHERE memory_id=?", (memory_id,)).fetchone()
            if not exists:
                return -1
            cur = conn.execute(
                "INSERT INTO annotations (memory_id, text, created_at) VALUES (?, ?, ?)",
                (memory_id, text, datetime.utcnow().isoformat()),
            )
            conn.commit()
        self.log_event(memory_id, "annotated", text[:100])
        return cur.lastrowid

    def get_annotations(self, memory_id: str) -> list:
        """Get all annotations for a memory, oldest first."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT annotation_id, text, created_at FROM annotations "
                "WHERE memory_id = ? ORDER BY created_at ASC",
                (memory_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    def search_annotations(self, query: str, limit: int = 5) -> list:
        """Search annotations text. Returns matching memory_ids."""
        words = query.strip().split()
        if not words:
            return []
        where = " AND ".join(["a.text LIKE ?"] * len(words))
        params = [f"%{w}%" for w in words]
        with self._conn() as conn:
            rows = conn.execute(
                f"SELECT DISTINCT a.memory_id, a.text, a.created_at FROM annotations a "
                f"WHERE {where} ORDER BY a.created_at DESC LIMIT ?",
                params + [limit],
            ).fetchall()
        return [dict(r) for r in rows]

    # ── Visual Embedding (Anchor Vision integration) ──

    def set_visual_embedding(self, memory_id: str, embedding_json: str):
        """Store a visual embedding (CLIP vector as JSON string) for a memory."""
        self._ensure_visual_column()
        with self._conn() as conn:
            conn.execute(
                "UPDATE memories SET visual_embedding = ? WHERE memory_id = ?",
                (embedding_json, memory_id),
            )
            conn.commit()

    def get_visual_embedding(self, memory_id: str) -> str:
        """Get visual embedding for a memory. Returns JSON string or empty."""
        self._ensure_visual_column()
        with self._conn() as conn:
            row = conn.execute(
                "SELECT visual_embedding FROM memories WHERE memory_id = ?",
                (memory_id,),
            ).fetchone()
        return row["visual_embedding"] if row and row["visual_embedding"] else ""

    def find_visual_memories(self) -> list:
        """Get all memories that have visual embeddings."""
        self._ensure_visual_column()
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT memory_id, text, visual_embedding FROM memories "
                "WHERE visual_embedding != '' AND visual_embedding IS NOT NULL"
            ).fetchall()
        return [dict(r) for r in rows]

    # ── Memory CRUD ──

    def _ensure_collection_column(self):
        """add collection column if missing. '' = daily memory; 'wenku' = corpus."""
        with self._conn() as conn:
            try:
                conn.execute("SELECT collection FROM memories LIMIT 1")
            except Exception:
                conn.execute("ALTER TABLE memories ADD COLUMN collection TEXT DEFAULT ''")
                conn.commit()

    def insert(self, memory_id: str, text: str, tag: str = "general",
               tier: str = "short", emotion_score: float = 0.5,
               context: str = "", level: str = "raw", collection: str = ""):
        """Insert or replace a memory. text = search summary, context = full original.
        level: raw (原始记录) / understanding (压缩理解) / cognition (认知提炼).
        collection: '' daily / 'wenku' corpus."""
        self._ensure_context_column()
        self._ensure_level_column()
        self._ensure_collection_column()
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO memories (memory_id, text, timestamp, tag, tier, emotion_score, context, level, collection)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(memory_id) DO UPDATE SET
                    text          = excluded.text,
                    tag           = excluded.tag,
                    tier          = excluded.tier,
                    context       = excluded.context,
                    level         = excluded.level,
                    collection    = excluded.collection,
                    timestamp     = COALESCE(memories.timestamp, excluded.timestamp),
                    emotion_score = COALESCE(memories.emotion_score, excluded.emotion_score)
                """,
                (memory_id, text, datetime.utcnow().isoformat(), tag, tier, emotion_score, context, level, collection),
            )
            conn.commit()
        self._upsert_kuzu_memory(memory_id)
        self.log_event(memory_id, "created", f"tag={tag} tier={tier} level={level} collection={collection}")
        try:
            self.fts_upsert(memory_id, text)
        except Exception:
            pass  # FTS is best-effort

    def get(self, memory_id: str) -> dict:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM memories WHERE memory_id = ?", (memory_id,)
            ).fetchone()
        return dict(row) if row else None

    def delete(self, memory_id: str, deleted_by: str = "unknown"):
        """Soft-delete: move to trash, then remove from memories."""
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc).isoformat()
        self.log_event(memory_id, "deleted", f"by: {deleted_by}")
        try:
            self.fts_delete(memory_id)
        except Exception:
            pass
        with self._conn() as conn:
            # copy to trash before deleting
            row = conn.execute(
                "SELECT memory_id, text, context, timestamp, tag, tier, emotion_score, level "
                "FROM memories WHERE memory_id = ?", (memory_id,)
            ).fetchone()
            if row:
                conn.execute(
                    "INSERT OR REPLACE INTO trash "
                    "(memory_id, text, context, timestamp, tag, tier, emotion_score, level, deleted_at, deleted_by) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (row[0], row[1], row[2] or '', row[3], row[4] or '', row[5] or '',
                     row[6] or 0.5, row[7] or 'raw', now, deleted_by)
                )
            conn.execute("DELETE FROM memories WHERE memory_id = ?", (memory_id,))
            conn.commit()
        self._delete_kuzu_memory(memory_id)

    def purge_trash(self, days: int = 7) -> int:
        """Permanently delete trash entries older than N days."""
        from datetime import datetime, timezone, timedelta
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        with self._conn() as conn:
            cursor = conn.execute("DELETE FROM trash WHERE deleted_at < ?", (cutoff,))
            count = cursor.rowcount
            conn.commit()
        return count

    def list_trash(self, limit: int = 50) -> list:
        """List items in trash."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT memory_id, text, tag, tier, deleted_at, deleted_by "
                "FROM trash ORDER BY deleted_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(r) for r in rows]

    def restore_from_trash(self, memory_id: str) -> bool:
        """Restore a memory from trash back to memories table."""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT memory_id, text, context, timestamp, tag, tier, emotion_score, level "
                "FROM trash WHERE memory_id = ?", (memory_id,)
            ).fetchone()
            if not row:
                return False
            conn.execute(
                "INSERT OR REPLACE INTO memories "
                "(memory_id, text, context, timestamp, tag, tier, emotion_score, level) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7])
            )
            conn.execute("DELETE FROM trash WHERE memory_id = ?", (memory_id,))
            conn.commit()
        self._upsert_kuzu_memory(memory_id)
        self.log_event(memory_id, "restored_from_trash")
        return True

    def list_all(self, limit: int = 50, offset: int = 0) -> list:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT memory_id, text, timestamp, tag, tier FROM memories "
                "ORDER BY timestamp DESC LIMIT ? OFFSET ?",
                (limit, offset)
            ).fetchall()
        return [dict(r) for r in rows]

    def list_collection(self, collection: str = "wenku", tag: str = None,
                        limit: int = 1000) -> list:
        """列出某 collection 的所有条目(轻量: 只取 id/text/tag/时间, 不碰向量)。给文库目录(TOC)用。
        tag: 限定 type(匹配 tag 第一段), None=不限。按时间升序(建船顺序)。"""
        self._ensure_collection_column()
        sql = ("SELECT memory_id, text, timestamp, tag FROM memories "
               "WHERE collection = ?")
        params = [collection]
        if tag:
            sql += " AND (tag = ? OR tag LIKE ?)"
            params += [tag, tag + ",%"]
        sql += " ORDER BY timestamp ASC LIMIT ?"
        params.append(limit)
        with self._conn() as conn:
            rows = conn.execute(sql, tuple(params)).fetchall()
        return [dict(r) for r in rows]

    def keyword_search(self, query: str, limit: int = 5, tag: str = None) -> list:
        """Search memories + annotations by keyword."""
        with self._conn() as conn:
            # Search in memory text
            if tag:
                rows = conn.execute(
                    "SELECT memory_id, text, timestamp, tag FROM memories "
                    "WHERE text LIKE ? AND tag LIKE ? LIMIT ?",
                    (f"%{query}%", f"%{tag}%", limit)
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT memory_id, text, timestamp, tag FROM memories "
                    "WHERE text LIKE ? LIMIT ?",
                    (f"%{query}%", limit)
                ).fetchall()
            results = [dict(r) for r in rows]
            found_ids = {r["memory_id"] for r in results}

            # Also search in annotations
            ann_rows = conn.execute(
                "SELECT DISTINCT a.memory_id FROM annotations a "
                "WHERE a.text LIKE ? LIMIT ?",
                (f"%{query}%", limit)
            ).fetchall()
            for ar in ann_rows:
                mid = ar["memory_id"]
                if mid not in found_ids:
                    mem = conn.execute(
                        "SELECT memory_id, text, timestamp, tag FROM memories "
                        "WHERE memory_id = ?", (mid,)
                    ).fetchone()
                    if mem:
                        results.append(dict(mem))
                        found_ids.add(mid)

        return results[:limit]


    # ── FTS5 BM25 Search (jieba tokenization) ──

    def _ensure_fts_tables(self):
        """Create FTS5 virtual table and mapping table for BM25 search."""
        with self._conn() as conn:
            conn.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(text_tokens)
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS fts_map (
                    memory_id TEXT PRIMARY KEY,
                    fts_rowid INTEGER NOT NULL
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_fts_map_rowid ON fts_map(fts_rowid)")
            conn.commit()
        # Auto-rebuild if fts_map is empty but memories exist
        with self._conn() as conn:
            fts_count = conn.execute("SELECT COUNT(*) FROM fts_map").fetchone()[0]
            mem_count = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
        if fts_count == 0 and mem_count > 0:
            print(f"[FTS] Index empty, rebuilding for {mem_count} memories...")
            self.rebuild_fts()
            print("[FTS] Rebuild complete.")

    def _tokenize(self, text: str) -> str:
        """Tokenize text using jieba. Filter single chars."""
        import jieba
        tokens = jieba.cut(text)
        return " ".join(t.strip() for t in tokens if len(t.strip()) >= 2)

    def _tokenize_query(self, query: str) -> str:
        """Tokenize query for FTS5 MATCH. Uses OR to match any term."""
        import jieba
        tokens = [t.strip() for t in jieba.cut(query) if len(t.strip()) >= 2]
        if not tokens:
            return ""
        sanitized = []
        for t in tokens:
            clean = re.sub(r'[^\w\u4e00-\u9fff]', '', t)
            if clean:
                sanitized.append(f'"{clean}"')
        return " OR ".join(sanitized) if sanitized else ""

    def fts_upsert(self, memory_id: str, text: str):
        """Insert or update tokenized text in FTS5 table."""
        tokens = self._tokenize(text)
        if not tokens.strip():
            return
        with self._conn() as conn:
            existing = conn.execute(
                "SELECT fts_rowid FROM fts_map WHERE memory_id = ?", (memory_id,)
            ).fetchone()
            if existing:
                conn.execute("DELETE FROM memories_fts WHERE rowid = ?", (existing['fts_rowid'],))
                conn.execute("DELETE FROM fts_map WHERE memory_id = ?", (memory_id,))
            cursor = conn.execute("INSERT INTO memories_fts(text_tokens) VALUES (?)", (tokens,))
            fts_rowid = cursor.lastrowid
            conn.execute(
                "INSERT INTO fts_map(memory_id, fts_rowid) VALUES (?, ?)",
                (memory_id, fts_rowid)
            )
            conn.commit()

    def fts_delete(self, memory_id: str):
        """Remove entry from FTS5 table."""
        with self._conn() as conn:
            existing = conn.execute(
                "SELECT fts_rowid FROM fts_map WHERE memory_id = ?", (memory_id,)
            ).fetchone()
            if existing:
                conn.execute("DELETE FROM memories_fts WHERE rowid = ?", (existing['fts_rowid'],))
                conn.execute("DELETE FROM fts_map WHERE memory_id = ?", (memory_id,))
                conn.commit()

    def bm25_search(self, query: str, limit: int = 10, tag: str = None,
                    collection: str = "exclude") -> list:
        """BM25 search using FTS5 + jieba tokenization.
        collection: 'exclude' / 'only' / 'all'."""
        match_expr = self._tokenize_query(query)
        if not match_expr:
            return []
        if collection == "only":
            coll_clause = "AND m.collection = 'wenku'"
        elif collection == "all":
            coll_clause = ""
        else:
            coll_clause = "AND (m.collection != 'wenku' OR m.collection IS NULL)"
        tag_clause = "AND m.tag LIKE ?" if tag else ""
        sql = f"""
            SELECT fm.memory_id, bm25(memories_fts) as score
            FROM memories_fts f
            JOIN fts_map fm ON f.rowid = fm.fts_rowid
            JOIN memories m ON fm.memory_id = m.memory_id
            WHERE memories_fts MATCH ? {tag_clause} {coll_clause}
            ORDER BY bm25(memories_fts)
            LIMIT ?
        """
        params = [match_expr]
        if tag:
            params.append(f"%{tag}%")
        params.append(limit)
        with self._conn() as conn:
            try:
                rows = conn.execute(sql, tuple(params)).fetchall()
            except Exception:
                return []
        return [{"memory_id": r["memory_id"], "bm25_score": r["score"]} for r in rows]

    def rebuild_fts(self) -> int:
        """Rebuild FTS index from all existing memories."""
        with self._conn() as conn:
            conn.execute("DELETE FROM memories_fts")
            conn.execute("DELETE FROM fts_map")
            conn.commit()
            rows = conn.execute("SELECT memory_id, text FROM memories").fetchall()
        count = 0
        for r in rows:
            try:
                self.fts_upsert(r["memory_id"], r["text"])
                count += 1
            except Exception:
                pass
        return count

    # ── Tier management ──

    def set_tag(self, memory_id: str, tag: str) -> bool:
        """Replace a memory's tag and record the change."""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT tag FROM memories WHERE memory_id = ?", (memory_id,)
            ).fetchone()
            if not row:
                return False
            old_tag = row["tag"] or ""
            conn.execute(
                "UPDATE memories SET tag = ? WHERE memory_id = ?", (tag, memory_id)
            )
            conn.commit()
        self.log_event(memory_id, "tag_changed", f"{old_tag} -> {tag}")
        return True

    def set_level(self, memory_id: str, level: str) -> bool:
        """Replace a reviewed memory level in SQLite and audit the correction."""
        level = (level or "").strip().lower()
        if level not in {"raw", "understanding", "cognition"}:
            raise ValueError("level must be raw, understanding, or cognition")
        with self._conn() as conn:
            row = conn.execute(
                "SELECT COALESCE(level,'raw') level FROM memories WHERE memory_id = ?",
                (memory_id,),
            ).fetchone()
            if not row:
                return False
            old_level = row["level"]
            conn.execute(
                "UPDATE memories SET level = ? WHERE memory_id = ?", (level, memory_id)
            )
            conn.commit()
        self.log_event(memory_id, "level_changed", f"{old_level} -> {level}")
        return True

    def set_tier(self, memory_id: str, tier: str):
        with self._conn() as conn:
            conn.execute("UPDATE memories SET tier = ? WHERE memory_id = ?", (tier, memory_id))
            conn.commit()

    def decay_short(self, days: int = 14) -> int:
        """Delete short-tier memories older than N days."""
        cutoff = (datetime.utcnow() - timedelta(days=days)).isoformat()
        with self._conn() as conn:
            doomed = conn.execute(
                "SELECT memory_id FROM memories WHERE tier = 'short' AND timestamp < ?",
                (cutoff,)
            ).fetchall()
            cursor = conn.execute(
                "DELETE FROM memories WHERE tier = 'short' AND timestamp < ?",
                (cutoff,)
            )
            conn.commit()
        for row in doomed:
            self._delete_kuzu_memory(row["memory_id"])
        return cursor.rowcount

    # ── Citation tracking ──

    def get_citation_count(self, memory_id: str) -> int:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT usage_count FROM memories WHERE memory_id = ?", (memory_id,)
            ).fetchone()
        return row["usage_count"] if row else 0

    def cite(self, memory_id: str):
        with self._conn() as conn:
            conn.execute(
                "UPDATE memories SET usage_count = usage_count + 1, last_used = ? WHERE memory_id = ?",
                (datetime.utcnow().isoformat(), memory_id),
            )
            conn.commit()

    # ── Emotion scoring ──

    def get_tier(self, memory_id: str) -> str:
        """Get the tier of a memory (core/long/short)."""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT tier FROM memories WHERE memory_id = ?", (memory_id,)
            ).fetchone()
            return row[0] if row else None

    def get_emotion_score(self, memory_id: str) -> float:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT emotion_score FROM memories WHERE memory_id = ?", (memory_id,)
            ).fetchone()
        if row and row["emotion_score"] is not None:
            return row["emotion_score"]
        return 0.5

    def set_emotion_score(self, memory_id: str, score: float):
        with self._conn() as conn:
            conn.execute(
                "UPDATE memories SET emotion_score = ? WHERE memory_id = ?",
                (max(0.0, min(1.0, score)), memory_id)
            )
            conn.commit()

    def equalize_emotion_scores(self, nudge: float = 0.05, threshold: float = 0.2) -> int:
        """Bidirectional emotion score equilibration across connected memories."""
        updated = 0
        adjacency = None
        if self._kuzu_conn:
            adjacency = {}
            graph_rows = self._kuzu_rows(
                "MATCH (a:Memory)-[e:EDGE]->(b:Memory) "
                "WHERE e.weight >= 0.5 RETURN a.memory_id, b.memory_id"
            )
            for source_id, target_id in graph_rows:
                adjacency.setdefault(source_id, []).append(target_id)
                adjacency.setdefault(target_id, []).append(source_id)

        with self._conn() as conn:
            memories = conn.execute(
                "SELECT memory_id, emotion_score FROM memories WHERE emotion_score IS NOT NULL"
            ).fetchall()

            for m in memories:
                mid = m["memory_id"]
                my_score = m["emotion_score"] or 0.5

                if adjacency is None:
                    neighbors = conn.execute("""
                        SELECT m.emotion_score FROM memories m
                        INNER JOIN edges e ON (e.target_id = m.memory_id AND e.source_id = ?)
                           OR (e.source_id = m.memory_id AND e.target_id = ?)
                        WHERE m.emotion_score IS NOT NULL AND e.weight >= 0.5
                    """, (mid, mid)).fetchall()
                    neighbor_scores = [n["emotion_score"] or 0.5 for n in neighbors]
                else:
                    neighbor_ids = adjacency.get(mid, [])
                    unique_ids = list(set(neighbor_ids))
                    if unique_ids:
                        qmarks = ",".join("?" * len(unique_ids))
                        score_rows = conn.execute(
                            f"SELECT memory_id, emotion_score FROM memories "
                            f"WHERE memory_id IN ({qmarks}) AND emotion_score IS NOT NULL",
                            tuple(unique_ids),
                        ).fetchall()
                        score_map = {
                            r["memory_id"]: r["emotion_score"] or 0.5
                            for r in score_rows
                        }
                        neighbor_scores = [
                            score_map[nid] for nid in neighbor_ids if nid in score_map
                        ]
                    else:
                        neighbor_scores = []

                if not neighbor_scores:
                    continue

                avg_neighbor = sum(neighbor_scores) / len(neighbor_scores)
                diff = avg_neighbor - my_score

                if abs(diff) > threshold:
                    new_score = my_score + nudge * (1 if diff > 0 else -1)
                    new_score = max(0.0, min(1.0, new_score))
                    conn.execute(
                        "UPDATE memories SET emotion_score = ? WHERE memory_id = ?",
                        (new_score, mid)
                    )
                    updated += 1

            conn.commit()
        return updated

    # ── Graph layer: synaptic edges ──

    @classmethod
    def canonical_edge_type(cls, edge_type: str) -> str:
        raw = (edge_type or "").strip()
        canonical = cls.EDGE_TYPE_ALIASES.get(raw.lower(), raw.lower())
        if canonical not in cls.EDGE_TYPES:
            raise ValueError(f"unknown edge type: {edge_type!r}")
        return canonical

    def validate_typed_edge(self, source_id: str, target_id: str,
                            edge_type: str) -> dict:
        """Validate one directed edge against the registered endpoint contract."""
        canonical = self.canonical_edge_type(edge_type)
        if canonical in self.READ_ONLY_EDGE_TYPES:
            raise ValueError(f"{canonical} is legacy read-only")
        if not source_id or not target_id or source_id == target_id:
            raise ValueError("typed edge requires two distinct memory ids")
        self._ensure_collection_column()
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT memory_id, COALESCE(level,'raw') level, "
                "COALESCE(collection,'') collection FROM memories "
                "WHERE memory_id IN (?, ?)", (source_id, target_id)
            ).fetchall()
        by_id = {row["memory_id"]: dict(row) for row in rows}
        if source_id not in by_id or target_id not in by_id:
            raise ValueError("typed edge endpoint does not exist")
        source, target = by_id[source_id], by_id[target_id]
        sl, tl = source["level"], target["level"]
        sc, tc = source["collection"], target["collection"]
        if dual_edge.enabled():
            if canonical in {"lateral", "temporal"}:
                return {"edge_type": canonical, "source": source, "target": target,
                        "edge_family": "flow"}
            if canonical == "GROUNDED_IN":
                raise ValueError("GROUNDED_IN is retained read-only in dual-edge mode")
            if canonical == "SUPPORTED_BY":
                valid = sc != "wenku" and tc != "wenku" and sl == "understanding" and tl == "raw"
            elif canonical == "EVOKES":
                valid = sc != "wenku" and sl in self.PLANE_LEVELS and tc == "wenku"
            else:  # updates / derived_from
                valid = True
            if not valid:
                raise ValueError(f"invalid {canonical} endpoints: {sl}/{sc or 'anchor'} -> {tl}/{tc or 'anchor'}")
            return {"edge_type": canonical, "source": source, "target": target,
                    "edge_family": "semantic"}
        daily = sc != "wenku" and tc != "wenku"
        if canonical in {"lateral", "temporal", "derived_from", "updates"}:
            valid = daily and sl == tl and sl in self.PLANE_LEVELS
        elif canonical == "SUPPORTED_BY":
            valid = daily and sl == "understanding" and tl == "raw"
        elif canonical == "GROUNDED_IN":
            valid = daily and sl == "cognition" and tl in {"raw", "understanding"}
        else:  # EVOKES
            valid = sc != "wenku" and sl in self.PLANE_LEVELS and tc == "wenku"
        if not valid:
            raise ValueError(
                f"invalid {canonical} endpoints: "
                f"{sl}/{sc or 'anchor'} -> {tl}/{tc or 'anchor'}"
            )
        return {"edge_type": canonical, "source": source, "target": target}

    def write_typed_edge(self, source_id: str, target_id: str,
                         edge_type: str, weight: float = 1.0,
                         replace_legacy: bool = False,
                         audit_note: str = "") -> dict:
        """Write exactly one directed typed edge; never creates a reverse edge."""
        if os.environ.get("ANCHOR_TYPED_GRAPH", "on").strip().lower() in {"0", "off", "false", "no"}:
            raise ValueError("typed graph writes disabled by ANCHOR_TYPED_GRAPH")
        spec = self.validate_typed_edge(source_id, target_id, edge_type)
        canonical = spec["edge_type"]
        weight = max(0.0, min(float(weight), self.MAX_EDGE_WEIGHT))
        if weight <= 0:
            raise ValueError("typed edge weight must be positive")
        if dual_edge.enabled():
            if canonical in dual_edge.FLOW_ROLES:
                result = dual_edge.write_flow(
                    self, source_id, target_id, weight, 1.0,
                    "manual", mode="manual",
                )
            else:
                result = dual_edge.write_semantic(
                    self, source_id, target_id, canonical,
                    strength=weight, conductance=0.0, confidence=1.0,
                    provenance="manual", review_state="approved",
                    created_by="agent", audit_note=audit_note,
                )
            self.log_event(
                source_id, "typed_edge_written",
                f"to={target_id} type={canonical} weight={weight} authoritative=dual",
            )
            return {**result, "edge_type": canonical, "weight": weight,
                    "replaced_edge_type": None}
        now = datetime.utcnow().isoformat()
        replaced = ""
        with self._conn() as conn:
            existing = conn.execute(
                "SELECT COALESCE(edge_type,'lateral') edge_type FROM edges "
                "WHERE source_id=? AND target_id=?", (source_id, target_id)
            ).fetchone()
            if existing and existing["edge_type"] != canonical:
                previous = existing["edge_type"]
                if not (replace_legacy and previous in {"lateral", "backfill"}):
                    raise ValueError(
                        f"edge pair already occupied by {previous}; "
                        "explicit legacy replacement required"
                    )
                replaced = previous
            conn.execute("""
                INSERT INTO edges
                    (source_id,target_id,weight,created,last_fired,edge_type)
                VALUES (?,?,?,?,?,?)
                ON CONFLICT(source_id,target_id) DO UPDATE SET
                    weight=MAX(edges.weight,excluded.weight),
                    last_fired=excluded.last_fired,
                    edge_type=excluded.edge_type
            """, (source_id, target_id, weight, now, now, canonical))
            conn.commit()
        self._drain_kuzu_outbox()
        detail = f"to={target_id} type={canonical} weight={weight}"
        if replaced:
            detail += f" replaced={replaced}"
        if audit_note:
            detail += f" note={audit_note[:240]}"
        self.log_event(source_id, "typed_edge_written", detail)
        return {"ok": True, "source_id": source_id, "target_id": target_id,
                "edge_type": canonical, "weight": weight,
                "replaced_edge_type": replaced or None}

    def write_flow_edge(self, source_id: str, target_id: str, weight: float = 1.0,
                        conductance: float = 1.0, provenance: str = "unknown",
                        mode: str = "auto") -> dict:
        return dual_edge.write_flow(self, source_id, target_id, weight,
                                    conductance, provenance, mode)

    def write_flow_pair(self, source_id: str, target_id: str, weight: float = 1.0,
                        conductance: float = 1.0, provenance: str = "manual",
                        mode: str = "manual") -> dict:
        return dual_edge.write_flow_pair(self, source_id, target_id, weight,
                                         conductance, provenance, mode)

    def write_semantic_edge(self, source_id: str, target_id: str, role: str,
                            strength: float = 1.0, conductance: float = 0.0,
                            confidence: float = 1.0, provenance: str = "manual",
                            review_state: str = "approved", created_by: str = "heng",
                            audit_note: str = "", valid_from: str = None,
                            valid_to: str = None) -> dict:
        role = self.canonical_edge_type(role)
        return dual_edge.write_semantic(self, source_id, target_id, role, strength,
            conductance, confidence, provenance, review_state, created_by,
            audit_note, valid_from, valid_to)

    def get_flow_neighbors(self, memory_id: str, direction: str = "outgoing",
                           min_weight: float = 0.0, limit: int = 20) -> list:
        return dual_edge.flow_neighbors(self, memory_id, direction, min_weight, limit)

    def get_semantic_neighbors(self, memory_id: str, direction: str = "outgoing",
                               roles=None, review_state: str = "approved",
                               limit: int = 20) -> list:
        return dual_edge.semantic_neighbors(self, memory_id, direction, roles,
                                            review_state, limit)

    def legacy_cross_layer_report(self) -> dict:
        """Aggregate legacy lateral/backfill violations; never enqueues rows."""
        with self._conn() as conn:
            rows = conn.execute("""
                SELECT COALESCE(e.edge_type,'lateral') edge_type,
                       COALESCE(s.level,'raw') source_level,
                       COALESCE(t.level,'raw') target_level,
                       COUNT(*) count
                FROM edges e
                JOIN memories s ON s.memory_id=e.source_id
                JOIN memories t ON t.memory_id=e.target_id
                WHERE COALESCE(e.edge_type,'lateral') IN ('lateral','backfill')
                  AND COALESCE(s.level,'raw') != COALESCE(t.level,'raw')
                GROUP BY 1,2,3 ORDER BY 1,2,3
            """).fetchall()
        groups = [dict(row) for row in rows]
        return {"total": sum(row["count"] for row in groups), "groups": groups,
                "queued": 0}

    def _upsert_kuzu_edge(self, source_id: str, target_id: str,
                          weight: float, now: str,
                          edge_type: str = "lateral"):
        if not self._kuzu_conn or source_id == target_id:
            return
        self._upsert_kuzu_memory(source_id)
        self._upsert_kuzu_memory(target_id)
        self._kuzu_write(
            """
            MATCH (a:Memory {memory_id: $source_id}),
                  (b:Memory {memory_id: $target_id})
            MERGE (a)-[e:EDGE]->(b)
            ON CREATE SET e.weight = $weight,
                          e.edge_type = $edge_type,
                          e.created = $now,
                          e.last_fired = $now
            ON MATCH SET e.weight =
                CASE WHEN e.weight + $weight > $max_weight
                     THEN $max_weight ELSE e.weight + $weight END,
                         e.last_fired = $now
            """,
            {
                "source_id": source_id,
                "target_id": target_id,
                "weight": float(weight),
                "edge_type": edge_type,
                "now": now,
                "max_weight": float(self.MAX_EDGE_WEIGHT),
            },
        )

    def _upsert_edge(self, conn, source_id: str, target_id: str,
                     weight: float, now: str, edge_type: str = "lateral"):
        # 自环护栏 (2026-06-18 B窗口, 对齐 upstream v1.10): 记忆不能连自己。
        if source_id == target_id:
            return
        conn.execute("""
            INSERT INTO edges
                (source_id, target_id, weight, created, last_fired, edge_type)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(source_id, target_id) DO UPDATE SET
                weight = MIN(edges.weight + excluded.weight, ?),
                last_fired = excluded.last_fired
        """, (
            source_id, target_id, weight, now, now, edge_type,
            self.MAX_EDGE_WEIGHT,
        ))
        if dual_edge.enabled() and edge_type in dual_edge.FLOW_ROLES:
            dual_edge._write_flow_conn(self, conn, source_id, target_id, weight,
                                       1.0, "manual", mode="manual", now=now)
        self._upsert_kuzu_edge(
            source_id, target_id, weight, now, edge_type=edge_type
        )

    def connect(self, source_id: str, target_id: str, weight: float = 1.0):
        """Create/strengthen a same-layer bidirectional lateral edge."""
        try:
            self.validate_typed_edge(source_id, target_id, "lateral")
        except ValueError:
            return False
        if dual_edge.enabled():
            dual_edge.write_flow_pair(
                self, source_id, target_id, weight, 1.0,
                provenance="manual", mode="manual",
            )
            self.log_event(source_id, "connected", f"to={target_id} weight={weight}")
            return True
        now = datetime.utcnow().isoformat()
        with self._conn() as conn:
            s = conn.execute(
                "SELECT 1 FROM memories WHERE memory_id=?", (source_id,)
            ).fetchone()
            t = conn.execute(
                "SELECT 1 FROM memories WHERE memory_id=?", (target_id,)
            ).fetchone()
            if not (s and t):
                return
            self._upsert_edge(conn, source_id, target_id, weight, now)
            self._upsert_edge(conn, target_id, source_id, weight, now)
            conn.commit()
        self.log_event(source_id, "connected", f"to={target_id} weight={weight}")
        return True

    def connect_batch(self, pairs: list, weight: float = 0.2):
        """Batch connect pairs of memories (for Hebbian learning)."""
        if dual_edge.enabled():
            for source_id, target_id in pairs:
                try:
                    self.validate_typed_edge(source_id, target_id, "lateral")
                    dual_edge.write_flow_pair(
                        self, source_id, target_id, weight, 1.0,
                        provenance="hebbian_legacy", mode="auto",
                    )
                except ValueError:
                    continue
            return
        now = datetime.utcnow().isoformat()
        with self._conn() as conn:
            for source_id, target_id in pairs:
                try:
                    self.validate_typed_edge(source_id, target_id, "lateral")
                except ValueError:
                    continue
                s = conn.execute(
                    "SELECT 1 FROM memories WHERE memory_id=?", (source_id,)
                ).fetchone()
                t = conn.execute(
                    "SELECT 1 FROM memories WHERE memory_id=?", (target_id,)
                ).fetchone()
                if s and t:
                    self._upsert_edge(conn, source_id, target_id, weight, now)
                    self._upsert_edge(conn, target_id, source_id, weight, now)
            conn.commit()
        if dual_edge.enabled():
            dual_edge.drain(self)

    # ── Updates edges (P2 时效边 2026-07-17) ──
    # 有向语义: source=新版, target=旧版, edge_type='updates'。
    # 与 lateral/vertical 不同, updates 边是单向的——旧版不"更新"新版。

    def mark_update(self, new_id: str, old_id: str, weight: float = 2.0) -> bool:
        """标记 new_id 是 old_id 的更新版；同一关系重复调用不重复记审计事件。"""
        if not new_id or not old_id or new_id == old_id:
            return False
        try:
            self.validate_typed_edge(new_id, old_id, "updates")
        except ValueError:
            return False
        now = datetime.utcnow().isoformat()
        already_marked = False
        with self._conn() as conn:
            s = conn.execute(
                "SELECT 1 FROM memories WHERE memory_id=?", (new_id,)
            ).fetchone()
            t = conn.execute(
                "SELECT 1 FROM memories WHERE memory_id=?", (old_id,)
            ).fetchone()
            if not (s and t):
                return False
            existing = conn.execute(
                "SELECT edge_type FROM edges WHERE source_id=? AND target_id=?",
                (new_id, old_id),
            ).fetchone()
            already_marked = bool(existing and existing["edge_type"] == "updates")
            # fail-closed: 如果 old_id 已经沿 updates 链走到 new_id 的后继，
            # 再落 new→old 会造环。UNION 去重也能安全检查已有脏环。
            would_cycle = conn.execute("""
                WITH RECURSIVE newer(memory_id) AS (
                    SELECT source_id FROM edges
                    WHERE edge_type = 'updates' AND target_id = ?
                    UNION
                    SELECT e.source_id FROM edges e
                    JOIN newer n ON e.target_id = n.memory_id
                    WHERE e.edge_type = 'updates'
                )
                SELECT 1 FROM newer WHERE memory_id = ? LIMIT 1
            """, (new_id, old_id)).fetchone()
            if would_cycle:
                return False
            # 不走 _upsert_edge: 它的 ON CONFLICT 不改 edge_type,
            # 已有 lateral 边(connect_to 常见)会吞掉 updates 语义。
            if not already_marked:
                conn.execute("""
                    INSERT INTO edges
                        (source_id, target_id, weight, created, last_fired, edge_type)
                    VALUES (?, ?, ?, ?, ?, 'updates')
                    ON CONFLICT(source_id, target_id) DO UPDATE SET
                        weight = MAX(edges.weight, excluded.weight),
                        last_fired = excluded.last_fired,
                        edge_type = 'updates'
                """, (new_id, old_id, weight, now, now))
                self._apply_updates_activation_bias_conn(conn, new_id, old_id, now)
            conn.commit()
        if self._kuzu_conn:
            self._kuzu_write(
                """
                MATCH (a:Memory {memory_id: $source_id}),
                      (b:Memory {memory_id: $target_id})
                MERGE (a)-[e:EDGE]->(b)
                ON CREATE SET e.weight = $weight, e.edge_type = 'updates',
                              e.created = $now, e.last_fired = $now
                ON MATCH SET e.weight =
                    CASE WHEN e.weight > $weight THEN e.weight ELSE $weight END,
                             e.edge_type = 'updates', e.last_fired = $now
                """,
                {"source_id": new_id, "target_id": old_id,
                 "weight": float(weight), "now": now},
            )
        if not already_marked:
            self.log_event(old_id, "superseded", f"by={new_id}")
        return True

    def resolve_updates(self, memory_ids: list, max_hops: int = 5) -> dict:
        """沿 updates 边追到最新版。返回 {旧id: 最新id}, 只含真有更新的。
        多个更新版取 timestamp 最新的; 环或超长链 fail-closed。读 SQLite 镜像(始终有写)。"""
        result = {}
        if not memory_ids:
            return result
        with self._conn() as conn:
            for mid in memory_ids:
                current, visited = mid, {mid}
                invalid_chain = False
                for _ in range(max_hops):
                    row = conn.execute("""
                        SELECT e.source_id FROM edges e
                        JOIN memories m ON m.memory_id = e.source_id
                        WHERE e.edge_type = 'updates' AND e.target_id = ?
                        ORDER BY m.timestamp DESC LIMIT 1
                    """, (current,)).fetchone()
                    if not row:
                        break
                    if row["source_id"] in visited:
                        invalid_chain = True
                        break
                    current = row["source_id"]
                    visited.add(current)
                else:
                    # 正好走满 max_hops 时再看一眼；仍有后继说明还没到最新版。
                    invalid_chain = conn.execute("""
                        SELECT 1 FROM edges
                        WHERE edge_type = 'updates' AND target_id = ?
                        LIMIT 1
                    """, (current,)).fetchone() is not None
                if not invalid_chain and current != mid:
                    result[mid] = current
        return result

    def remove_update_edge(self, source_id: str, target_id: str) -> bool:
        """agent review 专用：只删除精确匹配的 updates 边，并同步 Kuzu。"""
        with self._conn() as conn:
            cursor = conn.execute(
                "DELETE FROM edges WHERE source_id=? AND target_id=? "
                "AND edge_type='updates'",
                (source_id, target_id),
            )
            conn.commit()
            removed = cursor.rowcount == 1
        if not removed:
            return False
        self._drain_kuzu_outbox()
        self.log_event(target_id, "update_edge_removed", f"source={source_id}")
        return True

    def get_neighbors(self, memory_id: str, min_weight: float = 0.5,
                      limit: int = 5) -> list:
        if self._kuzu_conn:
            rows = self._kuzu_rows(
                """
                MATCH (a:Memory)-[e:EDGE]->(b:Memory)
                WHERE a.memory_id = $id AND e.weight >= $min_weight
                RETURN b.memory_id, e.weight
                ORDER BY e.weight DESC LIMIT $limit
                """,
                {
                    "id": memory_id,
                    "min_weight": float(min_weight),
                    "limit": int(limit),
                },
            )
            return [
                {"memory_id": row[0], "weight": row[1]}
                for row in rows
            ]

        with self._conn() as conn:
            rows = conn.execute("""
                SELECT target_id as memory_id, weight FROM edges
                WHERE source_id = ? AND weight >= ?
                ORDER BY weight DESC LIMIT ?
            """, (memory_id, min_weight, limit)).fetchall()
        return [dict(r) for r in rows]

    def get_typed_neighbors(self, memory_id: str, direction: str = "outgoing",
                            edge_types=None, min_weight: float = 0.0,
                            limit: int = 20) -> list:
        """Typed traversal with endpoint validation; legacy-invalid rows stay invisible."""
        if direction not in {"outgoing", "incoming", "both"}:
            raise ValueError("direction must be outgoing, incoming, or both")
        wanted = None
        if edge_types is not None:
            wanted = {self.canonical_edge_type(x) for x in edge_types}
        clauses, params = [], []
        if direction in {"outgoing", "both"}:
            clauses.append("e.source_id=?")
            params.append(memory_id)
        if direction in {"incoming", "both"}:
            clauses.append("e.target_id=?")
            params.append(memory_id)
        where = " OR ".join(clauses)
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT e.source_id,e.target_id,e.weight,"
                "COALESCE(e.edge_type,'lateral') edge_type FROM edges e "
                f"WHERE ({where}) AND e.weight>=? ORDER BY e.weight DESC",
                params + [float(min_weight)],
            ).fetchall()
        result = []
        for row in rows:
            try:
                edge_type = self.canonical_edge_type(row["edge_type"])
                if wanted is not None and edge_type not in wanted:
                    continue
                self.validate_typed_edge(row["source_id"], row["target_id"], edge_type)
            except ValueError:
                continue
            outgoing = row["source_id"] == memory_id
            result.append({
                "memory_id": row["target_id"] if outgoing else row["source_id"],
                "source_id": row["source_id"], "target_id": row["target_id"],
                "weight": row["weight"], "edge_type": edge_type,
                "direction": "outgoing" if outgoing else "incoming",
            })
            if len(result) >= int(limit):
                break
        return result

    def get_edge_weight(self, source_id: str, target_id: str):
        if self._kuzu_conn:
            rows = self._kuzu_rows(
                """
                MATCH (a:Memory)-[e:EDGE]->(b:Memory)
                WHERE a.memory_id = $source_id
                  AND b.memory_id = $target_id
                RETURN e.weight LIMIT 1
                """,
                {"source_id": source_id, "target_id": target_id},
            )
            return rows[0][0] if rows else None

        with self._conn() as conn:
            row = conn.execute(
                "SELECT weight FROM edges WHERE source_id = ? AND target_id = ?",
                (source_id, target_id)
            ).fetchone()
        return row["weight"] if row else None

    def decay_edges(self, min_weight: float = 0.1,
                    decay_factor: float = 0.9) -> int:
        """Decay both stores and prune weak edges without creating out-islands.
        updates 边(时效链 2026-07-17)豁免衰减与剪枝——事实取代不是联想强度, 不随时间松动。"""
        self._drain_kuzu_outbox()
        if not self._kuzu_conn:
            with self._conn() as conn:
                conn.execute(
                    "UPDATE edges SET weight = weight * ? "
                    "WHERE COALESCE(edge_type,'lateral') NOT IN ('updates','backfill')",
                    (decay_factor,)
                )
                cursor = conn.execute(
                    "DELETE FROM edges WHERE weight < ? "
                    "AND COALESCE(edge_type,'lateral') NOT IN ('updates','backfill') "
                    "AND rowid NOT IN ("
                    "SELECT rowid FROM (SELECT rowid, ROW_NUMBER() OVER ("
                    "PARTITION BY source_id "
                    "ORDER BY weight DESC, last_fired DESC, target_id ASC) AS rn "
                    "FROM edges) WHERE rn = 1)",
                    (min_weight,),
                )
                conn.commit()
            return cursor.rowcount

        self._kuzu_write(
            "MATCH ()-[e:EDGE]->() "
            "WHERE coalesce(e.edge_type,'lateral') NOT IN ('updates','backfill') "
            "SET e.weight = e.weight * $factor",
            {"factor": float(decay_factor)},
        )
        graph_rows = self._kuzu_rows(
            """
            MATCH (a:Memory)-[e:EDGE]->(b:Memory)
            RETURN a.memory_id, b.memory_id, e.weight, e.last_fired,
                   e.edge_type
            ORDER BY a.memory_id, e.weight DESC, e.last_fired DESC,
                     b.memory_id ASC
            """
        )
        protected_sources = set()
        doomed = []
        for source_id, target_id, weight, _last_fired, _etype in graph_rows:
            if source_id not in protected_sources:
                protected_sources.add(source_id)
            elif weight < min_weight and (_etype or "lateral") not in {"updates", "backfill"}:
                doomed.append({
                    "source_id": source_id,
                    "target_id": target_id,
                })
        if doomed:
            self._kuzu_write(
                """
                UNWIND $rows AS row
                MATCH (a:Memory)-[e:EDGE]->(b:Memory)
                WHERE a.memory_id = row.source_id
                  AND b.memory_id = row.target_id
                DELETE e
                """,
                {"rows": doomed},
            )

        with self._conn() as conn:
            conn.execute(
                "UPDATE edges SET weight = weight * ? "
                "WHERE COALESCE(edge_type,'lateral') NOT IN ('updates','backfill')",
                (decay_factor,)
            )
            sqlite_cursor = conn.execute(
                "DELETE FROM edges WHERE weight < ? "
                "AND COALESCE(edge_type,'lateral') NOT IN ('updates','backfill') "
                "AND rowid NOT IN ("
                "SELECT rowid FROM (SELECT rowid, ROW_NUMBER() OVER ("
                "PARTITION BY source_id "
                "ORDER BY weight DESC, last_fired DESC, target_id ASC) AS rn "
                "FROM edges) WHERE rn = 1)",
                (min_weight,),
            )
            conn.commit()
        if sqlite_cursor.rowcount != len(doomed):
            print(
                "[Kuzu] decay prune count differs from SQLite fallback: "
                f"kuzu={len(doomed)} sqlite={sqlite_cursor.rowcount}"
            )
        return len(doomed)

    def decay_strong_edges(self, min_weight: float = 1.5,
                           decay_factor: float = 0.95) -> int:
        """Slowly decay strong manual edges in both graph stores.
        updates 边豁免——同 decay_edges。"""
        self._drain_kuzu_outbox()
        if not self._kuzu_conn:
            with self._conn() as conn:
                cursor = conn.execute(
                    "UPDATE edges SET weight = weight * ? WHERE weight >= ? "
                    "AND COALESCE(edge_type,'lateral') NOT IN ('updates','backfill')",
                    (decay_factor, min_weight)
                )
                conn.commit()
            return cursor.rowcount

        count_rows = self._kuzu_rows(
            "MATCH ()-[e:EDGE]->() WHERE e.weight >= $min_weight "
            "AND coalesce(e.edge_type,'lateral') NOT IN ('updates','backfill') "
            "RETURN count(e)",
            {"min_weight": float(min_weight)},
        )
        count = int(count_rows[0][0]) if count_rows else 0
        self._kuzu_write(
            "MATCH ()-[e:EDGE]->() WHERE e.weight >= $min_weight "
            "AND coalesce(e.edge_type,'lateral') NOT IN ('updates','backfill') "
            "SET e.weight = e.weight * $factor",
            {
                "min_weight": float(min_weight),
                "factor": float(decay_factor),
            },
        )
        with self._conn() as conn:
            conn.execute(
                "UPDATE edges SET weight = weight * ? WHERE weight >= ? "
                "AND COALESCE(edge_type,'lateral') NOT IN ('updates','backfill')",
                (decay_factor, min_weight)
            )
            conn.commit()
        return count

    # ── Pinning ──

    def pin(self, memory_id: str):
        with self._conn() as conn:
            conn.execute("UPDATE memories SET pinned = 1 WHERE memory_id = ?", (memory_id,))
            conn.commit()

    def unpin(self, memory_id: str):
        with self._conn() as conn:
            conn.execute("UPDATE memories SET pinned = 0 WHERE memory_id = ?", (memory_id,))
            conn.commit()

    def get_pinned(self) -> list:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT memory_id, text, timestamp, tag FROM memories WHERE pinned = 1"
            ).fetchall()
        return [dict(r) for r in rows]

    # ── Comments: memory as conversation space (design: Veille & 吱吱) ──

    def insert_comment(self, memory_id: str, content: str,
                       author: str = "ai", reply_to: str = None) -> str:
        import uuid
        comment_id = f"comment_{uuid.uuid4().hex[:12]}"
        with self._conn() as conn:
            exists = conn.execute("SELECT 1 FROM memories WHERE memory_id=?", (memory_id,)).fetchone()
            if not exists:
                return ""
            conn.execute(
                "INSERT INTO comments (comment_id, memory_id, content, author, reply_to, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (comment_id, memory_id, content, author, reply_to,
                 datetime.utcnow().isoformat()),
            )
            conn.commit()
        return comment_id

    def get_comments(self, memory_id: str) -> list:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM comments WHERE memory_id = ? ORDER BY created_at",
                (memory_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_unread_comments(self, reader: str = "ai") -> list:
        col = "read_by_ai" if reader == "ai" else "read_by_human"
        with self._conn() as conn:
            rows = conn.execute(
                f"SELECT c.*, m.text as memory_text "
                f"FROM comments c JOIN memories m ON c.memory_id = m.memory_id "
                f"WHERE c.{col} = 0 ORDER BY c.created_at",
            ).fetchall()
        return [dict(r) for r in rows]

    def mark_comments_read(self, comment_ids: list, reader: str = "ai"):
        col = "read_by_ai" if reader == "ai" else "read_by_human"
        with self._conn() as conn:
            for cid in comment_ids:
                conn.execute(
                    f"UPDATE comments SET {col} = 1 WHERE comment_id = ?", (cid,)
                )
            conn.commit()

    # ── Wakeup: one-call cold start (design: Veille & 吱吱) ──

    def wakeup(self, n_high_emotion: int = 5, n_random: int = 3,
               high_emotion_days: int = 7, emotion_threshold: float = 0.7) -> dict:
        """Gather everything needed for cold start in one call.

        Returns: pinned, high_emotion (real), latest_diary, latest_memo,
                 random_old, unread_comments.

        v2 changes (2026-04-24):
        - high_emotion now actually filters by emotion_score >= threshold
        - Added latest_diary: most recent diary entry (tag contains 'diary')
        - Added latest_memo: most recent 换窗备忘 (tag contains 'memo')
        - Extended high_emotion window to 7 days
        - Increased random_old to 3
        """
        self._ensure_context_column()
        cutoff = (datetime.utcnow() - timedelta(days=high_emotion_days)).isoformat()

        with self._conn() as conn:
            pinned = conn.execute(
                "SELECT memory_id, text, tag, emotion_score, context FROM memories "
                "WHERE pinned = 1 ORDER BY timestamp"
            ).fetchall()

            high_emotion = conn.execute(
                "SELECT memory_id, text, tag, emotion_score, timestamp, context FROM memories "
                "WHERE emotion_score >= ? AND pinned = 0 "
                "ORDER BY timestamp DESC LIMIT ?",
                (emotion_threshold, n_high_emotion),
            ).fetchall()

            latest_recent = conn.execute(
                "SELECT memory_id, text, tag, emotion_score, timestamp FROM memories "
                "WHERE pinned = 0 "
                "ORDER BY timestamp DESC LIMIT 3"
            ).fetchall()

            random_old = conn.execute(
                "SELECT memory_id, text, tag, emotion_score, timestamp, context FROM memories "
                "WHERE timestamp < ? AND pinned = 0 "
                "ORDER BY RANDOM() LIMIT ?",
                (cutoff, n_random),
            ).fetchall()

            unread = conn.execute(
                "SELECT c.comment_id, c.memory_id, c.content, c.author, c.created_at, "
                "m.text as memory_text FROM comments c "
                "JOIN memories m ON c.memory_id = m.memory_id "
                "WHERE c.read_by_ai = 0 ORDER BY c.created_at"
            ).fetchall()

        return {
            "pinned": [dict(r) for r in pinned],
            "high_emotion": [dict(r) for r in high_emotion],
            "latest_recent": [dict(r) for r in latest_recent],
            "random_old": [dict(r) for r in random_old],
            "unread_comments": [dict(r) for r in unread],
        }


    # ── Activation (spreading activation + hot/cold) ──

    def _apply_updates_activation_bias_conn(self, conn, new_id: str, old_id: str,
                                            now: str = None) -> bool:
        """On first UPDATES link creation, gently favor new over old, once."""
        now = now or datetime.utcnow().isoformat()
        amount = self.UPDATES_ACTIVATION_NUDGE
        inserted = conn.execute(
            "INSERT OR IGNORE INTO heat_events(event_id,source,memory_ids,boost,created_at) "
            "VALUES(?,?,?,?,?)",
            (f"updates-bias:{new_id}:{old_id}", "updates_edge_bias",
             f"{new_id},{old_id}", amount, now),
        )
        if inserted.rowcount == 0:
            return False
        conn.execute(
            "UPDATE memories SET activation_score=MIN(8.0,"
            "COALESCE(activation_score,0)+?),last_heated_at=? WHERE memory_id=?",
            (amount, now, new_id),
        )
        conn.execute(
            "UPDATE memories SET activation_score=MAX(0.0,"
            "COALESCE(activation_score,0)-?) WHERE memory_id=?",
            (amount, old_id),
        )
        return True

    def apply_heat(self, memory_ids, boost: float, event_id: str,
                   spread: bool = True, spread_factor: float = 0.5,
                   max_depth: int = 3, neighbor_limit: int = 8,
                   node_budget: int = 64, edge_budget: int = 128,
                   source: str = "") -> dict:
        """Idempotently heat final touched memories and spread only through flow_edges."""
        roots = list(dict.fromkeys(
            str(memory_id).strip() for memory_id in (memory_ids or [])
            if str(memory_id).strip()
        ))[:32]
        event_id = str(event_id or "").strip()
        if not roots or boost <= 0 or not event_id:
            return {"applied": False, "duplicate": False, "nodes": 0, "edges": 0}
        boost = min(float(boost), 8.0)
        max_depth = max(0, min(int(max_depth), 3))
        neighbor_limit = max(1, min(int(neighbor_limit), 16))
        node_budget = max(len(roots), min(int(node_budget), 128))
        edge_budget = max(0, min(int(edge_budget), 256))
        now = datetime.utcnow().isoformat()
        self._ensure_activation_column()
        heated = []
        fired = []
        with self._conn() as conn:
            conn.execute("BEGIN IMMEDIATE")
            inserted = conn.execute(
                "INSERT OR IGNORE INTO heat_events(event_id,source,memory_ids,boost,created_at) "
                "VALUES(?,?,?,?,?)",
                (event_id, str(source or "")[:80], ",".join(roots), boost, now),
            )
            if inserted.rowcount == 0:
                conn.rollback()
                return {"applied": False, "duplicate": True, "nodes": 0, "edges": 0}
            existing = {row[0] for row in conn.execute(
                f"SELECT memory_id FROM memories WHERE memory_id IN ({','.join('?' for _ in roots)})",
                roots,
            )}
            queue = [(memory_id, boost, 0) for memory_id in roots if memory_id in existing]
            visited = {memory_id for memory_id, _, _ in queue}
            index = 0
            while index < len(queue) and len(heated) < node_budget:
                memory_id, amount, depth = queue[index]
                index += 1
                conn.execute(
                    "UPDATE memories SET activation_score=MIN(8.0,COALESCE(activation_score,0)+?), "
                    "last_heated_at=? WHERE memory_id=?",
                    (amount, now, memory_id),
                )
                heated.append((memory_id, amount))
                if not spread or depth >= max_depth or len(fired) >= edge_budget:
                    continue
                rows = conn.execute(
                    "SELECT source_id,target_id,weight,conductance FROM flow_edges "
                    "WHERE source_id=? AND weight>0 AND conductance>0 "
                    "ORDER BY weight*conductance DESC,target_id LIMIT ?",
                    (memory_id, neighbor_limit),
                ).fetchall()
                for row in rows:
                    target_id = row["target_id"]
                    if (target_id in visited or len(queue) >= node_budget
                            or len(fired) >= edge_budget):
                        continue
                    amount_next = (amount * float(spread_factor)
                                   * min(float(row["weight"]) / 1.5, 1.0)
                                   * float(row["conductance"]))
                    if amount_next <= 0:
                        continue
                    visited.add(target_id)
                    queue.append((target_id, amount_next, depth + 1))
                    conn.execute(
                        "UPDATE flow_edges SET last_fired=? WHERE source_id=? AND target_id=?",
                        (now, row["source_id"], target_id),
                    )
                    fired.append((row["source_id"], target_id))
            conn.commit()
        if fired and self._kuzu_conn:
            dual_edge.drain(self)
        return {"applied": True, "duplicate": False, "nodes": len(heated),
                "edges": len(fired), "heated": heated, "fired": fired}

    def activate(self, memory_id: str, boost: float = 1.0,
                 spread_factor: float = 0.5, max_depth: int = 3,
                 _visited: set = None):
        """Compatibility wrapper around the unified heat entrypoint."""
        return self.apply_heat(
            [memory_id], boost, f"legacy-activate:{uuid.uuid4().hex}",
            spread=True, spread_factor=spread_factor, max_depth=max_depth,
            source="legacy_activate",
        )

    def decay_activation(self, factor: float = 0.82,
                         hub_quantile: float = 0.9) -> int:
        """Daily activation decay using authoritative flow_edges; retention <= 0.90."""
        self._ensure_activation_column()
        base = max(0.0, min(float(factor), 0.90))
        with self._conn() as conn:
            sums = [float(row[0]) for row in conn.execute(
                "SELECT SUM(weight*conductance) FROM flow_edges "
                "WHERE weight>0 AND conductance>0 GROUP BY source_id"
            ).fetchall() if row[0] is not None]
            if sums:
                sums.sort()
                idx = int(float(hub_quantile) * (len(sums) - 1) + 0.5)
                hub_ref = max(sums[max(0, min(len(sums) - 1, idx))], 0.001)
            else:
                hub_ref = 1.0
            cursor = conn.execute("""
                UPDATE memories SET activation_score = MIN(8.0, activation_score * MIN(0.90,
                    ? + 0.05 * MIN(1.0, COALESCE((
                        SELECT SUM(weight*conductance) FROM flow_edges
                        WHERE source_id=memories.memory_id AND weight>0 AND conductance>0
                    ),0) / ?)
                    + 0.03 * MAX(0.0, MIN(1.0, (COALESCE(emotion_score,0.5)-0.5)*2.0))
                )) WHERE activation_score > 0.02
            """, (base, hub_ref))
            conn.execute(
                "UPDATE memories SET activation_score=0 WHERE activation_score<=0.02"
            )
            conn.commit()
            return cursor.rowcount

    def cool_activation(self, memory_id: str, amount: float = 2.0) -> float:
        """浮现冷却(2026-06-07): 反射弧第四槽浮现过的记忆扣 activation,
        让它退坑给别的记忆, 打断热门循环。clamp 到 [0,20]。返回新值。"""
        self._ensure_activation_column()
        with self._conn() as conn:
            row = conn.execute(
                "SELECT activation_score FROM memories WHERE memory_id = ?",
                (memory_id,)
            ).fetchone()
            if not row:
                return 0.0
            new_score = max(0.0, min((row["activation_score"] or 0) - amount, 20.0))
            conn.execute(
                "UPDATE memories SET activation_score = ? WHERE memory_id = ?",
                (new_score, memory_id)
            )
            conn.commit()
        return new_score

    def get_hot_ids(self, threshold: float = 2.0) -> set:
        """Return memory_ids with activation_score above threshold."""
        self._ensure_activation_column()
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT memory_id FROM memories WHERE activation_score >= ?",
                (threshold,)
            ).fetchall()
        return {r["memory_id"] for r in rows}

    def get_activation(self, memory_id: str) -> float:
        """Get current activation_score for a memory."""
        self._ensure_activation_column()
        with self._conn() as conn:
            row = conn.execute(
                "SELECT activation_score FROM memories WHERE memory_id = ?",
                (memory_id,)
            ).fetchone()
        return (row["activation_score"] or 0.0) if row else 0.0

    def get_hot(self, n: int = 5, threshold: float = 2.0,
                exclude_ids: set = None) -> list:
        """返回最热的n条记忆(完整行), 按 activation_score 降序。
        同分随机打散, 避免总是同一条浮上来; exclude_ids 跳过。"""
        self._ensure_activation_column()
        exclude_ids = exclude_ids or set()
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT memory_id, text, tag, tier, emotion_score, timestamp, "
                "activation_score FROM memories WHERE activation_score >= ? "
                "ORDER BY activation_score DESC, RANDOM() LIMIT ?",
                (threshold, max(n * 3, 15)),
            ).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            if d["memory_id"] in exclude_ids:
                continue
            out.append(d)
            if len(out) >= n:
                break
        return out

    def get_hot_neighbors(self, seed_ids, exclude_ids=None,
                          threshold: float = 2.0, n: int = 5) -> list:
        """Return hot one-hop graph neighbors with the seed that bridged them."""
        self._ensure_activation_column()
        seed_ids = [s for s in (seed_ids or []) if s]
        if not seed_ids:
            return []
        exclude = set(exclude_ids or [])
        seed_set = set(seed_ids)

        if not self._kuzu_conn:
            qmarks = ",".join("?" * len(seed_ids))
            with self._conn() as conn:
                rows = conn.execute(
                    f"""SELECT e.target_id AS memory_id,
                               e.source_id AS bridge_id,
                               e.weight AS weight, m.text AS text,
                               m.tag AS tag, m.timestamp AS timestamp,
                               m.activation_score AS activation_score,
                               s.text AS bridge_text, s.tag AS bridge_tag
                        FROM edges e
                        JOIN memories m ON m.memory_id = e.target_id
                        JOIN memories s ON s.memory_id = e.source_id
                        WHERE e.source_id IN ({qmarks})
                          AND m.activation_score >= ?
                        ORDER BY m.activation_score DESC,
                                 e.weight DESC, RANDOM()""",
                    (*seed_ids, threshold),
                ).fetchall()
            candidates = [dict(row) for row in rows]
        else:
            graph_rows = self._kuzu_rows(
                """
                MATCH (s:Memory)-[e:EDGE]->(m:Memory)
                WHERE s.memory_id IN $seed_ids
                RETURN m.memory_id, s.memory_id, e.weight
                """,
                {"seed_ids": seed_ids},
            )
            all_ids = {
                value
                for target_id, bridge_id, _weight in graph_rows
                for value in (target_id, bridge_id)
            }
            metadata = {}
            if all_ids:
                qmarks = ",".join("?" * len(all_ids))
                with self._conn() as conn:
                    rows = conn.execute(
                        f"SELECT memory_id, text, tag, timestamp, "
                        f"activation_score FROM memories "
                        f"WHERE memory_id IN ({qmarks})",
                        tuple(all_ids),
                    ).fetchall()
                metadata = {row["memory_id"]: dict(row) for row in rows}

            candidates = []
            for target_id, bridge_id, weight in graph_rows:
                target = metadata.get(target_id)
                bridge = metadata.get(bridge_id)
                if not target or not bridge:
                    continue
                activation = target.get("activation_score") or 0.0
                if activation < threshold:
                    continue
                candidates.append({
                    "memory_id": target_id,
                    "bridge_id": bridge_id,
                    "weight": weight,
                    "text": target.get("text"),
                    "tag": target.get("tag"),
                    "timestamp": target.get("timestamp"),
                    "activation_score": activation,
                    "bridge_text": bridge.get("text"),
                    "bridge_tag": bridge.get("tag"),
                })
            random.shuffle(candidates)
            candidates.sort(
                key=lambda row: (
                    row["activation_score"], row["weight"]
                ),
                reverse=True,
            )

        out, seen = [], set()
        for row in candidates:
            mid = row["memory_id"]
            if mid in exclude or mid in seed_set or mid in seen:
                continue
            seen.add(mid)
            out.append(row)
            if len(out) >= n:
                break
        return out

    # ── Dream Events: iOS感知层 (design: heartbeat system) ──

    def _ensure_dream_events_table(self):
        """Create dream_events table if not exists."""
        with self._conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS dream_events (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    type        TEXT NOT NULL,
                    value       TEXT NOT NULL,
                    created_at  TEXT NOT NULL
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_dream_events_type ON dream_events(type)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_dream_events_time ON dream_events(created_at)")
            conn.commit()

    def insert_dream_event(self, event_type: str, value: str) -> bool:
        """Insert a dream event with 5-min dedup per type."""
        self._ensure_dream_events_table()
        cutoff = (datetime.utcnow() - timedelta(minutes=5)).isoformat()
        with self._conn() as conn:
            existing = conn.execute(
                "SELECT id FROM dream_events WHERE type = ? AND created_at > ? LIMIT 1",
                (event_type, cutoff)
            ).fetchone()
            if existing:
                return False  # dedup
            conn.execute(
                "INSERT INTO dream_events (type, value, created_at) VALUES (?, ?, ?)",
                (event_type, value, datetime.utcnow().isoformat())
            )
            conn.commit()
        return True

    def get_recent_dream_events(self, hours: int = 6, limit: int = 20) -> list:
        """Get recent dream events for keepalive prompt injection."""
        self._ensure_dream_events_table()
        cutoff = (datetime.utcnow() - timedelta(hours=hours)).isoformat()
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT type, value, created_at FROM dream_events "
                "WHERE created_at > ? ORDER BY created_at DESC LIMIT ?",
                (cutoff, limit)
            ).fetchall()
        return [dict(r) for r in rows]

    # ── Keepalive Messages: 意识连续性 (design: heartbeat system) ──

    def _ensure_keepalive_table(self):
        """Create keepalive_messages table if not exists."""
        with self._conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS keepalive_messages (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    action_type     TEXT NOT NULL,
                    thoughts        TEXT DEFAULT '',
                    content         TEXT DEFAULT '',
                    created_at      TEXT NOT NULL,
                    consumed        INTEGER DEFAULT 0
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_keepalive_consumed ON keepalive_messages(consumed)")
            conn.commit()

    def insert_keepalive_message(self, action_type: str, thoughts: str, content: str) -> int:
        """Store a keepalive action result."""
        self._ensure_keepalive_table()
        with self._conn() as conn:
            cur = conn.execute(
                "INSERT INTO keepalive_messages (action_type, thoughts, content, created_at) "
                "VALUES (?, ?, ?, ?)",
                (action_type, thoughts, content, datetime.utcnow().isoformat())
            )
            conn.commit()
        return cur.lastrowid

    def get_pending_keepalive(self) -> list:
        """Get all unconsumed keepalive messages."""
        self._ensure_keepalive_table()
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT id, action_type, thoughts, content, created_at "
                "FROM keepalive_messages WHERE consumed = 0 ORDER BY created_at ASC"
            ).fetchall()
        return [dict(r) for r in rows]

    def consume_keepalive(self) -> int:
        """Mark all pending keepalive messages as consumed. Returns count."""
        self._ensure_keepalive_table()
        with self._conn() as conn:
            cur = conn.execute("UPDATE keepalive_messages SET consumed = 1 WHERE consumed = 0")
            conn.commit()
        return cur.rowcount

    def cleanup_old_keepalive(self, days: int = 7) -> int:
        """Clean up old consumed keepalive messages."""
        self._ensure_keepalive_table()
        cutoff = (datetime.utcnow() - timedelta(days=days)).isoformat()
        with self._conn() as conn:
            cur = conn.execute(
                "DELETE FROM keepalive_messages WHERE consumed = 1 AND created_at < ?",
                (cutoff,)
            )
            conn.commit()
        return cur.rowcount

    def get_recent_raw(self, hours: int = 48, level: str = "raw") -> list:
        """获取最近N小时内的指定level记忆"""
        from datetime import datetime, timedelta
        cutoff = (datetime.utcnow() - timedelta(hours=hours)).isoformat()
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT memory_id, text, timestamp, tag, tier, context, level "
                "FROM memories WHERE level = ? AND timestamp > ? "
                "ORDER BY timestamp ASC",
                (level, cutoff)
            ).fetchall()
        return [dict(r) for r in rows]

    def daily_recap(self, date_str: str) -> list:
        """获取指定 JST 日期的所有记忆。"""
        start_utc, end_utc = jst_day_bounds_utc(date_str)
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT memory_id, text, timestamp, tag, tier, emotion_score, level "
                "FROM memories WHERE timestamp >= ? AND timestamp < ? "
                "ORDER BY timestamp ASC",
                (start_utc, end_utc)
            ).fetchall()
        return [dict(r) for r in rows]

    def get_by_level(self, level: str) -> list:
        """获取指定level的所有记忆（cognition层用，量少）"""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT memory_id, text, timestamp, tag, tier, emotion_score, context, level "
                "FROM memories WHERE level = ? "
                "ORDER BY timestamp DESC",
                (level,)
            ).fetchall()
        return [dict(r) for r in rows]

    def calendar_density(self, start: str, end: str) -> list:
        """获取 JST 日期范围内每天的记忆数量，用于日历热力图。"""
        start_utc, end_utc = jst_range_bounds_utc(start, end)
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT date(timestamp, '+9 hours') as date, COUNT(*) as count "
                "FROM memories WHERE timestamp >= ? AND timestamp < ? "
                "GROUP BY date(timestamp, '+9 hours') ORDER BY date ASC",
                (start_utc, end_utc)
            ).fetchall()
        return [dict(r) for r in rows]
