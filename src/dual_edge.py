"""Phase-C dual edge storage adapter.

SQLite is authoritative. Kuzu is an eventually-consistent mirror drained from
the v2 outbox when AnchorDB owns the Kuzu process lock.
"""
from __future__ import annotations

import os
import sqlite3
import uuid
from datetime import datetime

NAMESPACE = uuid.uuid5(uuid.NAMESPACE_URL, "https://anchor-memory.example.invalid")
FLOW_ROLES = {"lateral", "temporal"}
SEMANTIC_ROLES = {"derived_from", "updates", "SUPPORTED_BY", "GROUNDED_IN", "EVOKES"}


def enabled() -> bool:
    return os.environ.get("ANCHOR_DUAL_EDGE", "on").strip().lower() in {"1", "on", "true", "yes"}


def edge_id(source_id: str, target_id: str, role: str) -> str:
    return str(uuid.uuid5(NAMESPACE, f"{source_id}\x1f{target_id}\x1f{role}"))


SCHEMA = r"""
CREATE TABLE IF NOT EXISTS flow_edges (
 source_id TEXT NOT NULL,target_id TEXT NOT NULL,weight REAL NOT NULL DEFAULT 1.0,
 conductance REAL NOT NULL DEFAULT 1.0,created TEXT NOT NULL,last_fired TEXT NOT NULL,
 provenance TEXT NOT NULL DEFAULT 'unknown',PRIMARY KEY(source_id,target_id),
 FOREIGN KEY(source_id) REFERENCES memories(memory_id) ON DELETE CASCADE,
 FOREIGN KEY(target_id) REFERENCES memories(memory_id) ON DELETE CASCADE);
CREATE INDEX IF NOT EXISTS idx_flow_edges_source ON flow_edges(source_id);
CREATE INDEX IF NOT EXISTS idx_flow_edges_target ON flow_edges(target_id);
CREATE INDEX IF NOT EXISTS idx_flow_edges_source_weight ON flow_edges(source_id,weight DESC);
CREATE TABLE IF NOT EXISTS semantic_edges (
 edge_id TEXT PRIMARY KEY,source_id TEXT NOT NULL,target_id TEXT NOT NULL,role TEXT NOT NULL,
 strength REAL NOT NULL DEFAULT 1.0,conductance REAL NOT NULL DEFAULT 0.0,
 confidence REAL NOT NULL DEFAULT 1.0,valid_from TEXT,valid_to TEXT,
 provenance TEXT NOT NULL DEFAULT 'manual',review_state TEXT NOT NULL DEFAULT 'approved',
 created_by TEXT NOT NULL DEFAULT 'agent',audit_note TEXT NOT NULL DEFAULT '',created TEXT NOT NULL,
 UNIQUE(source_id,target_id,role),
 FOREIGN KEY(source_id) REFERENCES memories(memory_id) ON DELETE CASCADE,
 FOREIGN KEY(target_id) REFERENCES memories(memory_id) ON DELETE CASCADE);
CREATE INDEX IF NOT EXISTS idx_semantic_edges_source ON semantic_edges(source_id);
CREATE INDEX IF NOT EXISTS idx_semantic_edges_target ON semantic_edges(target_id);
CREATE INDEX IF NOT EXISTS idx_semantic_edges_role ON semantic_edges(role);
CREATE INDEX IF NOT EXISTS idx_semantic_edges_review_state ON semantic_edges(review_state);
CREATE INDEX IF NOT EXISTS idx_semantic_edges_source_role_review ON semantic_edges(source_id,role,review_state);
CREATE INDEX IF NOT EXISTS idx_semantic_edges_target_role_review ON semantic_edges(target_id,role,review_state);
CREATE TABLE IF NOT EXISTS kuzu_edge_outbox_v2 (
 table_name TEXT NOT NULL DEFAULT 'edges',edge_key TEXT NOT NULL,source_id TEXT NOT NULL,
 target_id TEXT NOT NULL,op TEXT NOT NULL,weight REAL,conductance REAL,edge_type TEXT,
 edge_id TEXT,role TEXT,strength REAL,confidence REAL,valid_from TEXT,valid_to TEXT,
 provenance TEXT,review_state TEXT,created_by TEXT,audit_note TEXT,created TEXT,last_fired TEXT,
 revision INTEGER NOT NULL DEFAULT 1,PRIMARY KEY(table_name,edge_key));
CREATE TRIGGER IF NOT EXISTS trg_kuzu_flow_insert AFTER INSERT ON flow_edges BEGIN
 INSERT INTO kuzu_edge_outbox_v2(table_name,edge_key,source_id,target_id,op,weight,conductance,provenance,created,last_fired)
 VALUES('flow_edges',NEW.source_id||char(31)||NEW.target_id,NEW.source_id,NEW.target_id,'upsert',NEW.weight,NEW.conductance,NEW.provenance,NEW.created,NEW.last_fired)
 ON CONFLICT(table_name,edge_key) DO UPDATE SET op='upsert',weight=excluded.weight,conductance=excluded.conductance,provenance=excluded.provenance,created=excluded.created,last_fired=excluded.last_fired,revision=kuzu_edge_outbox_v2.revision+1; END;
CREATE TRIGGER IF NOT EXISTS trg_kuzu_flow_update AFTER UPDATE ON flow_edges BEGIN
 INSERT INTO kuzu_edge_outbox_v2(table_name,edge_key,source_id,target_id,op,weight,conductance,provenance,created,last_fired)
 VALUES('flow_edges',NEW.source_id||char(31)||NEW.target_id,NEW.source_id,NEW.target_id,'upsert',NEW.weight,NEW.conductance,NEW.provenance,NEW.created,NEW.last_fired)
 ON CONFLICT(table_name,edge_key) DO UPDATE SET op='upsert',weight=excluded.weight,conductance=excluded.conductance,provenance=excluded.provenance,created=excluded.created,last_fired=excluded.last_fired,revision=kuzu_edge_outbox_v2.revision+1; END;
CREATE TRIGGER IF NOT EXISTS trg_kuzu_flow_delete AFTER DELETE ON flow_edges BEGIN
 INSERT INTO kuzu_edge_outbox_v2(table_name,edge_key,source_id,target_id,op)
 VALUES('flow_edges',OLD.source_id||char(31)||OLD.target_id,OLD.source_id,OLD.target_id,'delete')
 ON CONFLICT(table_name,edge_key) DO UPDATE SET op='delete',revision=kuzu_edge_outbox_v2.revision+1; END;
CREATE TRIGGER IF NOT EXISTS trg_kuzu_semantic_insert AFTER INSERT ON semantic_edges BEGIN
 INSERT INTO kuzu_edge_outbox_v2(table_name,edge_key,source_id,target_id,op,edge_id,role,strength,conductance,confidence,valid_from,valid_to,provenance,review_state,created_by,audit_note,created)
 VALUES('semantic_edges',NEW.edge_id,NEW.source_id,NEW.target_id,'upsert',NEW.edge_id,NEW.role,NEW.strength,NEW.conductance,NEW.confidence,NEW.valid_from,NEW.valid_to,NEW.provenance,NEW.review_state,NEW.created_by,NEW.audit_note,NEW.created)
 ON CONFLICT(table_name,edge_key) DO UPDATE SET op='upsert',role=excluded.role,strength=excluded.strength,conductance=excluded.conductance,confidence=excluded.confidence,valid_from=excluded.valid_from,valid_to=excluded.valid_to,provenance=excluded.provenance,review_state=excluded.review_state,created_by=excluded.created_by,audit_note=excluded.audit_note,created=excluded.created,revision=kuzu_edge_outbox_v2.revision+1; END;
CREATE TRIGGER IF NOT EXISTS trg_kuzu_semantic_update AFTER UPDATE ON semantic_edges BEGIN
 INSERT INTO kuzu_edge_outbox_v2(table_name,edge_key,source_id,target_id,op,edge_id,role,strength,conductance,confidence,valid_from,valid_to,provenance,review_state,created_by,audit_note,created)
 VALUES('semantic_edges',NEW.edge_id,NEW.source_id,NEW.target_id,'upsert',NEW.edge_id,NEW.role,NEW.strength,NEW.conductance,NEW.confidence,NEW.valid_from,NEW.valid_to,NEW.provenance,NEW.review_state,NEW.created_by,NEW.audit_note,NEW.created)
 ON CONFLICT(table_name,edge_key) DO UPDATE SET op='upsert',role=excluded.role,strength=excluded.strength,conductance=excluded.conductance,confidence=excluded.confidence,valid_from=excluded.valid_from,valid_to=excluded.valid_to,provenance=excluded.provenance,review_state=excluded.review_state,created_by=excluded.created_by,audit_note=excluded.audit_note,created=excluded.created,revision=kuzu_edge_outbox_v2.revision+1; END;
CREATE TRIGGER IF NOT EXISTS trg_kuzu_semantic_delete AFTER DELETE ON semantic_edges BEGIN
 INSERT INTO kuzu_edge_outbox_v2(table_name,edge_key,source_id,target_id,op,edge_id,role)
 VALUES('semantic_edges',OLD.edge_id,OLD.source_id,OLD.target_id,'delete',OLD.edge_id,OLD.role)
 ON CONFLICT(table_name,edge_key) DO UPDATE SET op='delete',revision=kuzu_edge_outbox_v2.revision+1; END;
"""


def ensure_schema(db) -> None:
    with db._conn() as conn:
        cols = {r[1] for r in conn.execute("PRAGMA table_info(kuzu_edge_outbox)")}
        if cols and "table_name" not in cols:
            conn.execute("ALTER TABLE kuzu_edge_outbox ADD COLUMN table_name TEXT NOT NULL DEFAULT 'edges'")
        conn.executescript(SCHEMA)
        conn.commit()


def ensure_kuzu_schema(db) -> None:
    if not db._kuzu_conn:
        return
    db._kuzu_write("CREATE REL TABLE IF NOT EXISTS FlowEdge(FROM Memory TO Memory,weight DOUBLE,conductance DOUBLE,created STRING,last_fired STRING,provenance STRING)")
    db._kuzu_write("CREATE REL TABLE IF NOT EXISTS SemanticEdge(FROM Memory TO Memory,edge_id STRING,role STRING,strength DOUBLE,conductance DOUBLE,confidence DOUBLE,valid_from STRING,valid_to STRING,provenance STRING,review_state STRING,created_by STRING,audit_note STRING,created STRING)")


def _flow_params(row):
    return {k: row[k] for k in ("source_id", "target_id", "weight", "conductance",
                                "created", "last_fired", "provenance")}


def _semantic_params(row):
    d = {k: row[k] for k in ("edge_id", "source_id", "target_id", "role", "strength",
        "conductance", "confidence", "valid_from", "valid_to", "provenance", "review_state",
        "created_by", "audit_note", "created")}
    for k in ("valid_from", "valid_to"):
        if d[k] is None:
            d[k] = ""
    return d


def bootstrap_kuzu(db) -> None:
    """Idempotently seed new Kuzu relationship tables from SQLite after restart."""
    if not db._kuzu_conn:
        return
    ensure_kuzu_schema(db)
    with db._conn() as conn:
        nodes = [{"memory_id": x[0]} for x in conn.execute("SELECT memory_id FROM memories")]
        flow = [_flow_params(x) for x in conn.execute("SELECT * FROM flow_edges")]
        semantic = [_semantic_params(x) for x in conn.execute(
            "SELECT * FROM semantic_edges "
            "WHERE role!='belief_member' AND target_id NOT LIKE 'b-%'"
        )]
    # Tables are new in phase C. Rebuild only these tables; legacy EDGE is untouched.
    # Repair any pre-existing SQLite/Kuzu node lag first, otherwise MATCH would
    # silently omit relationships whose endpoint has not reached Kuzu yet.
    for start in range(0, len(nodes), 500):
        db._kuzu_write("UNWIND $rows AS row MERGE (m:Memory {memory_id:row.memory_id})",
                       {"rows": nodes[start:start+500]})
    db._kuzu_write("MATCH ()-[e:FlowEdge]->() DELETE e")
    db._kuzu_write("MATCH ()-[e:SemanticEdge]->() DELETE e")
    for start in range(0, len(flow), 500):
        db._kuzu_write("""UNWIND $rows AS row MATCH (a:Memory {memory_id:row.source_id}),(b:Memory {memory_id:row.target_id}) CREATE (a)-[:FlowEdge {weight:row.weight,conductance:row.conductance,created:row.created,last_fired:row.last_fired,provenance:row.provenance}]->(b)""", {"rows": flow[start:start+500]})
    for start in range(0, len(semantic), 500):
        db._kuzu_write("""UNWIND $rows AS row MATCH (a:Memory {memory_id:row.source_id}),(b:Memory {memory_id:row.target_id}) CREATE (a)-[:SemanticEdge {edge_id:row.edge_id,role:row.role,strength:row.strength,conductance:row.conductance,confidence:row.confidence,valid_from:row.valid_from,valid_to:row.valid_to,provenance:row.provenance,review_state:row.review_state,created_by:row.created_by,audit_note:row.audit_note,created:row.created}]->(b)""", {"rows": semantic[start:start+500]})
    with db._conn() as conn:
        conn.execute("DELETE FROM kuzu_edge_outbox_v2")
        conn.commit()


def drain(db) -> None:
    if not db._kuzu_conn:
        return
    with db._kuzu_outbox_lock:
        with db._conn() as conn:
            rows = conn.execute("SELECT * FROM kuzu_edge_outbox_v2 ORDER BY table_name,edge_key").fetchall()
        for row in rows:
            # Phase D keeps belief pseudo-nodes in SQLite only. Never mirror them
            # into Kuzu, where both SemanticEdge endpoints must be Memory nodes.
            if (row["table_name"] == "semantic_edges"
                    and str(row["target_id"] or "").startswith("b-")):
                with db._conn() as conn:
                    conn.execute(
                        "DELETE FROM kuzu_edge_outbox_v2 "
                        "WHERE table_name=? AND edge_key=? AND revision=?",
                        (row["table_name"], row["edge_key"], row["revision"]),
                    )
                    conn.commit()
                continue
            if row["table_name"] == "flow_edges":
                delete_params = {
                    "source_id": row["source_id"],
                    "target_id": row["target_id"],
                }
                db._kuzu_write("MATCH (a:Memory)-[e:FlowEdge]->(b:Memory) WHERE a.memory_id=$source_id AND b.memory_id=$target_id DELETE e", delete_params)
                if row["op"] != "delete":
                    flow_params = _flow_params(row)
                    for k in ("created", "last_fired", "provenance"):
                        if flow_params[k] is None:
                            flow_params[k] = ""
                    db._kuzu_write("MATCH (a:Memory {memory_id:$source_id}),(b:Memory {memory_id:$target_id}) CREATE (a)-[:FlowEdge {weight:$weight,conductance:$conductance,created:$created,last_fired:$last_fired,provenance:$provenance}]->(b)", flow_params)
            elif row["table_name"] == "semantic_edges":
                db._kuzu_write("MATCH ()-[e:SemanticEdge]->() WHERE e.edge_id=$edge_id DELETE e", {"edge_id": row["edge_id"]})
                if row["op"] != "delete":
                    semantic_params = _semantic_params(row)
                    for k in ("audit_note", "provenance", "review_state", "created_by", "created"):
                        if semantic_params[k] is None:
                            semantic_params[k] = ""
                    db._kuzu_write("MATCH (a:Memory {memory_id:$source_id}),(b:Memory {memory_id:$target_id}) CREATE (a)-[:SemanticEdge {edge_id:$edge_id,role:$role,strength:$strength,conductance:$conductance,confidence:$confidence,valid_from:$valid_from,valid_to:$valid_to,provenance:$provenance,review_state:$review_state,created_by:$created_by,audit_note:$audit_note,created:$created}]->(b)", semantic_params)
            with db._conn() as conn:
                conn.execute("DELETE FROM kuzu_edge_outbox_v2 WHERE table_name=? AND edge_key=? AND revision=?",
                             (row["table_name"], row["edge_key"], row["revision"]))
                conn.commit()


def _write_flow_conn(db, conn, source_id, target_id, weight, conductance, provenance,
                     mode="auto", now=None):
    if not source_id or not target_id or source_id == target_id:
        raise ValueError("flow edge requires two distinct memory ids")
    if mode not in {"auto", "manual"}:
        raise ValueError("mode must be auto or manual")
    if conn.execute("SELECT count(*) FROM memories WHERE memory_id IN (?,?)", (source_id,target_id)).fetchone()[0] != 2:
        raise ValueError("flow edge endpoint does not exist")
    now = now or datetime.utcnow().isoformat()
    weight = max(0.0, min(float(weight), db.MAX_EDGE_WEIGHT))
    conductance = max(0.0, min(float(conductance), 1.0))
    if mode == "manual":
        conn.execute("""INSERT INTO flow_edges VALUES(?,?,?,?,?,?,?) ON CONFLICT(source_id,target_id) DO UPDATE SET weight=MIN(flow_edges.weight+excluded.weight,?),conductance=excluded.conductance,last_fired=excluded.last_fired,provenance='manual'""",
                     (source_id,target_id,weight,conductance,now,now,provenance,db.MAX_EDGE_WEIGHT))
    else:
        conn.execute("""INSERT INTO flow_edges VALUES(?,?,?,?,?,?,?) ON CONFLICT(source_id,target_id) DO UPDATE SET weight=MAX(flow_edges.weight,excluded.weight),conductance=MAX(flow_edges.conductance,excluded.conductance),last_fired=CASE WHEN excluded.weight>flow_edges.weight THEN excluded.last_fired ELSE flow_edges.last_fired END,provenance=CASE WHEN flow_edges.provenance='manual' THEN 'manual' ELSE excluded.provenance END""",
                     (source_id,target_id,weight,conductance,now,now,provenance))


def _project_legacy(db, source_id, target_id, edge_type, weight, now=None):
    """Best-effort compatibility projection; never decides authoritative success."""
    now = now or datetime.utcnow().isoformat()
    warning = None
    try:
        with db._conn() as conn:
            row = conn.execute(
                "SELECT COALESCE(edge_type,'lateral') edge_type FROM edges "
                "WHERE source_id=? AND target_id=?", (source_id, target_id),
            ).fetchone()
            occupied = row["edge_type"] if row else None
            if occupied and occupied != edge_type:
                warning = f"legacy projection conflict: {occupied} retained, {edge_type} authoritative"
            else:
                conn.execute(
                    """INSERT INTO edges(source_id,target_id,weight,created,last_fired,edge_type)
                       VALUES(?,?,?,?,?,?)
                       ON CONFLICT(source_id,target_id) DO UPDATE SET
                         weight=MAX(edges.weight,excluded.weight),
                         last_fired=excluded.last_fired""",
                    (source_id, target_id, float(weight), now, now, edge_type),
                )
                conn.commit()
    except Exception as exc:
        warning = f"legacy projection failed: {type(exc).__name__}"
    if warning:
        try:
            db.log_event(source_id, "legacy_projection_warning", warning[:500])
        except Exception:
            pass
    return warning


def write_flow(db, source_id, target_id, weight=1.0, conductance=1.0,
               provenance="unknown", mode="auto"):
    with db._conn() as conn:
        _write_flow_conn(db, conn, source_id, target_id, weight, conductance, provenance, mode)
        conn.commit()
    warning = _project_legacy(db, source_id, target_id, "lateral", weight)
    warnings = [warning] if warning else []
    try:
        drain(db)
    except Exception as exc:
        warnings.append(f"Kuzu flow projection pending: {type(exc).__name__}")
    return {"ok": True, "source_id": source_id, "target_id": target_id,
            "edge_family": "flow", "warnings": warnings}


def write_flow_pair(db, source_id, target_id, weight=1.0, conductance=1.0,
                    provenance="manual", mode="manual"):
    now = datetime.utcnow().isoformat()
    with db._conn() as conn:
        _write_flow_conn(db, conn, source_id, target_id, weight, conductance, provenance, mode, now)
        _write_flow_conn(db, conn, target_id, source_id, weight, conductance, provenance, mode, now)
        conn.commit()
    warnings = [x for x in (
        _project_legacy(db, source_id, target_id, "lateral", weight, now),
        _project_legacy(db, target_id, source_id, "lateral", weight, now),
    ) if x]
    try:
        drain(db)
    except Exception as exc:
        warnings.append(f"Kuzu flow projection pending: {type(exc).__name__}")
    return {"ok": True, "rows": 2, "edge_family": "flow", "warnings": warnings}


def write_semantic(db, source_id, target_id, role, strength=1.0, conductance=0.0,
                   confidence=1.0, provenance="manual", review_state="approved",
                   created_by="agent", audit_note="", valid_from=None, valid_to=None):
    if role not in SEMANTIC_ROLES:
        raise ValueError("unknown semantic role")
    if review_state not in {"auto", "approved"}:
        raise ValueError("semantic_edges stores only auto/approved")
    db.validate_typed_edge(source_id, target_id, role)
    sid, now = edge_id(source_id,target_id,role), datetime.utcnow().isoformat()
    with db._conn() as conn:
        cur = conn.execute("""INSERT INTO semantic_edges(edge_id,source_id,target_id,role,strength,conductance,confidence,valid_from,valid_to,provenance,review_state,created_by,audit_note,created) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?) ON CONFLICT(source_id,target_id,role) DO NOTHING""",
            (sid,source_id,target_id,role,float(strength),float(conductance),max(0,min(float(confidence),1)),valid_from,valid_to,provenance,review_state,created_by,(audit_note or "")[:500],now))
        created = cur.rowcount == 1
        if created and role == "updates":
            db._apply_updates_activation_bias_conn(conn, source_id, target_id, now)
        conn.commit()
    warning = _project_legacy(db, source_id, target_id, role, strength)
    warnings = [warning] if warning else []
    try:
        drain(db)
    except Exception as exc:
        warnings.append(f"Kuzu semantic projection pending: {type(exc).__name__}")
    return {"ok": True, "created": created, "edge_id": sid, "source_id": source_id,
            "target_id": target_id, "role": role,
            "warnings": warnings}


def write_belief_member(db, source_id, belief_id, confidence=1.0,
                        audit_note=""):
    """Retired: Belief now has an independent node label and CONSTELLATES edge."""
    raise ValueError("belief_member retired; use belief_edit map_cognition")

    """Legacy phase-D implementation retained below for rollback archaeology.

    belief_id is deliberately not a Memory row in phase D. This narrow writer
    bypasses only semantic_edges' target FK, then removes the trigger-created
    Kuzu outbox row in the same transaction.
    """
    source_id = (source_id or "").strip()
    belief_id = (belief_id or "").strip()
    if not (belief_id.startswith("b-") and belief_id[2:].isdigit()):
        raise ValueError("belief_member target must be a b-NNNN belief id")
    if not source_id or source_id == belief_id:
        raise ValueError("belief_member requires distinct endpoints")
    sid, now = edge_id(source_id, belief_id, "belief_member"), datetime.utcnow().isoformat()
    conn = sqlite3.connect(db.db_path, timeout=5.0)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA foreign_keys=OFF")
        conn.execute("PRAGMA busy_timeout=5000")
        source = conn.execute(
            "SELECT memory_id,COALESCE(level,'raw') level,"
            "COALESCE(collection,'') collection FROM memories WHERE memory_id=?",
            (source_id,),
        ).fetchone()
        if not source or source["level"] != "cognition" or source["collection"] == "wenku":
            raise ValueError("belief_member source must be a non-wenku cognition memory")
        cur = conn.execute(
            """INSERT INTO semantic_edges(
                edge_id,source_id,target_id,role,strength,conductance,confidence,
                valid_from,valid_to,provenance,review_state,created_by,audit_note,created
            ) VALUES(?,?,?,?,1.0,0.0,?,NULL,NULL,'belief_mapping','approved',
                     'agent',?,?)
            ON CONFLICT(source_id,target_id,role) DO NOTHING""",
            (sid, source_id, belief_id, "belief_member",
             max(0.0, min(float(confidence), 1.0)), (audit_note or "")[:500], now),
        )
        created = cur.rowcount == 1
        conn.execute(
            "DELETE FROM kuzu_edge_outbox_v2 "
            "WHERE table_name='semantic_edges' AND edge_key=?", (sid,)
        )
        conn.commit()
    finally:
        conn.close()
    db.log_event(
        source_id, "belief_member_written",
        f"to={belief_id} confidence={float(confidence):.6f} note={(audit_note or '')[:240]}",
    )
    return {"ok": True, "created": created, "edge_id": sid,
            "source_id": source_id, "target_id": belief_id,
            "role": "belief_member", "sqlite_only": True}


def flow_neighbors(db, memory_id, direction="outgoing", min_weight=0.0, limit=20):
    if direction not in {"outgoing","incoming","both"}: raise ValueError("invalid direction")
    clauses=[]; params=[]
    if direction in {"outgoing","both"}: clauses.append("source_id=?"); params.append(memory_id)
    if direction in {"incoming","both"}: clauses.append("target_id=?"); params.append(memory_id)
    with db._conn() as conn:
        rows=conn.execute("SELECT * FROM flow_edges WHERE ("+" OR ".join(clauses)+") AND weight>=? ORDER BY weight DESC,source_id,target_id LIMIT ?",params+[float(min_weight),int(limit)]).fetchall()
    out=[]
    for r in rows:
        x=dict(r); outgoing=r["source_id"]==memory_id
        x.update(memory_id=r["target_id"] if outgoing else r["source_id"],direction="outgoing" if outgoing else "incoming",edge_family="flow"); out.append(x)
    return out


def semantic_neighbors(db, memory_id, direction="outgoing", roles=None,
                       review_state="approved", limit=20):
    if direction not in {"outgoing","incoming","both"}: raise ValueError("invalid direction")
    clauses=[]; params=[]
    if direction in {"outgoing","both"}: clauses.append("source_id=?"); params.append(memory_id)
    if direction in {"incoming","both"}: clauses.append("target_id=?"); params.append(memory_id)
    extra=""
    if roles: extra=" AND role IN ("+",".join("?" for _ in roles)+")"; params.extend(roles)
    if review_state is not None: extra+=" AND review_state=?"; params.append(review_state)
    with db._conn() as conn:
        rows=conn.execute("SELECT * FROM semantic_edges WHERE ("+" OR ".join(clauses)+")"+extra+" ORDER BY strength DESC,created DESC LIMIT ?",params+[int(limit)]).fetchall()
    out=[]
    for r in rows:
        x=dict(r); outgoing=r["source_id"]==memory_id
        x.update(memory_id=r["target_id"] if outgoing else r["source_id"],direction="outgoing" if outgoing else "incoming",edge_family="semantic"); out.append(x)
    return out


def migrate_legacy(db) -> dict:
    """Idempotent SQLite migration. Kuzu is bootstrapped by the owning process."""
    ensure_schema(db)
    with db._conn() as conn:
        rows=conn.execute("SELECT source_id,target_id,COALESCE(weight,1.0) weight,created,last_fired,COALESCE(edge_type,'lateral') edge_type FROM edges").fetchall()
        for r in rows:
            kind=r["edge_type"]
            if kind in {"lateral","backfill","temporal"}:
                provenance={"lateral":"legacy","backfill":"knn_legacy","temporal":"temporal"}[kind]
                conn.execute("INSERT OR IGNORE INTO flow_edges VALUES(?,?,?,?,?,?,?)",(r["source_id"],r["target_id"],r["weight"],1.0,r["created"],r["last_fired"],provenance))
            elif kind in {"derived_from","updates","SUPPORTED_BY","GROUNDED_IN","EVOKES"}:
                provenance="structured" if kind=="derived_from" else "manual"
                conn.execute("""INSERT OR IGNORE INTO semantic_edges(edge_id,source_id,target_id,role,strength,conductance,confidence,valid_from,valid_to,provenance,review_state,created_by,audit_note,created) VALUES(?,?,?,?,?,0,1,NULL,NULL,?,'approved','agent',?,?)""",
                    (edge_id(r["source_id"],r["target_id"],kind),r["source_id"],r["target_id"],kind,r["weight"],provenance,"phase_c_migration_from_edges:"+kind,r["created"]))
            else: raise RuntimeError("unmapped edge type: "+kind)
        conn.commit()
        source=len(rows); flow=conn.execute("SELECT count(*) FROM flow_edges").fetchone()[0]; semantic=conn.execute("SELECT count(*) FROM semantic_edges").fetchone()[0]
        missing_flow = conn.execute("""SELECT count(*) FROM edges e WHERE e.edge_type IN ('lateral','backfill','temporal') AND NOT EXISTS (SELECT 1 FROM flow_edges f WHERE f.source_id=e.source_id AND f.target_id=e.target_id)""").fetchone()[0]
        missing_semantic = conn.execute("""SELECT count(*) FROM edges e WHERE e.edge_type IN ('derived_from','updates','SUPPORTED_BY','GROUNDED_IN','EVOKES') AND NOT EXISTS (SELECT 1 FROM semantic_edges s WHERE s.source_id=e.source_id AND s.target_id=e.target_id AND s.role=e.edge_type)""").fetchone()[0]
        # Running anchor-sse owns Kuzu until the approved restart; bootstrap_kuzu
        # will rebuild both new relation tables then. Do not leave misleading pending rows.
        conn.execute("DELETE FROM kuzu_edge_outbox_v2")
        conn.commit()
    if missing_flow or missing_semantic:
        raise RuntimeError(f"migration incomplete: missing_flow={missing_flow} missing_semantic={missing_semantic}")
    return {"source_edges":source,"flow_edges":flow,"semantic_edges":semantic,
            "legacy_rows_missing": 0, "conserved": source == flow + semantic}
