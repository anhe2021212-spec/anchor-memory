"""Belief graph v2: SQLite authority with a Kuzu graph mirror.

Belief and BeliefCase are deliberately not Memory nodes.  Existing Anchor
activation and Recall V2 only traverse Memory flow_edges, so these graph
relations are structural until a dedicated activation design is approved.
"""
from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime


NAMESPACE = uuid.uuid5(uuid.NAMESPACE_URL, "https://anchor-memory.example.invalid")
CASE_KINDS = {"support": "SUPPORTS", "contradiction": "CONTRADICTS", "boundary": "BOUNDS"}


def _id(kind: str, *parts: str) -> str:
    return str(uuid.uuid5(NAMESPACE, "\x1f".join((kind,) + tuple(str(x) for x in parts))))


SCHEMA = r"""
CREATE TABLE IF NOT EXISTS belief_nodes (
 belief_id TEXT PRIMARY KEY, statement TEXT NOT NULL, kind TEXT NOT NULL DEFAULT 'propositional',
 status TEXT NOT NULL DEFAULT 'candidate', pinned INTEGER NOT NULL DEFAULT 0,
 origin TEXT NOT NULL DEFAULT '', activation_cues_json TEXT NOT NULL DEFAULT '[]',
 tensions_json TEXT NOT NULL DEFAULT '[]', notes_json TEXT NOT NULL DEFAULT '[]',
 created_at TEXT NOT NULL, updated_at TEXT NOT NULL, last_touched TEXT
);
CREATE INDEX IF NOT EXISTS idx_belief_nodes_status ON belief_nodes(status);

CREATE TABLE IF NOT EXISTS belief_cases (
 case_id TEXT PRIMARY KEY, belief_id TEXT NOT NULL, case_kind TEXT NOT NULL,
 memory_id TEXT, inline_text TEXT NOT NULL DEFAULT '', weight_note TEXT NOT NULL,
 occurred_at TEXT, added TEXT NOT NULL, emotion_score REAL NOT NULL DEFAULT 0.5,
 created_by TEXT NOT NULL DEFAULT 'heng',
 FOREIGN KEY(belief_id) REFERENCES belief_nodes(belief_id) ON DELETE CASCADE,
 FOREIGN KEY(memory_id) REFERENCES memories(memory_id) ON DELETE RESTRICT,
 CHECK(case_kind IN ('support','contradiction','boundary')),
 CHECK((memory_id IS NOT NULL AND inline_text='') OR (memory_id IS NULL AND length(trim(inline_text))>0))
);
CREATE INDEX IF NOT EXISTS idx_belief_cases_belief ON belief_cases(belief_id,case_kind);
CREATE INDEX IF NOT EXISTS idx_belief_cases_memory ON belief_cases(memory_id);

CREATE TABLE IF NOT EXISTS belief_constellations (
 cognition_id TEXT NOT NULL, belief_id TEXT NOT NULL, conductance REAL NOT NULL DEFAULT 0.0,
 created TEXT NOT NULL, created_by TEXT NOT NULL DEFAULT 'heng',
 PRIMARY KEY(cognition_id,belief_id),
 FOREIGN KEY(cognition_id) REFERENCES memories(memory_id) ON DELETE RESTRICT,
 FOREIGN KEY(belief_id) REFERENCES belief_nodes(belief_id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_belief_constellations_belief ON belief_constellations(belief_id);

CREATE TABLE IF NOT EXISTS kuzu_belief_outbox (
 entity_type TEXT NOT NULL, entity_key TEXT NOT NULL, op TEXT NOT NULL,
 revision INTEGER NOT NULL DEFAULT 1, PRIMARY KEY(entity_type,entity_key)
);

CREATE TRIGGER IF NOT EXISTS trg_kuzu_belief_insert AFTER INSERT ON belief_nodes BEGIN
 INSERT INTO kuzu_belief_outbox VALUES('belief',NEW.belief_id,'upsert',1)
 ON CONFLICT(entity_type,entity_key) DO UPDATE SET op='upsert',revision=kuzu_belief_outbox.revision+1; END;
CREATE TRIGGER IF NOT EXISTS trg_kuzu_belief_update AFTER UPDATE ON belief_nodes BEGIN
 INSERT INTO kuzu_belief_outbox VALUES('belief',NEW.belief_id,'upsert',1)
 ON CONFLICT(entity_type,entity_key) DO UPDATE SET op='upsert',revision=kuzu_belief_outbox.revision+1; END;
CREATE TRIGGER IF NOT EXISTS trg_kuzu_belief_delete AFTER DELETE ON belief_nodes BEGIN
 INSERT INTO kuzu_belief_outbox VALUES('belief',OLD.belief_id,'delete',1)
 ON CONFLICT(entity_type,entity_key) DO UPDATE SET op='delete',revision=kuzu_belief_outbox.revision+1; END;

CREATE TRIGGER IF NOT EXISTS trg_kuzu_belief_case_insert AFTER INSERT ON belief_cases BEGIN
 INSERT INTO kuzu_belief_outbox VALUES('case',NEW.case_id,'upsert',1)
 ON CONFLICT(entity_type,entity_key) DO UPDATE SET op='upsert',revision=kuzu_belief_outbox.revision+1; END;
CREATE TRIGGER IF NOT EXISTS trg_kuzu_belief_case_update AFTER UPDATE ON belief_cases BEGIN
 INSERT INTO kuzu_belief_outbox VALUES('case',NEW.case_id,'upsert',1)
 ON CONFLICT(entity_type,entity_key) DO UPDATE SET op='upsert',revision=kuzu_belief_outbox.revision+1; END;
CREATE TRIGGER IF NOT EXISTS trg_kuzu_belief_case_delete AFTER DELETE ON belief_cases BEGIN
 INSERT INTO kuzu_belief_outbox VALUES('case',OLD.case_id,'delete',1)
 ON CONFLICT(entity_type,entity_key) DO UPDATE SET op='delete',revision=kuzu_belief_outbox.revision+1; END;

CREATE TRIGGER IF NOT EXISTS trg_kuzu_constellation_insert AFTER INSERT ON belief_constellations BEGIN
 INSERT INTO kuzu_belief_outbox VALUES('constellates',NEW.cognition_id||char(31)||NEW.belief_id,'upsert',1)
 ON CONFLICT(entity_type,entity_key) DO UPDATE SET op='upsert',revision=kuzu_belief_outbox.revision+1; END;
CREATE TRIGGER IF NOT EXISTS trg_kuzu_constellation_delete AFTER DELETE ON belief_constellations BEGIN
 INSERT INTO kuzu_belief_outbox VALUES('constellates',OLD.cognition_id||char(31)||OLD.belief_id,'delete',1)
 ON CONFLICT(entity_type,entity_key) DO UPDATE SET op='delete',revision=kuzu_belief_outbox.revision+1; END;
"""


def ensure_schema(db) -> None:
    with db._conn() as conn:
        conn.executescript(SCHEMA)
        conn.commit()


def ensure_kuzu_schema(db) -> None:
    if not db._kuzu_conn:
        return
    db._kuzu_write("CREATE NODE TABLE IF NOT EXISTS Belief(belief_id STRING,statement STRING,kind STRING,status STRING,pinned BOOLEAN,origin STRING,activation_cues STRING,tensions STRING,notes STRING,created_at STRING,updated_at STRING,last_touched STRING,PRIMARY KEY(belief_id))")
    db._kuzu_write("CREATE NODE TABLE IF NOT EXISTS BeliefCase(case_id STRING,memory_id STRING,inline_text STRING,weight_note STRING,occurred_at STRING,added STRING,emotion_score DOUBLE,created_by STRING,PRIMARY KEY(case_id))")
    db._kuzu_write("CREATE REL TABLE IF NOT EXISTS CONSTELLATES(FROM Memory TO Belief,edge_id STRING,conductance DOUBLE,created STRING,created_by STRING)")
    db._kuzu_write("CREATE REL TABLE IF NOT EXISTS REFERENCES(FROM Memory TO BeliefCase,edge_id STRING,conductance DOUBLE,created STRING)")
    for role in CASE_KINDS.values():
        db._kuzu_write(f"CREATE REL TABLE IF NOT EXISTS {role}(FROM BeliefCase TO Belief,edge_id STRING,conductance DOUBLE,created STRING)")


def _belief_row(row) -> dict:
    return {
        "belief_id": row["belief_id"], "statement": row["statement"], "kind": row["kind"],
        "status": row["status"], "pinned": bool(row["pinned"]), "origin": row["origin"] or "",
        "activation_cues": row["activation_cues_json"] or "[]", "tensions": row["tensions_json"] or "[]",
        "notes": row["notes_json"] or "[]", "created_at": row["created_at"] or "",
        "updated_at": row["updated_at"] or "", "last_touched": row["last_touched"] or "",
    }


def _case_row(row) -> dict:
    return {
        "case_id": row["case_id"], "memory_id": row["memory_id"] or "",
        "inline_text": row["inline_text"] or "", "weight_note": row["weight_note"] or "",
        "occurred_at": row["occurred_at"] or "", "added": row["added"] or "",
        "emotion_score": float(row["emotion_score"]), "created_by": row["created_by"] or "heng",
        "belief_id": row["belief_id"], "case_kind": row["case_kind"],
    }


def _case_node_params(row: dict) -> dict:
    return {k: row[k] for k in (
        "case_id", "memory_id", "inline_text", "weight_note", "occurred_at",
        "added", "emotion_score", "created_by",
    )}


def bootstrap_kuzu(db) -> None:
    if not db._kuzu_conn:
        return
    ensure_kuzu_schema(db)
    with db._conn() as conn:
        beliefs = [_belief_row(x) for x in conn.execute("SELECT * FROM belief_nodes")]
        cases = [_case_row(x) for x in conn.execute("SELECT * FROM belief_cases")]
        constellations = [dict(x) for x in conn.execute("SELECT * FROM belief_constellations")]
    db._kuzu_write("MATCH ()-[e:CONSTELLATES]->() DELETE e")
    db._kuzu_write("MATCH ()-[e:REFERENCES]->() DELETE e")
    for role in CASE_KINDS.values():
        db._kuzu_write(f"MATCH ()-[e:{role}]->() DELETE e")
    db._kuzu_write("MATCH (n:BeliefCase) DELETE n")
    db._kuzu_write("MATCH (n:Belief) DELETE n")
    for row in beliefs:
        db._kuzu_write("CREATE (b:Belief {belief_id:$belief_id,statement:$statement,kind:$kind,status:$status,pinned:$pinned,origin:$origin,activation_cues:$activation_cues,tensions:$tensions,notes:$notes,created_at:$created_at,updated_at:$updated_at,last_touched:$last_touched})", row)
    for row in cases:
        db._kuzu_write("CREATE (c:BeliefCase {case_id:$case_id,memory_id:$memory_id,inline_text:$inline_text,weight_note:$weight_note,occurred_at:$occurred_at,added:$added,emotion_score:$emotion_score,created_by:$created_by})", _case_node_params(row))
        if row["memory_id"]:
            db._kuzu_write("MATCH (m:Memory {memory_id:$memory_id}),(c:BeliefCase {case_id:$case_id}) CREATE (m)-[:REFERENCES {edge_id:$edge_id,conductance:0.0,created:$created}]->(c)", {
                "memory_id": row["memory_id"], "case_id": row["case_id"],
                "edge_id": _id("REFERENCES", row["memory_id"], row["case_id"]), "created": row["added"],
            })
        role = CASE_KINDS[row["case_kind"]]
        db._kuzu_write(f"MATCH (c:BeliefCase {{case_id:$case_id}}),(b:Belief {{belief_id:$belief_id}}) CREATE (c)-[:{role} {{edge_id:$edge_id,conductance:0.0,created:$created}}]->(b)", {
            "case_id": row["case_id"], "belief_id": row["belief_id"],
            "edge_id": _id(role, row["case_id"], row["belief_id"]), "created": row["added"],
        })
    for row in constellations:
        db._kuzu_write("MATCH (m:Memory {memory_id:$cognition_id}),(b:Belief {belief_id:$belief_id}) CREATE (m)-[:CONSTELLATES {edge_id:$edge_id,conductance:$conductance,created:$created,created_by:$created_by}]->(b)", {
            **row, "edge_id": _id("CONSTELLATES", row["cognition_id"], row["belief_id"]),
        })
    with db._conn() as conn:
        conn.execute("DELETE FROM kuzu_belief_outbox")
        conn.commit()


def drain(db) -> None:
    if not db._kuzu_conn:
        return
    with db._kuzu_outbox_lock:
        with db._conn() as conn:
            rows = list(conn.execute("SELECT * FROM kuzu_belief_outbox ORDER BY entity_type,entity_key"))
        for item in rows:
            kind, key, op, revision = item["entity_type"], item["entity_key"], item["op"], item["revision"]
            if kind == "belief":
                if op == "delete":
                    db._kuzu_write("MATCH (b:Belief {belief_id:$id}) DETACH DELETE b", {"id": key})
                else:
                    with db._conn() as conn:
                        row = conn.execute("SELECT * FROM belief_nodes WHERE belief_id=?", (key,)).fetchone()
                    if row:
                        db._kuzu_write("MERGE (b:Belief {belief_id:$belief_id}) SET b.statement=$statement,b.kind=$kind,b.status=$status,b.pinned=$pinned,b.origin=$origin,b.activation_cues=$activation_cues,b.tensions=$tensions,b.notes=$notes,b.created_at=$created_at,b.updated_at=$updated_at,b.last_touched=$last_touched", _belief_row(row))
            elif kind == "case":
                db._kuzu_write("MATCH (c:BeliefCase {case_id:$id}) DETACH DELETE c", {"id": key})
                if op != "delete":
                    with db._conn() as conn:
                        row = conn.execute("SELECT * FROM belief_cases WHERE case_id=?", (key,)).fetchone()
                    if row:
                        row = _case_row(row)
                        db._kuzu_write("CREATE (c:BeliefCase {case_id:$case_id,memory_id:$memory_id,inline_text:$inline_text,weight_note:$weight_note,occurred_at:$occurred_at,added:$added,emotion_score:$emotion_score,created_by:$created_by})", _case_node_params(row))
                        if row["memory_id"]:
                            db._kuzu_write("MATCH (m:Memory {memory_id:$memory_id}),(c:BeliefCase {case_id:$case_id}) CREATE (m)-[:REFERENCES {edge_id:$edge_id,conductance:0.0,created:$created}]->(c)", {"memory_id": row["memory_id"], "case_id": key, "edge_id": _id("REFERENCES", row["memory_id"], key), "created": row["added"]})
                        role = CASE_KINDS[row["case_kind"]]
                        db._kuzu_write(f"MATCH (c:BeliefCase {{case_id:$case_id}}),(b:Belief {{belief_id:$belief_id}}) CREATE (c)-[:{role} {{edge_id:$edge_id,conductance:0.0,created:$created}}]->(b)", {"case_id": key, "belief_id": row["belief_id"], "edge_id": _id(role, key, row["belief_id"]), "created": row["added"]})
            elif kind == "constellates":
                cognition_id, belief_id = key.split("\x1f", 1)
                db._kuzu_write("MATCH (m:Memory {memory_id:$cognition_id})-[e:CONSTELLATES]->(b:Belief {belief_id:$belief_id}) DELETE e", {"cognition_id": cognition_id, "belief_id": belief_id})
                if op != "delete":
                    with db._conn() as conn:
                        row = conn.execute("SELECT * FROM belief_constellations WHERE cognition_id=? AND belief_id=?", (cognition_id, belief_id)).fetchone()
                    if row:
                        params = dict(row); params["edge_id"] = _id("CONSTELLATES", cognition_id, belief_id)
                        db._kuzu_write("MATCH (m:Memory {memory_id:$cognition_id}),(b:Belief {belief_id:$belief_id}) CREATE (m)-[:CONSTELLATES {edge_id:$edge_id,conductance:$conductance,created:$created,created_by:$created_by}]->(b)", params)
            with db._conn() as conn:
                conn.execute("DELETE FROM kuzu_belief_outbox WHERE entity_type=? AND entity_key=? AND revision=?", (kind, key, revision))
                conn.commit()


def _stable_case_id(belief_id: str, case_kind: str, case: dict, ordinal: int) -> str:
    if case.get("case_id"):
        return str(case["case_id"])
    source = case.get("id") or case.get("memory_id") or case.get("inline_text") or ""
    return "bc-" + _id("case", belief_id, case_kind, source, case.get("added", ""), case.get("weight_note", ""), ordinal)[:18]


def import_legacy(db, data: dict) -> dict:
    """Import legacy JSON once. Invalid cognition mappings are reported, never guessed."""
    ensure_schema(db)
    with db._conn() as conn:
        if conn.execute("SELECT count(*) FROM belief_nodes").fetchone()[0]:
            return {"imported": False, "reason": "belief_nodes not empty"}
        now = datetime.now().astimezone().date().isoformat()
        invalid = []
        for belief in data.get("beliefs", []):
            bid = str(belief["id"])
            conn.execute("INSERT INTO belief_nodes VALUES(?,?,?,?,?,?,?,?,?,?,?,?)", (
                bid, belief.get("statement", ""), belief.get("kind", "propositional"),
                belief.get("status", "candidate"), int(bool(belief.get("pinned"))), belief.get("origin", ""),
                json.dumps(belief.get("activation_cues", []), ensure_ascii=False),
                json.dumps(belief.get("tensions", []), ensure_ascii=False),
                json.dumps(belief.get("notes", []), ensure_ascii=False),
                belief.get("created_at") or now, belief.get("updated_at") or now, belief.get("last_touched"),
            ))
            for case_kind, key in (("support", "support_cases"), ("contradiction", "contradiction_cases"), ("boundary", "boundary_cases")):
                for ordinal, case in enumerate(belief.get(key, []) or []):
                    memory_id = case.get("id") or case.get("memory_id")
                    if memory_id and not conn.execute("SELECT 1 FROM memories WHERE memory_id=?", (memory_id,)).fetchone():
                        invalid.append({"type": "case_memory", "belief_id": bid, "memory_id": memory_id})
                        continue
                    conn.execute("INSERT INTO belief_cases VALUES(?,?,?,?,?,?,?,?,?,?)", (
                        _stable_case_id(bid, case_kind, case, ordinal), bid, case_kind, memory_id,
                        case.get("inline_text", "") if not memory_id else "", case.get("weight_note", ""),
                        case.get("occurred_at"), case.get("added") or now,
                        float(case.get("emotion_score", 0.5)), case.get("created_by", "heng"),
                    ))
            for cognition_id in dict.fromkeys(str(x) for x in belief.get("cognition_ids", []) if x):
                row = conn.execute("SELECT level,collection FROM memories WHERE memory_id=?", (cognition_id,)).fetchone()
                if not row or row["level"] != "cognition" or (row["collection"] or "") == "wenku":
                    invalid.append({"type": "constellation", "belief_id": bid, "memory_id": cognition_id})
                    continue
                conn.execute("INSERT INTO belief_constellations VALUES(?,?,0.0,?,?)", (cognition_id, bid, now, "migration"))
        conn.commit()
        counts = {
            "beliefs": conn.execute("SELECT count(*) FROM belief_nodes").fetchone()[0],
            "cases": conn.execute("SELECT count(*) FROM belief_cases").fetchone()[0],
            "constellations": conn.execute("SELECT count(*) FROM belief_constellations").fetchone()[0],
        }
    return {"imported": True, **counts, "invalid": invalid}


def export_data(db, params: dict | None = None) -> dict:
    with db._conn() as conn:
        beliefs = []
        for row in conn.execute("SELECT * FROM belief_nodes ORDER BY belief_id"):
            b = {
                "id": row["belief_id"], "statement": row["statement"], "kind": row["kind"],
                "pinned": bool(row["pinned"]), "status": row["status"], "origin": row["origin"],
                "activation_cues": json.loads(row["activation_cues_json"] or "[]"),
                "cognition_ids": [x[0] for x in conn.execute("SELECT cognition_id FROM belief_constellations WHERE belief_id=? ORDER BY cognition_id", (row["belief_id"],))],
                "tensions": json.loads(row["tensions_json"] or "[]"), "support_cases": [],
                "contradiction_cases": [], "boundary_cases": [],
                "notes": json.loads(row["notes_json"] or "[]"), "created_at": row["created_at"],
                "updated_at": row["updated_at"],
            }
            if row["last_touched"]:
                b["last_touched"] = row["last_touched"]
            for case in conn.execute("SELECT * FROM belief_cases WHERE belief_id=? ORDER BY added,case_id", (row["belief_id"],)):
                item = {"case_id": case["case_id"], "weight_note": case["weight_note"], "added": case["added"]}
                if case["memory_id"]:
                    item["id"] = case["memory_id"]
                else:
                    item.update({"inline_text": case["inline_text"], "occurred_at": case["occurred_at"], "emotion_score": case["emotion_score"]})
                b[{"support": "support_cases", "contradiction": "contradiction_cases", "boundary": "boundary_cases"}[case["case_kind"]]].append(item)
            beliefs.append(b)
    return {"version": "2.0", "created": datetime.now().astimezone().date().isoformat(), "params": params or {}, "beliefs": beliefs}


def replace_data(db, data: dict) -> None:
    """Apply a compatibility document by diff, preserving stable graph identities."""
    ensure_schema(db)
    desired_beliefs = {str(b["id"]): b for b in data.get("beliefs", [])}
    desired_cases = {}
    desired_constellations = set()
    for bid, b in desired_beliefs.items():
        for kind, key in (("support", "support_cases"), ("contradiction", "contradiction_cases"), ("boundary", "boundary_cases")):
            for ordinal, case in enumerate(b.get(key, []) or []):
                cid = _stable_case_id(bid, kind, case, ordinal)
                desired_cases[cid] = (bid, kind, case)
        desired_constellations.update((str(mid), bid) for mid in b.get("cognition_ids", []) if mid)
    with db._conn() as conn:
        for cid in set(x[0] for x in conn.execute("SELECT case_id FROM belief_cases")) - set(desired_cases):
            conn.execute("DELETE FROM belief_cases WHERE case_id=?", (cid,))
        for pair in set(map(tuple, conn.execute("SELECT cognition_id,belief_id FROM belief_constellations"))) - desired_constellations:
            conn.execute("DELETE FROM belief_constellations WHERE cognition_id=? AND belief_id=?", pair)
        for bid in set(x[0] for x in conn.execute("SELECT belief_id FROM belief_nodes")) - set(desired_beliefs):
            conn.execute("DELETE FROM belief_nodes WHERE belief_id=?", (bid,))
        for bid, b in desired_beliefs.items():
            values = (b.get("statement", ""), b.get("kind", "propositional"), b.get("status", "candidate"), int(bool(b.get("pinned"))), b.get("origin", ""), json.dumps(b.get("activation_cues", []), ensure_ascii=False), json.dumps(b.get("tensions", []), ensure_ascii=False), json.dumps(b.get("notes", []), ensure_ascii=False), b.get("created_at", ""), b.get("updated_at", ""), b.get("last_touched"))
            old = conn.execute("SELECT statement,kind,status,pinned,origin,activation_cues_json,tensions_json,notes_json,created_at,updated_at,last_touched FROM belief_nodes WHERE belief_id=?", (bid,)).fetchone()
            if old is None:
                conn.execute("INSERT INTO belief_nodes VALUES(?,?,?,?,?,?,?,?,?,?,?,?)", (bid, *values))
            elif tuple(old) != values:
                conn.execute("UPDATE belief_nodes SET statement=?,kind=?,status=?,pinned=?,origin=?,activation_cues_json=?,tensions_json=?,notes_json=?,created_at=?,updated_at=?,last_touched=? WHERE belief_id=?", (*values, bid))
        for cid, (bid, kind, case) in desired_cases.items():
            memory_id = case.get("id") or case.get("memory_id")
            inline_text = "" if memory_id else str(case.get("inline_text") or "").strip()
            if not memory_id and not inline_text:
                raise ValueError(f"case {cid} needs memory_id or inline_text")
            if memory_id and not conn.execute("SELECT 1 FROM memories WHERE memory_id=?", (memory_id,)).fetchone():
                raise ValueError(f"case memory does not exist: {memory_id}")
            values = (bid, kind, memory_id, inline_text, str(case.get("weight_note") or "").strip(), case.get("occurred_at"), case.get("added") or datetime.now().date().isoformat(), float(case.get("emotion_score", 0.5)), case.get("created_by", "heng"))
            old = conn.execute("SELECT belief_id,case_kind,memory_id,inline_text,weight_note,occurred_at,added,emotion_score,created_by FROM belief_cases WHERE case_id=?", (cid,)).fetchone()
            if old is None:
                conn.execute("INSERT INTO belief_cases VALUES(?,?,?,?,?,?,?,?,?,?)", (cid, *values))
            elif tuple(old) != values:
                conn.execute("UPDATE belief_cases SET belief_id=?,case_kind=?,memory_id=?,inline_text=?,weight_note=?,occurred_at=?,added=?,emotion_score=?,created_by=? WHERE case_id=?", (*values, cid))
        for cognition_id, bid in desired_constellations:
            row = conn.execute("SELECT level,collection FROM memories WHERE memory_id=?", (cognition_id,)).fetchone()
            if not row or row["level"] != "cognition" or (row["collection"] or "") == "wenku":
                raise ValueError(f"CONSTELLATES source must be cognition: {cognition_id}")
            conn.execute("INSERT OR IGNORE INTO belief_constellations VALUES(?,?,0.0,?,?)", (cognition_id, bid, datetime.now().date().isoformat(), "heng"))
        conn.commit()


def counts(db) -> dict:
    with db._conn() as conn:
        return {
            "beliefs": conn.execute("SELECT count(*) FROM belief_nodes").fetchone()[0],
            "cases": conn.execute("SELECT count(*) FROM belief_cases").fetchone()[0],
            "memory_cases": conn.execute("SELECT count(*) FROM belief_cases WHERE memory_id IS NOT NULL").fetchone()[0],
            "inline_cases": conn.execute("SELECT count(*) FROM belief_cases WHERE memory_id IS NULL").fetchone()[0],
            "constellations": conn.execute("SELECT count(*) FROM belief_constellations").fetchone()[0],
            "outbox": conn.execute("SELECT count(*) FROM kuzu_belief_outbox").fetchone()[0],
        }
