#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""agent daily health dashboard.

Read-only against service data; writes only reports/ and its own baseline state.
"""
from __future__ import annotations

import json
import math
import os
import pwd
import grp
import random
import re
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from release_config import AnchorConfig

CONFIG = AnchorConfig.load()
ROOT = CONFIG.data_dir
_RUNTIME_MODULE_DIR = os.environ.get("ANCHOR_RUNTIME_MODULE_DIR", "").strip()
ANCHOR_DIR = Path(_RUNTIME_MODULE_DIR) if _RUNTIME_MODULE_DIR else None
REPORT_DIR = ROOT / "reports"
STATE_PATH = REPORT_DIR / ".daily_dashboard_state.json"

MEM_DB = CONFIG.db_path
KUZU_DB = CONFIG.kuzu_dir or (ROOT / "kuzu_db")
KUZU_WAL = Path(str(KUZU_DB) + ".wal")
RECALL_TRACE = Path(os.environ.get("ANCHOR_RECALL_TRACE", ROOT / "recall_trace.jsonl"))
BELIEFS_JSON = Path(os.environ.get("ANCHOR_BELIEFS_PATH", ROOT / "beliefs.json"))
RELAY_DB = CONFIG.chat_history_path or (ROOT / "chat-history.sqlite3")
DREAM_LOG = Path(os.environ.get("ANCHOR_DREAM_LOG", CONFIG.log_dir / "dream.log"))
MAINTENANCE_LOG = Path(os.environ.get("ANCHOR_MAINTENANCE_LOG", CONFIG.log_dir / "maintenance.log"))
THESEUS_SHADOW_LOG = Path(os.environ.get("ANCHOR_THESEUS_LOG", CONFIG.log_dir / "theseus-shadow.log"))
BACKUP_LOG = Path(os.environ.get("ANCHOR_BACKUP_LOG", CONFIG.log_dir / "backup.log"))
HEARTBEAT_LOG = ROOT / "heartbeat" / "heartbeat.log"

HOT_ACTIVATION_THRESHOLD = 2.0
GRAPH_HEALTH_URL = os.environ.get("ANCHOR_GRAPH_HEALTH_URL", "http://127.0.0.1:8765/api/graph/health")
CHAT_KINDS = ("user", "voice", "reply", "hug")
SYSTEMD_SERVICES = json.loads(os.environ.get("ANCHOR_DASHBOARD_SERVICES_JSON", "{}"))


def now_local() -> datetime:
    return datetime.now().astimezone()


def parse_dt(value: Any) -> datetime | None:
    if not value:
        return None
    s = str(value).strip()
    if not s:
        return None
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    for candidate in (s, s.replace(" ", "T", 1)):
        try:
            dt = datetime.fromisoformat(candidate)
            if dt.tzinfo is None:
                return dt.replace(tzinfo=now_local().tzinfo)
            return dt.astimezone(now_local().tzinfo)
        except ValueError:
            pass
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(s[:19], fmt).replace(tzinfo=now_local().tzinfo)
        except ValueError:
            pass
    return None


def parse_utc_dt(value: Any) -> datetime | None:
    """Parse SQLite/maintenance UTC timestamps and return local aware time."""
    if not value:
        return None
    s = str(value).strip()
    if not s:
        return None
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(s.replace(" ", "T", 1))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(now_local().tzinfo)


def local_stamp(dt: datetime | None) -> str | None:
    return dt.strftime("%Y-%m-%d %H:%M:%S") if dt else None


def ymd(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d")


def compact_ymd(dt: datetime) -> str:
    return dt.strftime("%Y%m%d")


def redact_text(text: str) -> str:
    text = re.sub(r"(?i)(bearer\s+)[A-Za-z0-9._~+/-]{12,}", r"\1[REDACTED_TOKEN]", text)
    text = re.sub(r"(?i)\b(sk-[A-Za-z0-9_-]{12,})\b", "[REDACTED_TOKEN]", text)
    text = re.sub(r"\bPDU[A-Za-z0-9_-]{16,}\b", "[REDACTED_TOKEN]", text)
    text = re.sub(r"\b[A-Za-z0-9_-]{36,}\b", "[REDACTED_TOKEN]", text)
    text = re.sub(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", "[REDACTED_IP]", text)
    return text


def sanitize_obj(obj: Any) -> Any:
    if isinstance(obj, str):
        return redact_text(obj)
    if isinstance(obj, list):
        return [sanitize_obj(x) for x in obj]
    if isinstance(obj, dict):
        return {k: sanitize_obj(v) for k, v in obj.items()}
    return obj


def collect_review_queue(today: datetime) -> dict[str, Any]:
    """Read pending decisions through update_review's only public reader."""
    result = {"available": False, "error": None, "updates": {}, "evokes": {}}
    try:
        if ANCHOR_DIR is not None and str(ANCHOR_DIR) not in sys.path:
            sys.path.insert(0, str(ANCHOR_DIR))
        import update_review
        pending = update_review.list_proposals(status="pending", limit=10000)
    except Exception as exc:
        result["error"] = f"review queue unavailable: {type(exc).__name__}: {str(exc)[:160]}"
        return result
    groups = {
        "evokes": [row for row in pending if row.get("kind") == "evokes"],
        "updates": [row for row in pending if row.get("kind") != "evokes"],
    }
    cutoff = today - timedelta(hours=24)
    for key, rows in groups.items():
        stamps = [parse_dt(row.get("created_at")) for row in rows]
        stamps = [stamp for stamp in stamps if stamp]
        oldest_hours = (max(0.0, (today - min(stamps)).total_seconds() / 3600)
                        if stamps else None)
        details = []
        for row in rows[:10]:
            details.append({
                "proposal_id": row.get("id", ""),
                "source_id": row.get("source_id") or row.get("new_id") or "",
                "target_id": row.get("target_id") or row.get("old_id") or "",
                "reason": first_line(row.get("reason") or "", 120),
                "confidence": row.get("confidence", row.get("llm_confidence")),
                "created_at": row.get("created_at", ""),
            })
        result[key] = {
            "pending": len(rows),
            "new_24h": sum(1 for stamp in stamps if stamp >= cutoff),
            "oldest_wait_hours": round(oldest_hours, 1) if oldest_hours is not None else None,
            "items": details,
        }
    result["available"] = True
    return result


def first_line(text: str, limit: int = 120) -> str:
    line = (text or "").splitlines()[0] if text else ""
    line = re.sub(r"\s+", " ", line).strip()
    line = redact_text(line)
    return line[:limit] + ("..." if len(line) > limit else "")


def fmt_bytes(n: int | float | None) -> str:
    if n is None:
        return "n/a"
    n = float(n)
    units = ["B", "KB", "MB", "GB", "TB"]
    for unit in units:
        if abs(n) < 1024 or unit == units[-1]:
            if unit == "B":
                return f"{int(n)} {unit}"
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def pct(part: int | float, total: int | float) -> str:
    return "0.0%" if not total else f"{(part / total) * 100:.1f}%"


def fmt_ms(value: int | float | None) -> str:
    return "n/a" if value is None else f"{value:.0f}"


def fmt_hours(value: int | float | None) -> str:
    return "n/a" if value is None else f"{value}h"


def quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    vals = sorted(values)
    if len(vals) == 1:
        return vals[0]
    pos = (len(vals) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return vals[lo]
    return vals[lo] * (hi - pos) + vals[hi] * (pos - lo)


def activation_distribution(values: list[float]) -> dict[str, Any]:
    return {
        "total": len(values),
        "quantiles": {
            "p25": quantile(values, 0.25),
            "p50": quantile(values, 0.50),
            "p75": quantile(values, 0.75),
            "max": max(values) if values else None,
        },
        "buckets": {
            "0-0.5": sum(1 for value in values if value < 0.5),
            "0.5-1": sum(1 for value in values if 0.5 <= value < 1.0),
            "1-2": sum(1 for value in values if 1.0 <= value < 2.0),
            "2+": sum(1 for value in values if value >= 2.0),
        },
    }


def table(headers: list[str], rows: list[list[Any]]) -> str:
    if not rows:
        return "_无_\n"
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        out.append("| " + " | ".join(str(x).replace("\n", "<br>") for x in row) + " |")
    return "\n".join(out) + "\n"


def q(conn: sqlite3.Connection, sql: str, params: tuple[Any, ...] = ()) -> list[sqlite3.Row]:
    return list(conn.execute(sql, params).fetchall())


def q1(conn: sqlite3.Connection, sql: str, params: tuple[Any, ...] = ()) -> Any:
    row = conn.execute(sql, params).fetchone()
    if row is None:
        return None
    return row[0]


def read_state() -> dict[str, Any]:
    try:
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def write_state(state: dict[str, Any]) -> None:
    tmp = STATE_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(STATE_PATH)


def read_previous_daily(today: datetime) -> dict[str, Any]:
    """Read the newest JSON report strictly older than today's report."""
    current = compact_ymd(today)
    for path in sorted(REPORT_DIR.glob("daily-????????.json"), reverse=True):
        stamp = path.stem.removeprefix("daily-")
        if stamp >= current:
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except Exception:
            continue
    return {}


def table_exists(conn: sqlite3.Connection, name: str) -> bool:
    return bool(q1(conn, "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)))


def snapshot_kuzu_edge_counts() -> dict[str, Any]:
    """Count live Kuzu relations from a disposable filesystem snapshot.

    Kuzu is process-exclusive even in read-only mode. Copying the small database
    and WAL avoids competing with anchor-sse's live lock. Failure is observable
    and never blocks the rest of the dashboard.
    """
    result: dict[str, Any] = {
        "available": False, "flow_edges": None, "semantic_edges": None,
        "legacy_edges": None, "source": "temporary_snapshot", "error": None,
    }
    try:
        import kuzu
        with tempfile.TemporaryDirectory(prefix="anchor-dashboard-kuzu-") as tmp:
            target = Path(tmp) / "kuzu_db"
            shutil.copy2(KUZU_DB, target)
            if KUZU_WAL.exists():
                shutil.copy2(KUZU_WAL, Path(str(target) + ".wal"))
            database = kuzu.Database(str(target))
            connection = kuzu.Connection(database)

            def visible_count(relation: str, expression: str) -> int:
                query = f"MATCH (a:Memory)-[e:{relation}]->(b:Memory) RETURN {expression}"
                cursor = connection.execute(query)
                count = 0
                while cursor.has_next():
                    cursor.get_next()
                    count += 1
                return count

            result.update({
                "available": True,
                "flow_edges": visible_count("FlowEdge", "a.memory_id, b.memory_id"),
                "semantic_edges": visible_count("SemanticEdge", "e.edge_id"),
                "legacy_edges": visible_count("EDGE", "a.memory_id, b.memory_id"),
            })
            del connection, database
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def verify_cross_layer_activate(rows: list[dict[str, Any]], today: datetime) -> dict[str, Any]:
    """Verify the heatflow contract in a disposable SQLite backup, never live data."""
    result: dict[str, Any] = {"ok": False, "mode": "sqlite_copy", "checks": {}, "error": None}
    candidates = [row for row in rows
                  if 0.12 <= float(row.get("weight") or 0) <= 0.24
                  and abs(float(row.get("conductance") or 0) - 0.25) < 1e-9
                  and (str(row.get("provenance") or "").startswith("auto_")
                       or row.get("provenance") in {"cluster", "knn_legacy"})]
    random.Random(today.strftime("%Y%m%d")).shuffle(candidates)
    old_flag = os.environ.get("ANCHOR_DUAL_EDGE")
    old_path = list(sys.path)
    original_init_kuzu = None
    try:
        with tempfile.TemporaryDirectory(prefix="anchor-dashboard-heatflow-") as tmp:
            copy_path = Path(tmp) / "memories.db"
            source = sqlite3.connect(f"file:{MEM_DB}?mode=ro", uri=True)
            target = sqlite3.connect(copy_path)
            source.backup(target)
            target.close(); source.close()
            sys.path.insert(0, str(ANCHOR_DIR))
            os.environ["ANCHOR_DUAL_EDGE"] = "on"
            from anchor_db import AnchorDB
            original_init_kuzu = AnchorDB._init_kuzu
            def no_kuzu(instance):
                instance._kuzu_db = None
                instance._kuzu_conn = None
            AnchorDB._init_kuzu = no_kuzu
            db = AnchorDB(str(copy_path))
            if candidates:
                chosen = candidates[0]
            else:
                with db._conn() as conn:
                    pair = conn.execute("""
                        SELECT a.memory_id,a.level,b.memory_id,b.level
                        FROM memories a JOIN memories b ON a.memory_id!=b.memory_id
                        WHERE COALESCE(a.level,'raw')!=COALESCE(b.level,'raw') LIMIT 1
                    """).fetchone()
                    source_id,target_id = pair[0],pair[2]
                    conn.execute("DELETE FROM flow_edges WHERE source_id=? AND target_id=?",
                                 (source_id,target_id))
                    conn.execute("INSERT INTO flow_edges VALUES(?,?,?,?,?,?,?)",
                                 (source_id,target_id,0.18,0.25,"dashboard","dashboard","auto_dashboard"))
                    conn.commit()
                chosen = {"source_id":source_id,"target_id":target_id,"weight":0.18,
                          "conductance":0.25,"provenance":"auto_dashboard",
                          "source_level":pair[1],"target_level":pair[3],"synthetic":True}
            source_id, target_id = chosen["source_id"], chosen["target_id"]
            with db._conn() as conn:
                before = float(q1(conn, "SELECT activation_score FROM memories WHERE memory_id=?", (target_id,)) or 0)
                edge_before = conn.execute(
                    "SELECT weight,last_fired FROM flow_edges WHERE source_id=? AND target_id=?",
                    (source_id,target_id)).fetchone()
            heat = db.apply_heat([source_id], 0.12, "dashboard-heat-once",
                                 spread=True, max_depth=1, source="dashboard")
            with db._conn() as conn:
                after = float(q1(conn, "SELECT activation_score FROM memories WHERE memory_id=?", (target_id,)) or 0)
                edge_after = conn.execute(
                    "SELECT weight,last_fired FROM flow_edges WHERE source_id=? AND target_id=?",
                    (source_id,target_id)).fetchone()
            expected = 0.12 * 0.5 * min(float(chosen["weight"])/1.5,1.0) * float(chosen["conductance"])
            delta = after-before
            result["checks"]["weak_flow_conductance"] = {
                "ok": abs(delta-expected) < 1e-9, "sample": chosen,
                "delta": delta, "expected": expected,
            }
            result["checks"]["last_fired_only_on_path"] = {
                "ok": bool(heat.get("edges")) and edge_before[0] == edge_after[0]
                      and edge_before[1] != edge_after[1],
                "edges_fired": heat.get("edges"), "weight_unchanged": edge_before[0] == edge_after[0],
            }
            replay = db.apply_heat([source_id], 0.12, "dashboard-heat-once",
                                   spread=True, max_depth=1, source="dashboard")
            with db._conn() as conn:
                replay_after = float(q1(conn, "SELECT activation_score FROM memories WHERE memory_id=?", (target_id,)) or 0)
            result["checks"]["event_id_idempotent"] = {
                "ok": replay.get("duplicate") is True and replay_after == after,
            }
            with db._conn() as conn:
                semantic = conn.execute("""
                    SELECT s.source_id,s.target_id FROM semantic_edges s
                    WHERE NOT EXISTS (
                      SELECT 1 FROM flow_edges f
                      WHERE f.source_id=s.source_id AND f.target_id=s.target_id
                    ) LIMIT 1
                """).fetchone()
                if semantic:
                    conn.execute("UPDATE semantic_edges SET strength=8.0 WHERE source_id=? AND target_id=?",
                                 (semantic[0],semantic[1]))
                    sem_before = float(q1(conn, "SELECT activation_score FROM memories WHERE memory_id=?", (semantic[1],)) or 0)
                    conn.commit()
            sem_ok = semantic is not None
            if semantic:
                db.apply_heat([semantic[0]], 0.12, "dashboard-semantic-isolation",
                              spread=True, max_depth=1, source="dashboard")
                with db._conn() as conn:
                    sem_after = float(q1(conn, "SELECT activation_score FROM memories WHERE memory_id=?", (semantic[1],)) or 0)
                sem_ok = sem_after == sem_before
            result["checks"]["semantic_never_conducts"] = {"ok": sem_ok}
            with db._conn() as conn:
                ordinary = conn.execute("""
                    SELECT memory_id FROM memories m WHERE NOT EXISTS (
                      SELECT 1 FROM flow_edges f WHERE f.source_id=m.memory_id
                    ) LIMIT 1
                """).fetchone()[0]
                protected = conn.execute("""
                    SELECT source_id FROM flow_edges GROUP BY source_id
                    ORDER BY SUM(weight*conductance) DESC LIMIT 1
                """).fetchone()[0]
                conn.execute("UPDATE memories SET activation_score=1.0,emotion_score=0.5 WHERE memory_id=?",(ordinary,))
                conn.execute("UPDATE memories SET activation_score=1.0,emotion_score=1.0 WHERE memory_id=?",(protected,))
                before_changes = conn.total_changes
                conn.execute("""
                    SELECT memory_id FROM memories m WHERE activation_score>=0.6
                    AND NOT EXISTS (SELECT 1 FROM semantic_edges s WHERE s.target_id=m.memory_id AND s.role='updates' AND s.review_state IN ('auto','approved'))
                    ORDER BY activation_score DESC,COALESCE(last_heated_at,'') DESC LIMIT 3
                """).fetchall()
                briefing_writes = conn.total_changes-before_changes
                conn.commit()
            db.decay_activation(factor=0.82)
            with db._conn() as conn:
                ordinary_after=float(q1(conn,"SELECT activation_score FROM memories WHERE memory_id=?",(ordinary,)))
                protected_after=float(q1(conn,"SELECT activation_score FROM memories WHERE memory_id=?",(protected,)))
            result["checks"]["daily_decay"] = {
                "ok": abs(ordinary_after-0.82)<1e-9 and 0.82 <= protected_after <= 0.90,
                "ordinary": ordinary_after, "protected": protected_after,
            }
            result["checks"]["briefing_zero_write"] = {"ok": briefing_writes == 0}
            anchor_code=(ANCHOR_DIR/"anchor_sse.py").read_text(encoding="utf-8")
            night=anchor_code[anchor_code.index("def _night_flow_repair_sync"):anchor_code.index('@_rest.post("/api/internal/night-flow-repair")')]
            maintenance=(ANCHOR_DIR/"maintenance.py").read_text(encoding="utf-8")
            result["checks"]["single_flow_decay_owner"] = {
                "ok": "weight*0.96" not in night.replace(" ","") and "0.96" in maintenance,
                "owner": "03:30 maintenance",
            }
            result["ok"] = all(item.get("ok") for item in result["checks"].values())
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if original_init_kuzu is not None:
            try:
                from anchor_db import AnchorDB
                AnchorDB._init_kuzu = original_init_kuzu
            except Exception:
                pass
        if old_flag is None: os.environ.pop("ANCHOR_DUAL_EDGE", None)
        else: os.environ["ANCHOR_DUAL_EDGE"] = old_flag
        sys.path[:] = old_path
    return result

def collect_tag_health(today: datetime, yesterday: datetime) -> dict[str, Any]:
    conn = sqlite3.connect(MEM_DB)
    conn.row_factory = sqlite3.Row
    y = ymd(yesterday)
    leaks = [dict(r) for r in q(conn, """
        SELECT memory_id, tag, timestamp, replace(substr(text, 1, 120), char(10), ' / ') AS first
        FROM memories
        WHERE COALESCE(collection, '') != 'wenku'
          AND COALESCE(tag, '') NOT LIKE 'state:%'
        ORDER BY timestamp DESC
    """)]
    currents = []
    for r in q(conn, """
        SELECT memory_id, text, tag, timestamp
        FROM memories
        WHERE COALESCE(collection, '') != 'wenku'
          AND COALESCE(tag, '') LIKE 'state:current%'
        ORDER BY timestamp DESC
    """):
        ts = parse_dt(r["timestamp"])
        days = (today - ts).days if ts else None
        currents.append({
            "id": r["memory_id"],
            "first": first_line(r["text"]),
            "tag": r["tag"],
            "timestamp": r["timestamp"],
            "days": days,
            "over_14": bool(days is not None and days > 14),
        })
    new_count = q1(conn, """
        SELECT COUNT(*) FROM memories
        WHERE COALESCE(collection, '') != 'wenku'
          AND date(timestamp, 'localtime') = ?
    """, (y,)) or 0
    new_wenku = q1(conn, """
        SELECT COUNT(*) FROM memories
        WHERE COALESCE(collection, '') = 'wenku'
          AND date(timestamp, 'localtime') = ?
    """, (y,)) or 0
    tag_dist = [dict(r) for r in q(conn, """
        SELECT COALESCE(tag, '') AS tag, COUNT(*) AS count
        FROM memories
        WHERE COALESCE(collection, '') != 'wenku'
          AND date(timestamp, 'localtime') = ?
        GROUP BY COALESCE(tag, '')
        ORDER BY count DESC, tag ASC
    """, (y,))]
    conn.close()
    return {"leaks": leaks, "currents": currents, "new_count": new_count, "new_wenku": new_wenku, "tag_distribution": tag_dist}


def load_recall_records(since: datetime) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not RECALL_TRACE.exists():
        return records
    with RECALL_TRACE.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
            except Exception:
                continue
            dt = parse_dt(obj.get("t"))
            if dt and dt >= since:
                records.append(obj)
    return records


def collect_recall(today: datetime) -> dict[str, Any]:
    records = [r for r in load_recall_records(today - timedelta(hours=24)) if r.get("stage") == "reflex"]
    total = len(records)
    rerank_ms: list[float] = []
    total_ms: list[float] = []
    fallback = 0
    empty_judgement = 0
    low_signal = 0
    technical = 0
    recently_rejected_items = 0
    recently_rejected_calls = 0
    candidates: list[dict[str, Any]] = []
    for r in records:
        gate = r.get("gate")
        if gate == "low_signal":
            low_signal += 1
        if gate == "technical_no_reflex":
            technical += 1
        timings = r.get("timings_ms") or {}
        if isinstance(timings, dict):
            if isinstance(timings.get("rerank"), (int, float)):
                rerank_ms.append(float(timings["rerank"]))
            if isinstance(timings.get("total"), (int, float)):
                total_ms.append(float(timings["total"]))
        rr = r.get("rerank") or {}
        fb = rr.get("fallback") or {}
        if isinstance(fb, dict) and fb.get("used"):
            fallback += 1
        if gate == "ok" and isinstance(rr, dict) and rr.get("model_selected") == [] and not (isinstance(fb, dict) and fb.get("used")):
            empty_judgement += 1
        call_recent = False
        for rej in r.get("rejected") or []:
            reason = str(rej.get("reason", ""))
            if rej.get("recently_injected") or "刚注入" in reason or "recently_injected" in reason:
                recently_rejected_items += 1
                call_recent = True
        for rej in (rr.get("model_rejected") or []):
            reason = str(rej.get("reason", ""))
            if rej.get("recently_injected") or "刚注入" in reason or "recently_injected" in reason:
                recently_rejected_items += 1
                call_recent = True
        if call_recent:
            recently_rejected_calls += 1
        if gate == "ok" and (r.get("selected") or r.get("rejected")):
            candidates.append(r)
    rng = random.Random(today.strftime("%Y%m%d"))
    sample = rng.sample(candidates, min(3, len(candidates))) if candidates else []
    return {
        "calls": total,
        "rerank_ms_p50": quantile(rerank_ms, 0.50),
        "rerank_ms_p95": quantile(rerank_ms, 0.95),
        "total_ms_p95": quantile(total_ms, 0.95),
        "fallback": fallback,
        "empty_judgement": empty_judgement,
        "low_signal": low_signal,
        "technical": technical,
        "low_signal_pct": pct(low_signal, total),
        "technical_pct": pct(technical, total),
        "recently_rejected_items": recently_rejected_items,
        "recently_rejected_calls": recently_rejected_calls,
        "samples": sample,
    }


def latest_dream(today: datetime) -> dict[str, Any]:
    due = today.hour > 4 or (today.hour == 4 and today.minute >= 45)
    result = {"ran_since_midnight": False, "due": due, "last_time": None, "stats": {}, "line": ""}
    if not DREAM_LOG.exists():
        return result
    pattern = re.compile(r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\] dream_pass: (\{.*?\})(?: \||$)")
    midnight = today.replace(hour=0, minute=0, second=0, microsecond=0)
    try:
        lines = DREAM_LOG.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception:
        return result
    for line in lines:
        m = pattern.search(line)
        if not m:
            continue
        dt = parse_dt(m.group(1))
        stats = {}
        try:
            stats = json.loads(m.group(2))
        except Exception:
            pass
        result = {"ran_since_midnight": bool(dt and dt >= midnight), "due": due, "last_time": m.group(1), "stats": stats, "line": line}
    return result


def latest_json_object(path: Path) -> dict[str, Any] | None:
    """Return the last complete object from a concatenated pretty-JSON log."""
    try:
        raw = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return None
    decoder = json.JSONDecoder()
    found: dict[str, Any] | None = None
    cursor = 0
    while cursor < len(raw):
        start = raw.find("{", cursor)
        if start < 0:
            break
        try:
            value, end = decoder.raw_decode(raw, start)
        except json.JSONDecodeError:
            cursor = start + 1
            continue
        if isinstance(value, dict):
            found = value
        cursor = end
    return found


def latest_json_line(path: Path) -> dict[str, Any] | None:
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception:
        return None
    for line in reversed(lines):
        try:
            value = json.loads(line)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(value, dict):
            return value
    return None


def latest_maintenance(conn: sqlite3.Connection, since: datetime) -> dict[str, Any]:
    rows = q(conn, """
        SELECT detail, created_at FROM events
        WHERE event_type='maintenance'
        ORDER BY created_at DESC LIMIT 1
    """)
    candidates: list[dict[str, Any]] = []
    if rows:
        detail = rows[0]["detail"] or "{}"
        try:
            data = json.loads(detail)
        except Exception:
            data = {"raw": detail}
        candidates.append({
            "dt": parse_utc_dt(rows[0]["created_at"]),
            "summary": data.get("summary", []) if isinstance(data, dict) else [],
            "source": "events",
        })

    log_report = latest_json_object(MAINTENANCE_LOG)
    if log_report and not log_report.get("dry_run"):
        results = log_report.get("results") or []
        if isinstance(results, list) and not any(isinstance(row, dict) and row.get("error") for row in results):
            candidates.append({
                "dt": parse_utc_dt(log_report.get("finished_at")),
                "summary": results,
                "source": "maintenance.log",
            })

    candidates = [item for item in candidates if item.get("dt")]
    if not candidates:
        return {"last_time": None, "recent": False, "summary": [], "source": None}
    latest = max(candidates, key=lambda item: item["dt"])
    return {
        "last_time": local_stamp(latest["dt"]),
        "recent": latest["dt"] >= since,
        "summary": latest["summary"],
        "source": latest["source"],
    }


def compact_actions(rows: list[dict[str, Any]], allowed: tuple[str, ...]) -> dict[str, Any]:
    actions: dict[str, Any] = {}
    for row in rows:
        task = str(row.get("task") or "")
        for key in allowed:
            if key in row:
                actions[f"{task}.{key}" if task else key] = row[key]
    return actions


def collect_night_batches(today: datetime, maintenance: dict[str, Any], dream: dict[str, Any]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    rows.append({
        "task": "maintenance",
        "cron": "03:30",
        "last_success": maintenance.get("last_time"),
        "recent": maintenance.get("recent", False),
        "actions": compact_actions(maintenance.get("summary") or [], (
            "candidates", "orphans", "weak_deleted", "stale_deleted", "machine_decayed",
            "selected_pairs", "missing_repaired", "dirty", "reclaimed",
        )),
    })

    dream_dt = parse_dt(dream.get("last_time"))
    dream_stats = dream.get("stats") or {}
    rows.append({
        "task": "dream_pass → night_repair",
        "cron": "04:30",
        "last_success": local_stamp(dream_dt),
        "recent": bool(dream_dt and dream_dt >= today - timedelta(hours=24)),
        "actions": {key: dream_stats[key] for key in (
            "nodes_scanned", "pairs_created", "flow_rows_created", "flow_rows_decayed",
            "activation_decayed",
        ) if key in dream_stats},
    })

    shadow = latest_json_line(THESEUS_SHADOW_LOG)
    shadow_dt = (datetime.fromtimestamp(THESEUS_SHADOW_LOG.stat().st_mtime, tz=now_local().tzinfo)
                 if shadow and THESEUS_SHADOW_LOG.exists() else None)
    shadow_actions: dict[str, Any] = {}
    if shadow:
        for key in ("scanned", "succeeded", "failed", "chunks", "indexed"):
            if key in shadow:
                shadow_actions[key] = shadow[key]
        audit = shadow.get("audit") or {}
        if "consistent" in audit:
            shadow_actions["index_consistent"] = audit["consistent"]
    rows.append({
        "task": "Theseus shadow",
        "cron": "05:30",
        "last_success": local_stamp(shadow_dt),
        "recent": bool(shadow_dt and shadow_dt >= today - timedelta(hours=24)),
        "actions": shadow_actions,
    })
    return {"rows": rows, "all_recent": all(row["recent"] for row in rows)}


def collect_theseus(today: datetime) -> dict[str, Any]:
    result: dict[str, Any] = {
        "available": False, "error": None, "seed_total": 0, "new_seeds_7d": 0,
        "shadow_chunks": 0, "indexed_chunks": 0, "shadow_parents": 0,
        "evokes": {"approved": 0, "rejected": 0, "pending": 0, "decided": 0, "pass_rate": None},
    }
    try:
        conn = sqlite3.connect(f"file:{MEM_DB}?mode=ro", uri=True, timeout=10)
        conn.row_factory = sqlite3.Row
        seed_rows = q(conn, """
            SELECT timestamp FROM memories
            WHERE collection='wenku' AND instr(',' || tag || ',', ',种子,') > 0
        """)
        result["seed_total"] = len(seed_rows)
        cutoff_utc = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=7)
        result["new_seeds_7d"] = sum(
            1 for row in seed_rows
            if (lambda stamp: bool(stamp and stamp >= cutoff_utc))(
                datetime.fromisoformat(str(row["timestamp"]).replace("Z", "+00:00")).replace(tzinfo=None)
                if row["timestamp"] else None
            )
        )
        if table_exists(conn, "theseus_shadows"):
            result["shadow_chunks"] = int(q1(conn, "SELECT COUNT(*) FROM theseus_shadows") or 0)
            result["indexed_chunks"] = int(q1(conn, "SELECT COUNT(*) FROM theseus_shadows WHERE index_policy='index'") or 0)
            result["shadow_parents"] = int(q1(conn, "SELECT COUNT(DISTINCT parent_memory_id) FROM theseus_shadows") or 0)
        conn.close()

        if ANCHOR_DIR is not None and str(ANCHOR_DIR) not in sys.path:
            sys.path.insert(0, str(ANCHOR_DIR))
        import update_review
        proposals = update_review.list_proposals(status="all", limit=100000)
        evokes_rows = [row for row in proposals if row.get("kind") == "evokes"]
        counts = Counter(str(row.get("status") or "pending") for row in evokes_rows)
        approved = counts.get("approved", 0)
        rejected = counts.get("rejected", 0)
        decided = approved + rejected
        result["evokes"] = {
            "approved": approved,
            "rejected": rejected,
            "pending": counts.get("pending", 0),
            "decided": decided,
            "pass_rate": round(approved * 100 / decided, 1) if decided else None,
        }
        result["available"] = True
    except Exception as exc:
        result["error"] = f"Theseus health unavailable: {type(exc).__name__}: {str(exc)[:160]}"
    return result


def fetch_graph_health() -> dict[str, Any]:
    try:
        with urllib.request.urlopen(GRAPH_HEALTH_URL, timeout=10) as response:
            return json.loads(response.read().decode("utf-8"))
    except (OSError, ValueError, urllib.error.URLError) as exc:
        return {
            "ok": False,
            "backend": "sqlite_fallback",
            "error": f"{type(exc).__name__}: {exc}",
        }


def collect_graph(today: datetime, state: dict[str, Any]) -> dict[str, Any]:
    conn = sqlite3.connect(MEM_DB)
    conn.row_factory = sqlite3.Row
    previous_daily = read_previous_daily(today)
    sqlite_island_count = q1(conn, """
        SELECT COUNT(*) FROM memories m
        WHERE COALESCE(m.collection, '') != 'wenku'
          AND NOT EXISTS (SELECT 1 FROM edges e WHERE e.source_id=m.memory_id OR e.target_id=m.memory_id)
    """) or 0
    sqlite_edge_total = q1(conn, "SELECT COUNT(*) FROM edges") or 0
    flow_total = q1(conn, "SELECT COUNT(*) FROM flow_edges") if table_exists(conn, "flow_edges") else 0
    semantic_total = q1(conn, "SELECT COUNT(*) FROM semantic_edges") if table_exists(conn, "semantic_edges") else 0
    flow_total = int(flow_total or 0)
    semantic_total = int(semantic_total or 0)
    flow_by_provenance = {
        r["provenance"]: r["n"] for r in q(conn, """
            SELECT COALESCE(provenance, 'unknown') provenance, COUNT(*) n
            FROM flow_edges GROUP BY COALESCE(provenance, 'unknown')
            ORDER BY n DESC, provenance
        """)
    } if table_exists(conn, "flow_edges") else {}
    semantic_breakdown: dict[str, Any] = {}
    if table_exists(conn, "semantic_edges"):
        for row in q(conn, """
            SELECT role, review_state, COUNT(*) n FROM semantic_edges
            GROUP BY role, review_state ORDER BY role, review_state
        """):
            role = row["role"]
            item = semantic_breakdown.setdefault(role, {"total": 0, "review_states": {}})
            item["total"] += row["n"]
            item["review_states"][row["review_state"]] = row["n"]
    dual_total = flow_total + semantic_total
    migration_gap = max(0, sqlite_edge_total - dual_total)
    growth_over_legacy = max(0, dual_total - sqlite_edge_total)
    previous_dual = (((state.get("graph") or {}).get("dual_edges") or {}).get("total"))
    dual_delta = dual_total - previous_dual if isinstance(previous_dual, int) else None

    outbox_v2_pending = (q1(conn, "SELECT COUNT(*) FROM kuzu_edge_outbox_v2") or 0
                         if table_exists(conn, "kuzu_edge_outbox_v2") else None)
    legacy_edge_outbox = (q1(conn, "SELECT COUNT(*) FROM kuzu_edge_outbox") or 0
                          if table_exists(conn, "kuzu_edge_outbox") else None)
    hot_rows = [dict(r) for r in q(conn, """
        SELECT memory_id, activation_score, tag, replace(substr(text, 1, 120), char(10), ' / ') AS first
        FROM memories
        WHERE COALESCE(collection, '') != 'wenku'
          AND activation_score >= ?
        ORDER BY activation_score DESC, timestamp DESC
        LIMIT 50
    """, (HOT_ACTIVATION_THRESHOLD,))]
    hot_count = q1(conn, """
        SELECT COUNT(*) FROM memories
        WHERE COALESCE(collection, '') != 'wenku'
          AND activation_score >= ?
    """, (HOT_ACTIVATION_THRESHOLD,)) or 0
    activation_values = [float(r[0] or 0.0) for r in q(conn, """
        SELECT activation_score FROM memories WHERE COALESCE(collection, '') != 'wenku'
    """)]
    activation_stats = activation_distribution(activation_values)
    dream = latest_dream(today)
    maint = latest_maintenance(conn, today - timedelta(hours=24))
    pending_db = q1(conn, "SELECT COUNT(*) FROM pending_memories WHERE status='pending'") or 0
    typed_edges = {r["edge_type"]: r["n"] for r in q(conn, """
        SELECT edge_type, COUNT(*) AS n FROM edges
        WHERE edge_type IN ('SUPPORTED_BY', 'GROUNDED_IN', 'EVOKES', 'updates')
        GROUP BY edge_type
    """)}

    since_utc = (datetime.now(timezone.utc) - timedelta(hours=24)).replace(tzinfo=None).isoformat()
    integrated_rows = q(conn, """
        SELECT created_at, detail FROM events
        WHERE event_type='integrated' AND created_at >= ? ORDER BY created_at
    """, (since_utc,))
    integrated_summary = {
        "new_memories": len(integrated_rows), "flow_edges_event_total": 0,
        "auto_flow_edges": 0, "semantic_edges_event_total": 0,
        "auto_semantic_edges": 0, "review_proposals": 0,
        "degree_cap_hits": None, "degree_cap_nodes": [], "parse_errors": 0,
    }
    cap_nodes: set[str] = set()
    cap_hits = 0
    cap_field_seen = False
    for row in integrated_rows:
        try:
            detail = json.loads(row["detail"] or "{}")
            if not isinstance(detail, dict):
                raise ValueError("integrated detail is not an object")
        except Exception:
            integrated_summary["parse_errors"] += 1
            continue
        integrated_summary["flow_edges_event_total"] += int(detail.get("flow_edges_created") or 0)
        integrated_summary["semantic_edges_event_total"] += int(detail.get("semantic_edges_created") or 0)
        integrated_summary["review_proposals"] += int(detail.get("review_proposals_created") or 0)
        if "degree_cap_hits" in detail:
            cap_field_seen = True
            cap_hits += int(detail.get("degree_cap_hits") or 0)
        for node in detail.get("degree_cap_nodes") or []:
            cap_nodes.add(str(node))
        for warning in detail.get("warnings") or []:
            if "degree_cap" in str(warning).lower() or "degree cap" in str(warning).lower():
                cap_field_seen = True
                cap_hits += 1
    if table_exists(conn, "flow_edges"):
        integrated_summary["auto_flow_edges"] = q1(conn, """
            SELECT COUNT(*) FROM flow_edges
            WHERE provenance='auto_integrate' AND created >= ?
        """, (since_utc,)) or 0
    if table_exists(conn, "semantic_edges"):
        integrated_summary["auto_semantic_edges"] = q1(conn, """
            SELECT COUNT(*) FROM semantic_edges
            WHERE review_state='auto' AND created >= ?
        """, (since_utc,)) or 0
    integrated_summary["degree_cap_hits"] = cap_hits if cap_field_seen else None
    integrated_summary["degree_cap_nodes"] = sorted(cap_nodes)

    cross_rows = [dict(r) for r in q(conn, """
        SELECT f.source_id, f.target_id, f.weight, f.conductance, f.provenance,
               COALESCE(s.level, 'raw') source_level,
               COALESCE(t.level, 'raw') target_level
        FROM flow_edges f
        JOIN memories s ON s.memory_id=f.source_id
        JOIN memories t ON t.memory_id=f.target_id
        WHERE COALESCE(s.level, 'raw') != COALESCE(t.level, 'raw')
        ORDER BY f.source_id, f.target_id
    """)] if table_exists(conn, "flow_edges") else []
    cross_eligible = [row for row in cross_rows
                      if float(row.get("weight") or 0) > 0
                      and float(row.get("conductance") or 0) > 0]
    conn.close()
    graph_health = fetch_graph_health()
    kuzu_v2 = snapshot_kuzu_edge_counts()
    kuzu_v2["flow_consistent"] = bool(
        kuzu_v2.get("available") and kuzu_v2.get("flow_edges") == flow_total)
    kuzu_v2["semantic_consistent"] = bool(
        kuzu_v2.get("available") and kuzu_v2.get("semantic_edges") == semantic_total)
    kuzu_v2["legacy_consistent"] = bool(
        kuzu_v2.get("available") and kuzu_v2.get("legacy_edges") == sqlite_edge_total)
    kuzu_v2["consistent"] = bool(
        kuzu_v2.get("flow_consistent") and kuzu_v2.get("semantic_consistent")
        and kuzu_v2.get("legacy_consistent") and outbox_v2_pending in (None, 0)
        and legacy_edge_outbox in (None, 0))
    graph_from_kuzu = graph_health.get("ok") and graph_health.get("backend") == "kuzu"
    island_count = int(graph_health.get("islands", sqlite_island_count)) if graph_from_kuzu else sqlite_island_count
    edge_total = int(graph_health.get("edges", sqlite_edge_total)) if graph_from_kuzu else sqlite_edge_total
    pending_files = len(list((ANCHOR_DIR / "drafts" / "pending").glob("*.json"))) if (ANCHOR_DIR / "drafts" / "pending").exists() else 0
    drafts_root = ANCHOR_DIR / "drafts"
    draft_counts = {}
    if drafts_root.exists():
        for sub in ("pending", "done", "skipped", "archived"):
            p = drafts_root / sub
            draft_counts[sub] = len(list(p.glob("*.json"))) if p.exists() else 0
    previous_drafts = (((previous_daily.get("graph") or {}).get("draft_counts")) or {})
    draft_trend = {
        key: {
            "previous": previous_drafts.get(key),
            "current": draft_counts.get(key, 0),
            "delta": (draft_counts.get(key, 0) - previous_drafts[key]
                     if isinstance(previous_drafts.get(key), int) else None),
        }
        for key in ("pending", "done", "skipped", "archived")
    }
    previous = state.get("graph", {})
    edge_delta = edge_total - previous.get("edge_total") if isinstance(previous.get("edge_total"), int) else None
    prev_typed = previous.get("typed_edges") or {}
    typed_delta = {
        k: typed_edges.get(k, 0) - prev_typed[k]
        for k in prev_typed
        if isinstance(prev_typed.get(k), int) and typed_edges.get(k, 0) != prev_typed[k]
    }
    return {
        "islands": island_count,
        "edge_total": edge_total,
        "edge_delta": edge_delta,
        "source": "kuzu" if graph_from_kuzu else "sqlite_fallback",
        "health": graph_health,
        "hot_threshold": HOT_ACTIVATION_THRESHOLD,
        "hot_count": hot_count,
        "hot_top": hot_rows,
        "activation_distribution": activation_stats,
        "dream": dream,
        "maintenance": maint,
        "night_batches": collect_night_batches(today, maint, dream),
        "pending_memories": pending_db,
        "draft_counts": draft_counts,
        "draft_pending_files": pending_files,
        "typed_edges": typed_edges,
        "typed_edges_delta": typed_delta,
        "dual_edges": {
            "flow_total": flow_total,
            "flow_by_provenance": flow_by_provenance,
            "semantic_total": semantic_total,
            "semantic_breakdown": semantic_breakdown,
            "legacy_total": sqlite_edge_total,
            "total": dual_total,
            "delta": dual_delta,
            "migration_gap": migration_gap,
            "growth_over_legacy": growth_over_legacy,
        },
        "kuzu_v2": {
            **kuzu_v2,
            "sqlite_flow_edges": flow_total,
            "sqlite_semantic_edges": semantic_total,
            "sqlite_legacy_edges": sqlite_edge_total,
            "outbox_v2_pending": outbox_v2_pending,
            "legacy_edge_outbox_pending": legacy_edge_outbox,
        },
        "integrate_24h": integrated_summary,
        "draft_trend": draft_trend,
        "cross_layer_flow": {
            "total": len(cross_rows),
            "conducting_edges": len(cross_eligible),
            "activation_test": verify_cross_layer_activate(cross_eligible, today),
        },
    }


def case_dates(belief: dict[str, Any]) -> list[datetime]:
    dates: list[datetime] = []
    for key in ("support_cases", "contradiction_cases", "boundary_cases"):
        for case in belief.get(key) or []:
            dt = parse_dt(case.get("added") or case.get("created_at") or case.get("date"))
            if dt:
                dates.append(dt)
    return dates


def collect_beliefs(today: datetime, state: dict[str, Any]) -> dict[str, Any]:
    # Belief graph v2: SQLite is authoritative; beliefs.json is only a
    # compatibility snapshot. Fall back to the legacy reader during rollback.
    try:
        raw = json.loads(BELIEFS_JSON.read_text(encoding="utf-8"))
        params = {"half_life_days": 90.0, "recency_floor": 0.3,
                  "prior_k": 2.0, "routing_cutoff": 0.4}
        params.update(raw.get("params") or {})
        conn = sqlite3.connect(f"file:{MEM_DB}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        tables = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )}
        if "belief_nodes" in tables:
            seven_days = today - timedelta(days=7)
            thirty_days = today - timedelta(days=30)
            previous_conf = (state.get("beliefs", {}) or {}).get("confidence", {})
            new_case, stale, confidence_now, confidence_changed = [], [], {}, []
            non_routing_active = []

            def case_weight(case: sqlite3.Row) -> float:
                emotion = float(case["case_emotion"] or 0.5)
                stamp = case["case_time"]
                dt = parse_dt(stamp)
                age = max(0.0, (today - dt).total_seconds() / 86400.0) if dt else 0.0
                recency = max(float(params["recency_floor"]),
                              0.5 ** (age / float(params["half_life_days"])))
                return emotion * recency

            for b in conn.execute("SELECT * FROM belief_nodes ORDER BY belief_id"):
                cases = list(conn.execute("""
                    SELECT c.case_kind,c.added,
                           CASE WHEN c.memory_id IS NULL THEN c.emotion_score ELSE m.emotion_score END case_emotion,
                           CASE WHEN c.memory_id IS NULL THEN COALESCE(c.occurred_at,c.added) ELSE m.timestamp END case_time
                    FROM belief_cases c LEFT JOIN memories m ON m.memory_id=c.memory_id
                    WHERE c.belief_id=?
                """, (b["belief_id"],)))
                support = sum(case_weight(c) for c in cases if c["case_kind"] == "support")
                contradiction = sum(case_weight(c) for c in cases if c["case_kind"] == "contradiction")
                conf = support / (support + contradiction + float(params["prior_k"]))
                confidence_now[b["belief_id"]] = round(conf, 6)
                if b["belief_id"] in previous_conf and previous_conf[b["belief_id"]] != round(conf, 6):
                    confidence_changed.append({"id": b["belief_id"], "before": previous_conf[b["belief_id"]], "after": round(conf, 6)})
                recent = [c for c in cases if parse_dt(c["added"]) and parse_dt(c["added"]) >= seven_days]
                if recent:
                    new_case.append({"id": b["belief_id"], "statement": b["statement"], "new_cases": len(recent)})
                touch = parse_dt(b["last_touched"] or b["updated_at"] or b["created_at"])
                if not touch or touch < thirty_days:
                    stale.append({"id": b["belief_id"], "statement": b["statement"], "last_touched": b["last_touched"] or b["updated_at"] or b["created_at"] or "无"})
                if b["status"] == "active" and not (b["pinned"] or conf >= float(params["routing_cutoff"])):
                    non_routing_active.append({"id": b["belief_id"], "confidence": round(conf, 3), "statement": b["statement"]})
            graph = {
                "beliefs": conn.execute("SELECT count(*) FROM belief_nodes").fetchone()[0],
                "cases": conn.execute("SELECT count(*) FROM belief_cases").fetchone()[0],
                "memory_cases": conn.execute("SELECT count(*) FROM belief_cases WHERE memory_id IS NOT NULL").fetchone()[0],
                "inline_cases": conn.execute("SELECT count(*) FROM belief_cases WHERE memory_id IS NULL").fetchone()[0],
                "constellations": conn.execute("SELECT count(*) FROM belief_constellations").fetchone()[0],
                "outbox": conn.execute("SELECT count(*) FROM kuzu_belief_outbox").fetchone()[0],
                "invalid_constellations": conn.execute("""
                    SELECT count(*) FROM belief_constellations c
                    LEFT JOIN memories m ON m.memory_id=c.cognition_id
                    WHERE m.memory_id IS NULL OR COALESCE(m.level,'raw')!='cognition'
                       OR COALESCE(m.collection,'')='wenku'
                """).fetchone()[0],
            }
            conn.close()
            return {"total": graph["beliefs"], "new_case_7d": new_case,
                    "confidence_field": True, "confidence": confidence_now,
                    "confidence_changed": confidence_changed, "stale_30d": stale,
                    "non_routing_active": non_routing_active, "graph": graph}
        conn.close()
    except Exception:
        pass
    try:
        data = json.loads(BELIEFS_JSON.read_text(encoding="utf-8"))
    except Exception as e:
        return {"error": str(e)}
    beliefs = data.get("beliefs", []) if isinstance(data, dict) else []
    seven_days = today - timedelta(days=7)
    thirty_days = today - timedelta(days=30)
    new_case = []
    stale = []
    confidence_now = {}
    previous_conf = (state.get("beliefs", {}) or {}).get("confidence", {})
    confidence_changed = []
    has_confidence_field = False
    for b in beliefs:
        bid = b.get("id", "")
        dates = case_dates(b)
        if any(dt >= seven_days for dt in dates):
            new_case.append({"id": bid, "statement": b.get("statement", ""), "new_cases": sum(1 for dt in dates if dt >= seven_days)})
        touch = parse_dt(b.get("last_touched") or b.get("updated_at") or b.get("created_at"))
        if not touch or touch < thirty_days:
            stale.append({"id": bid, "statement": b.get("statement", ""), "last_touched": b.get("last_touched") or b.get("updated_at") or b.get("created_at") or "无"})
        if "confidence" in b:
            has_confidence_field = True
            confidence_now[bid] = b.get("confidence")
            if bid in previous_conf and previous_conf[bid] != b.get("confidence"):
                confidence_changed.append({"id": bid, "before": previous_conf[bid], "after": b.get("confidence")})
    return {
        "total": len(beliefs),
        "new_case_7d": new_case,
        "confidence_field": has_confidence_field,
        "confidence": confidence_now,
        "confidence_changed": confidence_changed,
        "stale_30d": stale,
    }


def run_cmd(args: list[str], timeout: int = 10) -> tuple[int, str]:
    try:
        p = subprocess.run(args, text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=timeout)
        return p.returncode, p.stdout.strip()
    except Exception as e:
        return 999, str(e)


def systemd_active(unit: str) -> str:
    code, out = run_cmd(["systemctl", "is-active", unit], timeout=5)
    return out.splitlines()[0] if out else ("unknown" if code else "active")


def restart_count(unit: str, start: datetime, end: datetime) -> int | None:
    start_s = start.strftime("%Y-%m-%d %H:%M:%S")
    end_s = end.strftime("%Y-%m-%d %H:%M:%S")
    code, out = run_cmd(["journalctl", "-u", unit, "--since", start_s, "--until", end_s, "--no-pager", "-o", "cat"], timeout=15)
    if code not in (0, 1):
        return None
    count = 0
    for line in out.splitlines():
        if re.search(r"\bStarted\b|\bStarting\b", line):
            count += 1
    return count


def crontab_has_heartbeat() -> bool:
    # /etc/cron.d runs this report as root; inspect the live body's crontab explicitly.
    code, out = run_cmd(["crontab", "-u", "local-user", "-l"], timeout=5)
    return code == 0 and "heartbeat/scheduler_tick.py" in out


def file_mtime(path: Path) -> datetime | None:
    try:
        return datetime.fromtimestamp(path.stat().st_mtime).astimezone()
    except Exception:
        return None


def collect_services(today: datetime, yesterday_start: datetime, today_start: datetime, state: dict[str, Any]) -> dict[str, Any]:
    services = []
    for label, unit in SYSTEMD_SERVICES.items():
        services.append({"name": label, "unit": unit, "active": systemd_active(unit), "yesterday_restarts": restart_count(unit, yesterday_start, today_start)})
    hb_mtime = file_mtime(HEARTBEAT_LOG)
    hb_cron = crontab_has_heartbeat()
    services.append({
        "name": "heartbeat",
        "unit": "local-user crontab heartbeat/scheduler_tick.py",
        "active": "active" if hb_cron else "warning",
        "yesterday_restarts": None,
        "last_log": hb_mtime.isoformat(timespec="seconds") if hb_mtime else None,
    })
    files = {}
    for name, path in {
        "memories.db": MEM_DB,
        "kuzu_db": KUZU_DB,
        "kuzu_db.wal": KUZU_WAL,
        "recall_trace.jsonl": RECALL_TRACE,
    }.items():
        try:
            size = path.stat().st_size
        except Exception:
            size = None
        prev = ((state.get("files", {}) or {}).get(name, {}) or {}).get("size")
        files[name] = {"path": str(path), "size": size, "delta": (size - prev) if isinstance(size, int) and isinstance(prev, int) else None}
    backup_ok = None
    if BACKUP_LOG.exists():
        for line in BACKUP_LOG.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("OK "):
                backup_ok = line
    return {"services": services, "files": files, "backup_last_ok": backup_ok}


def collect_reconcile(yesterday: datetime) -> dict[str, Any]:
    y = ymd(yesterday)
    mem_conn = sqlite3.connect(MEM_DB)
    mem_new = q1(mem_conn, """
        SELECT COUNT(*) FROM memories
        WHERE COALESCE(collection, '') != 'wenku'
          AND date(timestamp, 'localtime') = ?
    """, (y,)) or 0
    mem_conn.close()
    relay_conn = sqlite3.connect(RELAY_DB)
    relay_conn.row_factory = sqlite3.Row
    placeholders = ",".join("?" for _ in CHAT_KINDS)
    chat_count = q1(relay_conn, f"""
        SELECT COUNT(*) FROM messages
        WHERE kind IN ({placeholders})
          AND date(ts, 'localtime') = ?
    """, (*CHAT_KINDS, y)) or 0
    dist = [dict(r) for r in q(relay_conn, f"""
        SELECT kind, direction, COUNT(*) AS count
        FROM messages
        WHERE kind IN ({placeholders})
          AND date(ts, 'localtime') = ?
        GROUP BY kind, direction
        ORDER BY count DESC
    """, (*CHAT_KINDS, y))]
    relay_conn.close()
    ratio = (mem_new / chat_count) if chat_count else None
    alert = None
    if chat_count >= 100 and mem_new == 0:
        alert = "严重：聊天很多但昨日新增记忆为 0"
    elif chat_count >= 200 and ratio is not None and ratio < 0.005:
        alert = "偏低：聊天很多但记忆新增率 < 0.5%"
    return {"chat_count": chat_count, "new_memories": mem_new, "ratio": ratio, "alert": alert, "distribution": dist}


def render_sample(sample: dict[str, Any], index: int) -> str:
    lines = [f"### 抽样 {index}: {first_line(sample.get('q', ''), 160)}", ""]
    selected = sample.get("selected") or []
    rejected = sample.get("rejected") or []
    lines.append("**selected**")
    if selected:
        rows = []
        for item in selected:
            rows.append([item.get("id", ""), item.get("lane", ""), first_line(item.get("reason", ""), 80), first_line(item.get("snippet") or item.get("text") or "", 100)])
        lines.append(table(["id", "lane", "reason", "snippet"], rows))
    else:
        lines.append("_无_\n")
    lines.append("**rejected**")
    if rejected:
        rows = []
        for item in rejected[:10]:
            rows.append([item.get("id", ""), first_line(item.get("reason", ""), 90), first_line(item.get("text") or item.get("snippet") or "", 100)])
        if len(rejected) > 10:
            rows.append(["...", f"另 {len(rejected)-10} 条", ""])
        lines.append(table(["id", "reason", "text"], rows))
    else:
        lines.append("_无_\n")
    return "\n".join(lines)


def render_report(data: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append(f"# AI agent每日仪表盘 {data['report_date']}")
    lines.append("")
    lines.append(f"生成时间：{data['generated_at']}；统计昨日：{data['yesterday']}；反射弧窗口：近 24h。")
    alerts = data.get("alerts") or []
    lines.append("")
    lines.append("## 摘要")
    if alerts:
        for a in alerts:
            lines.append(f"- **{a}**")
    else:
        lines.append("- 无红色告警。")
    review = data.get("review_queue") or {}
    if review.get("available"):
        lines.append(
            f"- 待AI agent审批：updates {review.get('updates', {}).get('pending', 0)}，"
            f"EVOKES {review.get('evokes', {}).get('pending', 0)}"
        )
    else:
        lines.append("- **review queue unavailable**")

    lines.append("\n## 待AI agent审批")
    if not review.get("available"):
        lines.append(f"- {first_line(review.get('error') or 'review queue unavailable', 180)}")
    else:
        for key, label in (("updates", "updates/supersede"), ("evokes", "EVOKES")):
            group = review.get(key) or {}
            oldest = group.get("oldest_wait_hours")
            lines.append(
                f"- {label}：pending **{group.get('pending', 0)}**；近24h新增 "
                f"**{group.get('new_24h', 0)}**；最老等待 "
                f"**{fmt_hours(oldest)}**"
            )
            lines.append(table(
                ["proposal", "source", "target", "reason", "confidence", "created"],
                [[item.get("proposal_id", ""), item.get("source_id", ""),
                  item.get("target_id", ""), item.get("reason", ""),
                  item.get("confidence") if item.get("confidence") is not None else "n/a",
                  item.get("created_at", "")]
                 for item in group.get("items", [])],
            ))

    tag = data["tag_health"]
    lines.append("\n## 1. tag 健康")
    lines.append(f"- 漏网数：**{len(tag['leaks'])}**")
    if tag["leaks"]:
        lines.append("- 漏网 id：" + ", ".join(r["memory_id"] for r in tag["leaks"]))
    lines.append(f"- state:current：**{len(tag['currents'])}** 条；>14 天：**{sum(1 for r in tag['currents'] if r['over_14'])}** 条")
    rows = []
    for r in tag["currents"]:
        days = r["days"] if r["days"] is not None else "?"
        day_text = f"<span style='color:red'>{days}</span>" if r["over_14"] else str(days)
        rows.append([r["id"], day_text, r["timestamp"], r["first"]])
    lines.append(table(["id", "已挂天数", "timestamp", "首行"], rows))
    lines.append(f"- 昨日新增记忆：**{tag['new_count']}** 条（另 wenku {tag['new_wenku']} 条）")
    lines.append(table(["tag", "count"], [[r["tag"], r["count"]] for r in tag["tag_distribution"][:40]]))

    rec = data["recall"]
    lines.append("\n## 2. 反射弧性能")
    lines.append(f"- 调用次数：**{rec['calls']}**")
    lines.append(f"- rerank_ms p50/p95：**{fmt_ms(rec['rerank_ms_p50'])} / {fmt_ms(rec['rerank_ms_p95'])} ms**")
    lines.append(f"- total_ms p95：**{fmt_ms(rec['total_ms_p95'])} ms**")
    lines.append(f"- fallback：**{rec['fallback']}**；判空：**{rec['empty_judgement']}**")
    lines.append(f"- low_signal：**{rec['low_signal']} ({rec['low_signal_pct']})**；technical：**{rec['technical']} ({rec['technical_pct']})**")
    lines.append(f"- recently_injected 被拒：**{rec['recently_rejected_items']}** 条 / **{rec['recently_rejected_calls']}** 次调用")
    for i, sample in enumerate(rec["samples"], 1):
        lines.append(render_sample(sample, i))

    graph = data["graph"]
    delta = "首次建立基线" if graph["edge_delta"] is None else f"{graph['edge_delta']:+d}"
    lines.append("\n## 3. 记忆图")
    health = graph.get("health") or {}
    if graph.get("source") == "kuzu":
        sync_text = "一致" if health.get("consistent") else "不一致"
        lines.append(
            f"- 图数据源：**Kuzu**；同步：**{sync_text}**；"
            f"Kuzu nodes/edges={health.get('nodes', 'n/a')}/{health.get('edges', 'n/a')}；"
            f"SQLite mirror={health.get('sqlite_nodes', 'n/a')}/{health.get('sqlite_edges', 'n/a')}；"
            f"outbox node/edge={health.get('node_outbox', 'n/a')}/{health.get('edge_outbox', 'n/a')}"
        )
    else:
        lines.append(
            f"- 图数据源：**SQLite fallback**；Kuzu 健康快照失败："
            f"{first_line(health.get('error', 'unknown'), 180)}"
        )
    lines.append(f"- 孤岛数：**{graph['islands']}**")
    lines.append(f"- legacy 边总数（只读）：**{graph['edge_total']}**（对比上一份日报：{delta}）")
    lines.append(f"- activation >= {graph['hot_threshold']}：**{graph['hot_count']}** 条")
    activation = graph.get("activation_distribution") or {}
    aq = activation.get("quantiles") or {}
    ab = activation.get("buckets") or {}
    fmt_activation = lambda value: "n/a" if value is None else f"{float(value):.3f}"
    lines.append(
        f"- activation 分位数（{activation.get('total', 0)} 条）："
        f"p25 **{fmt_activation(aq.get('p25'))}** / p50 **{fmt_activation(aq.get('p50'))}** / "
        f"p75 **{fmt_activation(aq.get('p75'))}** / max **{fmt_activation(aq.get('max'))}**"
    )
    lines.append(
        f"- activation 分桶：0–0.5 **{ab.get('0-0.5', 0)}**；0.5–1 **{ab.get('0.5-1', 0)}**；"
        f"1–2 **{ab.get('1-2', 0)}**；2+ **{ab.get('2+', 0)}**"
    )
    lines.append(table(["id", "activation", "tag", "首行"], [[r["memory_id"], f"{r['activation_score']:.2f}", r["tag"], r["first"]] for r in graph["hot_top"][:20]]))
    lines.append("\n### 3.1 夜批统一看板")
    night_rows = (graph.get("night_batches") or {}).get("rows") or []
    lines.append(table(
        ["任务", "cron", "上次成功", "近24h", "本轮动作摘要"],
        [[row.get("task", ""), row.get("cron", ""), row.get("last_success") or "n/a",
          "✓" if row.get("recent") else "✗",
          "，".join(f"{k}={v}" for k, v in (row.get("actions") or {}).items()) or "无动作"]
         for row in night_rows],
    ))
    lines.append(f"- pending_memories：**{graph['pending_memories']}**；drafts/pending：**{graph['draft_pending_files']}**")
    typed = graph.get("typed_edges") or {}
    typed_s = "，".join(f"{k}={typed.get(k, 0)}" for k in ("SUPPORTED_BY", "GROUNDED_IN", "EVOKES", "updates"))
    typed_d = graph.get("typed_edges_delta") or {}
    delta_s = "；变动：" + "，".join(f"{k}{'+' if v > 0 else ''}{v}" for k, v in sorted(typed_d.items())) if typed_d else ""
    lines.append(f"- legacy typed 边（只读）：{typed_s}{delta_s}")
    lines.append(table(["draft box", "count"], [[k, v] for k, v in graph["draft_counts"].items()]))

    dual = graph.get("dual_edges") or {}
    lines.append("\n### 3.2 边统计（双表）")
    flow_parts = "，".join(
        f"{k}: {v}" for k, v in sorted((dual.get("flow_by_provenance") or {}).items())
    ) or "无"
    lines.append(f"- flow_edges：**{dual.get('flow_total', 0)}**（{flow_parts}）")
    semantic_parts = []
    for role, item in sorted((dual.get("semantic_breakdown") or {}).items()):
        states = "，".join(f"{k}: {v}" for k, v in sorted((item.get("review_states") or {}).items()))
        semantic_parts.append(f"{role}: {item.get('total', 0)}" + (f" [{states}]" if states else ""))
    lines.append(f"- semantic_edges：**{dual.get('semantic_total', 0)}**（{'；'.join(semantic_parts) or '无'}）")
    lines.append(f"- legacy_edges：**{dual.get('legacy_total', 0)}**（只读，不再增长）")
    if dual.get("migration_gap"):
        lines.append(f"- **warning：双表比 legacy 少 {dual['migration_gap']} 条，存在迁移缺口。**")
    elif dual.get("growth_over_legacy"):
        lines.append(f"- 双表已较 legacy 正常增长：**+{dual['growth_over_legacy']}** 条。")
    else:
        lines.append("- 双表与 legacy 基线守恒。")

    kv2 = graph.get("kuzu_v2") or {}
    lines.append("\n### 3.3 Kuzu 一致性")
    if kv2.get("available"):
        lines.append(
            f"- Kuzu FlowEdge vs SQLite flow_edges："
            f"**{kv2.get('flow_edges')}/{kv2.get('sqlite_flow_edges')}** "
            f"{'✓' if kv2.get('flow_consistent') else '✗'}"
        )
        lines.append(
            f"- Kuzu SemanticEdge vs SQLite semantic_edges："
            f"**{kv2.get('semantic_edges')}/{kv2.get('sqlite_semantic_edges')}** "
            f"{'✓' if kv2.get('semantic_consistent') else '✗'}"
        )
        lines.append(
            f"- Kuzu EDGE (legacy) vs SQLite edges："
            f"**{kv2.get('legacy_edges')}/{kv2.get('sqlite_legacy_edges')}** "
            f"{'✓' if kv2.get('legacy_consistent') else '✗'}（只读）"
        )
    else:
        lines.append(f"- Kuzu 临时快照不可用：{first_line(kv2.get('error') or 'unknown', 180)}")
    lines.append(
        f"- outbox_v2 pending：**{kv2.get('outbox_v2_pending', 'n/a')}**；"
        f"legacy edge outbox pending：**{kv2.get('legacy_edge_outbox_pending', 'n/a')}**"
    )

    integ = graph.get("integrate_24h") or {}
    lines.append("\n### 3.4 过去24h integrate统计")
    lines.append(f"- 新记忆：**{integ.get('new_memories', 0)}** 条")
    lines.append(
        f"- 自动建弱边(flow)：**{integ.get('auto_flow_edges', 0)}** 条"
        f"（integrated 事件记录全部 flow：{integ.get('flow_edges_event_total', 0)}）"
    )
    lines.append(
        f"- 自动建语义边(semantic)：**{integ.get('auto_semantic_edges', 0)}** 条"
        f"（integrated 事件记录全部 semantic：{integ.get('semantic_edges_event_total', 0)}）"
    )
    lines.append(f"- review 候选：**{integ.get('review_proposals', 0)}** 条")
    if integ.get("degree_cap_hits") is None:
        lines.append("- degree_cap 命中：事件格式暂未提供该字段，跳过且不报错。")
    else:
        nodes = "，".join(integ.get("degree_cap_nodes") or []) or "未记录节点"
        lines.append(f"- degree_cap 命中：**{integ.get('degree_cap_hits', 0)}** 次（{nodes}）")
    if integ.get("parse_errors"):
        lines.append(f"- integrated 事件解析失败：{integ['parse_errors']} 条。")

    lines.append("\n### 3.5 drafts")
    for key in ("pending", "done", "skipped", "archived"):
        trend = (graph.get("draft_trend") or {}).get(key, {})
        before = trend.get("previous")
        current = trend.get("current", 0)
        arrow = f"{before} → {current}" if isinstance(before, int) else f"{current}（首次趋势基线）"
        lines.append(f"- {key}：**{arrow}**")

    cross = graph.get("cross_layer_flow") or {}
    test = cross.get("activation_test") or {}
    lines.append("\n### 3.6 跨层 flow_edges")
    lines.append(
        f"- 跨层边总数：**{cross.get('total', 0)}** 条；"
        f"满足 activate 门槛：**{cross.get('eligible_for_activate', 0)}** 条"
    )
    if test.get("ok"):
        sample = test.get("sample") or {}
        lines.append(
            f"- 跨层边传热验证：**✓** `{sample.get('source_id')}` "
            f"({sample.get('source_level')}) → `{sample.get('target_id')}` "
            f"({sample.get('target_level')})；隔离副本 activation +{test.get('delta', 0):.4f}。"
        )
    else:
        lines.append(f"- 跨层边传热验证：**✗ {first_line(test.get('error') or 'unknown', 180)}**")

    beliefs = data["beliefs"]
    lines.append("\n## 4. belief graph")
    if beliefs.get("error"):
        lines.append(f"读取失败：{beliefs['error']}")
    else:
        lines.append(f"- 骨头总数：**{beliefs['total']}**")
        if beliefs.get("graph"):
            g = beliefs["graph"]
            lines.append(
                f"- 图节点/边：Belief **{g['beliefs']}**；BeliefCase **{g['cases']}** "
                f"（Memory引用 {g['memory_cases']} / inline {g['inline_cases']}）；"
                f"CONSTELLATES **{g['constellations']}**；outbox **{g['outbox']}**。"
            )
            lines.append(f"- 非法 CONSTELLATES：**{g['invalid_constellations']}**")
        lines.append(f"- 7天内有新 case：**{len(beliefs['new_case_7d'])}**")
        lines.append(table(["id", "new_cases", "statement"], [[b["id"], b["new_cases"], first_line(b["statement"], 100)] for b in beliefs["new_case_7d"]]))
        if beliefs.get("confidence_field"):
            lines.append(f"- confidence 有变动：**{len(beliefs['confidence_changed'])}**")
            lines.append(table(["id", "before", "after"], [[b["id"], b["before"], b["after"]] for b in beliefs["confidence_changed"]]))
        else:
            lines.append("- confidence 有变动：beliefs.json 当前没有 confidence 字段，先只做字段存在性和未来基线监测。")
        lines.append(f"- 30天没 touch 的僵骨：**{len(beliefs['stale_30d'])}**")
        lines.append(table(["id", "last_touched", "statement"], [[b["id"], b["last_touched"], first_line(b["statement"], 100)] for b in beliefs["stale_30d"]]))
        lines.append(f"- active 但低于路由门：**{len(beliefs.get('non_routing_active', []))}**")
        lines.append(table(["id", "confidence", "statement"], [[b["id"], b["confidence"], first_line(b["statement"], 100)] for b in beliefs.get("non_routing_active", [])]))

    theseus = data.get("theseus") or {}
    lines.append("\n## 5. Theseus 健康")
    if not theseus.get("available"):
        lines.append(f"- **读取失败：{first_line(theseus.get('error') or 'unknown', 180)}**")
    else:
        evokes = theseus.get("evokes") or {}
        pass_rate = ("n/a" if evokes.get("pass_rate") is None else f"{evokes['pass_rate']:.1f}%")
        lines.append(
            f"- 种子总数：**{theseus.get('seed_total', 0)}**；最近7天新增："
            f"**{theseus.get('new_seeds_7d', 0)}**"
        )
        lines.append(
            f"- shadow chunks：**{theseus.get('shadow_chunks', 0)}**；其中入专用索引："
            f"**{theseus.get('indexed_chunks', 0)}**；覆盖父条：**{theseus.get('shadow_parents', 0)}**"
        )
        lines.append(
            f"- EVOKES 已裁决通过率：**{pass_rate}**（approved {evokes.get('approved', 0)} / "
            f"decided {evokes.get('decided', 0)}；rejected {evokes.get('rejected', 0)}；"
            f"pending {evokes.get('pending', 0)}）"
        )

    svc = data["services"]
    lines.append("\n## 6. 服务与资源")
    lines.append(table(["name", "unit", "active", "昨日重启次数", "last_log"], [[s["name"], s["unit"], s["active"], s.get("yesterday_restarts") if s.get("yesterday_restarts") is not None else "n/a", s.get("last_log", "")] for s in svc["services"]]))
    lines.append(table(["file", "size", "日增速"], [[name, fmt_bytes(v.get("size")), "首次建立基线" if v.get("delta") is None else fmt_bytes(v.get("delta"))] for name, v in svc["files"].items()]))
    lines.append(f"- backup 最后 OK：{svc.get('backup_last_ok') or '未找到 OK 行'}")

    recn = data["reconcile"]
    ratio_s = "n/a" if recn["ratio"] is None else f"{recn['ratio']:.2%}"
    lines.append("\n## 7. 溯源对账")
    lines.append(f"- 昨日聊天消息数（{', '.join(CHAT_KINDS)}）：**{recn['chat_count']}**")
    lines.append(f"- 昨日新增记忆数：**{recn['new_memories']}**")
    lines.append(f"- 记忆/聊天比率：**{ratio_s}**")
    if recn.get("alert"):
        lines.append(f"- **异常：{recn['alert']}**")
    lines.append(table(["kind", "direction", "count"], [[r["kind"], r["direction"], r["count"]] for r in recn["distribution"]]))

    lines.append("\n## 我建议额外维护")
    lines.append("- belief confidence 已由 SQLite 的 Memory/BeliefCase 实时重算并保存日报基线；不再要求把派生值写回 beliefs.json。")
    lines.append("- 给 trace 里的 `recently_injected` 拒绝保留结构化字段到 rejected item；目前日报已兼容理由文本，但结构化计数更稳。")
    return "\n".join(lines) + "\n"


def chown_report_files(paths: list[Path]) -> None:
    try:
        uid = pwd.getpwnam("local-user").pw_uid
        gid = grp.getgrnam("agent-body").gr_gid
    except Exception:
        return
    for path in paths:
        try:
            os.chown(path, uid, gid)
        except Exception:
            pass


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    today = now_local()
    today_start = today.replace(hour=0, minute=0, second=0, microsecond=0)
    yesterday_start = today_start - timedelta(days=1)
    state = read_state()

    data: dict[str, Any] = {
        "report_date": compact_ymd(today),
        "generated_at": today.isoformat(timespec="seconds"),
        "yesterday": ymd(yesterday_start),
        "tag_health": collect_tag_health(today, yesterday_start),
        "recall": collect_recall(today),
        "graph": collect_graph(today, state),
        "beliefs": collect_beliefs(today, state),
        "review_queue": collect_review_queue(today),
        "theseus": collect_theseus(today),
        "services": collect_services(today, yesterday_start, today_start, state),
        "reconcile": collect_reconcile(yesterday_start),
    }
    alerts: list[str] = []
    if data["tag_health"]["leaks"]:
        alerts.append(f"tag 漏网 {len(data['tag_health']['leaks'])} 条")
    current_old = sum(1 for r in data["tag_health"]["currents"] if r.get("over_14"))
    if current_old:
        alerts.append(f"state:current 超 14 天 {current_old} 条，需要夜批复审")
    if data["graph"]["dream"].get("due") and not data["graph"]["dream"].get("ran_since_midnight"):
        alerts.append("今晨 dream_pass 未确认跑过")
    if not (data["graph"].get("night_batches") or {}).get("all_recent"):
        alerts.append("夜批统一看板存在近24h未成功任务")
    graph_health = data["graph"].get("health") or {}
    if data["graph"].get("source") != "kuzu":
        alerts.append("Kuzu 图健康快照不可用，日报已回退 SQLite 镜像")
    elif not graph_health.get("consistent"):
        alerts.append("Kuzu 与 SQLite 图镜像计数或 outbox 不一致")
    dual = data["graph"].get("dual_edges") or {}
    if dual.get("migration_gap"):
        alerts.append(f"双表迁移缺口 {dual['migration_gap']} 条")
    kuzu_v2 = data["graph"].get("kuzu_v2") or {}
    if not kuzu_v2.get("available"):
        alerts.append("Kuzu v2 关系计数快照不可用")
    elif not kuzu_v2.get("consistent"):
        alerts.append("Kuzu v2 双关系表与 SQLite 或 outbox 不一致")
    draft_pending = ((data["graph"].get("draft_counts") or {}).get("pending") or 0)
    if draft_pending > 100:
        alerts.append(f"drafts pending {draft_pending} 条，超过 100")
    cross_test = ((data["graph"].get("cross_layer_flow") or {}).get("activation_test") or {})
    if not cross_test.get("ok"):
        alerts.append("热流合同隔离验证失败（弱边/conductance/幂等/单日衰减/briefing零写）")
    if data["reconcile"].get("alert"):
        alerts.append(data["reconcile"]["alert"])
    if not data["review_queue"].get("available"):
        alerts.append("review queue unavailable")
    if not data["theseus"].get("available"):
        alerts.append("Theseus 健康指标不可用")
    for s in data["services"]["services"]:
        if s["name"] != "scheduler" and s["active"] not in ("active",):
            alerts.append(f"服务 {s['name']} 状态 {s['active']}")
    data["alerts"] = alerts
    data = sanitize_obj(data)

    report_path = REPORT_DIR / f"daily-{compact_ymd(today)}.md"
    json_path = REPORT_DIR / f"daily-{compact_ymd(today)}.json"
    report_path.write_text(render_report(data), encoding="utf-8")
    json_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    latest = REPORT_DIR / "daily-latest.md"
    latest.write_text(report_path.read_text(encoding="utf-8"), encoding="utf-8")

    new_state = {
        "updated_at": today.isoformat(timespec="seconds"),
        "graph": {
            "edge_total": data["graph"]["edge_total"],
            "dual_edges": {"total": data["graph"].get("dual_edges", {}).get("total", 0)},
            "draft_counts": data["graph"].get("draft_counts", {}),
        },
        "files": {name: {"size": v.get("size")} for name, v in data["services"]["files"].items()},
        "beliefs": {"confidence": data["beliefs"].get("confidence", {}) if isinstance(data.get("beliefs"), dict) else {}},
    }
    write_state(new_state)
    chown_report_files([report_path, json_path, latest, STATE_PATH])
    print(str(report_path))


if __name__ == "__main__":
    main()
