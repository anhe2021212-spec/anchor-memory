"""Anchor W1: updates proposals and agent-owned review queue."""
from __future__ import annotations
import argparse, contextlib, datetime as dt, fcntl, hashlib, json, os, re, sqlite3, tempfile
from pathlib import Path
from release_config import AnchorConfig


_CONFIG = AnchorConfig.load()
DATA = _CONFIG.data_dir
DB_PATH = _CONFIG.db_path
CHROMA_PATH = _CONFIG.chroma_dir or (DATA / "chroma")
QUEUE_PATH = Path(os.environ.get("ANCHOR_UPDATE_REVIEW_QUEUE", _CONFIG.review_dir / "update_review_queue.json"))
LOCK_PATH = Path(str(QUEUE_PATH) + ".lock")
MAX_HOPS = 5
DONE_RE = re.compile(r"已完成|完成了|做完了|修好了|已修复|已上线|上线完成|已关闭|已解决|跑通|成功落地")
REPLACE_RE = re.compile(r"取代|替代|不再|下岗|退役|作废|废弃|停用|已迁移|搬到|现役|当前现状")
OPEN_RE = re.compile(r"待做|还要|准备|计划|草稿|未完成|没完成|正在|等待|待定|TODO|todo|下一步")
ACTION_OPEN = {"action:todo", "action:waiting", "action:ongoing"}

def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()

def pid(kind, *parts):
    raw = "|".join([kind] + [str(x) for x in parts])
    return "up_" + hashlib.sha256(raw.encode()).hexdigest()[:16]

def summary(text, limit=260):
    clean = " ".join((text or "").split())
    return clean[:limit] + ("…" if len(clean) > limit else "")

def empty_queue():
    return {"version": 1, "updated_at": now(), "proposals": []}

def read_unlocked():
    if not QUEUE_PATH.exists():
        return empty_queue()
    data = json.loads(QUEUE_PATH.read_text())
    if not isinstance(data, dict) or not isinstance(data.get("proposals"), list):
        raise RuntimeError("invalid update review queue")
    return data

def write_unlocked(data):
    QUEUE_PATH.parent.mkdir(parents=True, exist_ok=True)
    data["version"], data["updated_at"] = 1, now()
    fd, name = tempfile.mkstemp(prefix=QUEUE_PATH.name + ".", suffix=".tmp", dir=str(QUEUE_PATH.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as out:
            json.dump(data, out, ensure_ascii=False, indent=2)
            out.write("\n")
            out.flush()
            os.fsync(out.fileno())
        # Keep the queue private from other users while preserving the
        # directory's local-user/agent-runtime default ACL for dashboard review.
        os.chmod(name, 0o660)
        os.replace(name, QUEUE_PATH)
    finally:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(name)

@contextlib.contextmanager
def queue_lock():
    LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOCK_PATH, "a+", encoding="utf-8") as lock:
        with contextlib.suppress(PermissionError):
            os.chmod(LOCK_PATH, 0o660)
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

def enqueue(items):
    with queue_lock():
        data = read_unlocked()
        known = {x.get("id") for x in data["proposals"]}
        added = 0
        for raw in items:
            if raw["id"] in known:
                continue
            item = dict(raw)
            item.setdefault("status", "pending")
            item.setdefault("created_at", now())
            data["proposals"].append(item)
            known.add(item["id"])
            added += 1
        if added or not QUEUE_PATH.exists():
            write_unlocked(data)
        pending = sum(x.get("status") == "pending" for x in data["proposals"])
    return {"discovered": len(items), "enqueued": added, "pending": pending}

def list_proposals(status="pending", limit=20):
    with queue_lock():
        rows = list(read_unlocked()["proposals"])
    if status and status != "all":
        rows = [x for x in rows if x.get("status") == status]
    rows.sort(key=lambda x: x.get("created_at", ""), reverse=True)
    rows.sort(key=lambda x: x.get("priority") == "low")
    return rows[:max(1, min(int(limit), 10000))]

def set_status(proposal_id, expected, **updates):
    with queue_lock():
        data = read_unlocked()
        for item in data["proposals"]:
            if item.get("id") != proposal_id:
                continue
            if item.get("status") not in expected:
                raise RuntimeError(f"{proposal_id} is {item.get('status')}")
            item.update(updates)
            write_unlocked(data)
            return dict(item)
    raise KeyError(f"proposal not found: {proposal_id}")

def propose_level_change(memory_id, new_level, reason, db_path=DB_PATH):
    """Enqueue a reviewed/audited level correction; never mutates the memory."""
    if os.environ.get("ANCHOR_LEVEL_REVIEW", "on").strip().lower() == "off":
        raise RuntimeError("level correction review is disabled")
    memory_id = (memory_id or "").strip()
    new_level = (new_level or "").strip().lower()
    reason = (reason or "").strip()
    if new_level not in {"raw", "understanding", "cognition"}:
        raise ValueError("new_level must be raw, understanding, or cognition")
    if not reason:
        raise ValueError("reason is required")
    conn = sqlite3.connect(str(db_path), timeout=10)
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        "SELECT memory_id,text,COALESCE(level,'raw') level,"
        "COALESCE(collection,'') collection FROM memories WHERE memory_id=?",
        (memory_id,),
    ).fetchone()
    if not row:
        conn.close()
        raise KeyError(f"memory not found: {memory_id}")
    if row["collection"] == "wenku":
        conn.close()
        raise ValueError("wenku entries do not use Anchor semantic levels")
    impacted = conn.execute(
        "SELECT COUNT(*) count FROM ("
        "SELECT source_id,target_id FROM flow_edges WHERE source_id=? OR target_id=? "
        "UNION ALL SELECT source_id,target_id FROM semantic_edges WHERE source_id=? OR target_id=?"
        ")", (memory_id, memory_id, memory_id, memory_id),
    ).fetchone()["count"]
    conn.close()
    item = {
        "id": pid("set_level", memory_id, row["level"], new_level),
        "kind": "set_level", "risk": "review_only",
        "memory_id": memory_id, "old_level": row["level"],
        "new_level": new_level, "incident_edges": impacted,
        "memory_summary": summary(row["text"] or "", 700),
        "reason": summary(reason, 500), "source": "explicit_level_audit",
    }
    return {**enqueue([item]), "proposal_id": item["id"], "proposal": item}

def propose_evokes(source_id, target_id, reason, db_path=DB_PATH):
    """Enqueue one anchor-to-wenku EVOKES edge; never writes the edge."""
    source_id, target_id = (source_id or "").strip(), (target_id or "").strip()
    reason = (reason or "").strip()
    if not reason:
        raise ValueError("reason is required")
    conn = sqlite3.connect(str(db_path), timeout=10)
    conn.row_factory = sqlite3.Row
    rows = {r["memory_id"]: dict(r) for r in conn.execute(
        "SELECT memory_id,text,COALESCE(level,'raw') level,"
        "COALESCE(collection,'') collection FROM memories WHERE memory_id IN (?,?)",
        (source_id, target_id),
    )}
    edge = conn.execute(
        "SELECT role FROM semantic_edges WHERE source_id=? AND target_id=? "
        "AND role='EVOKES'", (source_id, target_id),
    ).fetchone()
    conn.close()
    source, target = rows.get(source_id), rows.get(target_id)
    if not source or not target:
        raise KeyError("EVOKES endpoint missing")
    if source["collection"] == "wenku" or source["level"] not in {"raw", "understanding", "cognition"}:
        raise ValueError("EVOKES source must be an Anchor memory")
    if target["collection"] != "wenku":
        raise ValueError("EVOKES target must be a wenku entry")
    if edge:
        raise ValueError("EVOKES edge already exists")
    item = {
        "id": pid("evokes", source_id, target_id),
        "kind": "evokes", "risk": "review_only",
        "source_id": source_id, "target_id": target_id,
        "edge_type": "EVOKES", "weight": 1.0,
        "replace_legacy": False, "occupied_edge_type": None,
        "source_summary": summary(source["text"] or "", 700),
        "target_summary": summary(target["text"] or "", 700),
        "reason": summary(reason, 500), "source": "explicit_evokes_review",
    }
    return {**enqueue([item]), "proposal_id": item["id"], "proposal": item}

def decide(memory, proposal_id, decision, note=""):
    decision = (decision or "").strip().lower()
    if decision not in {"approve", "reject"}:
        raise ValueError("decision must be approve or reject")
    if decision == "reject":
        item = set_status(proposal_id, {"pending"}, status="rejected", decided_at=now(), review_note=note[:500])
        return {"ok": True, "id": proposal_id, "status": item["status"]}
    item = set_status(proposal_id, {"pending"}, status="applying", apply_started_at=now(), review_note=note[:500])
    try:
        if item["kind"] in {"supersede", "structure_overlong"}:
            result = memory.mark_update(item["new_id"], item["old_id"])
            if not result.get("ok"):
                raise RuntimeError(result.get("reason") or "mark_update rejected")
        elif item["kind"] == "structure_cycle":
            if not memory.db.remove_update_edge(item["source_id"], item["target_id"]):
                raise RuntimeError("updates edge missing or changed")
            result = {"ok": True, "removed": {"source_id": item["source_id"], "target_id": item["target_id"]}}
        elif item["kind"] == "grounded_in":
            reason = "GROUNDED_IN is retained read-only in dual-edge mode"
            set_status(
                proposal_id, {"applying"}, status="rejected",
                decided_at=now(), last_error="", rejection_reason=reason,
                action_result={"ok": False, "reason": reason},
            )
            return {"ok": False, "id": proposal_id, "status": "rejected",
                    "reason": reason}
        elif item["kind"] in {"supported_by", "evokes"}:
            edge_type = {
                "supported_by": "SUPPORTED_BY",
                "evokes": "EVOKES",
            }[item["kind"]]
            result = memory.write_typed_edge(
                item["source_id"], item["target_id"], edge_type,
                weight=float(item.get("weight", 1.0)),
                replace_legacy=bool(item.get("replace_legacy")),
                audit_note=f"review:{proposal_id}",
            )
        elif item["kind"] == "set_level":
            if os.environ.get("ANCHOR_LEVEL_REVIEW", "on").strip().lower() == "off":
                raise RuntimeError("level correction review is disabled")
            current = memory.db.get(item["memory_id"])
            if not current:
                raise RuntimeError("memory missing before level correction")
            if (current.get("level") or "raw") != item["old_level"]:
                raise RuntimeError("memory level changed since proposal")
            result = memory.set_level(item["memory_id"], item["new_level"])
        else:
            raise RuntimeError(f"unsupported proposal kind: {item['kind']}")
    except Exception as exc:
        set_status(proposal_id, {"applying"}, status="pending", last_error=f"{type(exc).__name__}: {str(exc)[:300]}", apply_failed_at=now())
        raise
    set_status(proposal_id, {"applying"}, status="approved", decided_at=now(), action_result=result, last_error="")
    return {"ok": True, "id": proposal_id, "status": "approved", "action_result": result}

def rows_edges(db_path=DB_PATH):
    conn = sqlite3.connect(str(db_path), timeout=10)
    conn.row_factory = sqlite3.Row
    rows = {r["memory_id"]: dict(r) for r in conn.execute(
        "SELECT memory_id,text,timestamp,tag,COALESCE(collection,'') collection FROM memories WHERE COALESCE(collection,'') != 'wenku'"
    )}
    edges = [dict(r) for r in conn.execute(
        "SELECT source_id,target_id,weight,created,last_fired,'flow' edge_type FROM flow_edges "
        "UNION ALL SELECT source_id,target_id,strength weight,created,created last_fired,role edge_type "
        "FROM semantic_edges"
    )]
    conn.close()
    return rows, edges

def tags(row):
    return {x.strip() for x in (row.get("tag") or "").split(",") if x.strip()}

def domains(row):
    return {x for x in tags(row) if x.startswith("domain:")}

def load_env():
    configured = os.environ.get("ANCHOR_RUNTIME_ENV_FILE", "").strip()
    if not configured:
        return
    path = Path(configured)
    if not path.exists():
        return
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.strip().strip('"').strip("'")
        if key.strip() and value:
            os.environ.setdefault(key.strip(), value)

def load_embeddings(chroma_path=CHROMA_PATH):
    load_env()
    import chromadb
    provider = os.environ.get("ANCHOR_EMBED_PROVIDER", "voyage").strip().lower()
    if provider == "voyage":
        suffix = os.environ.get("VOYAGE_COLLECTION_SUFFIX", "voyage4_1024")
        name = os.environ.get("ANCHOR_CHROMA_COLLECTION", f"memories_{suffix}")
    else:
        name = os.environ.get("ANCHOR_CHROMA_COLLECTION", "memories")
    col = chromadb.PersistentClient(path=str(chroma_path)).get_collection(name)
    data = col.get(include=["embeddings"])
    emb = data.get("embeddings")
    return {} if emb is None else dict(zip(data.get("ids") or [], emb))

def deterministic_candidates(rows, edges, embeddings, min_similarity=.74, max_candidates=80, lookback_days=0):
    import numpy as np
    ids = [mid for mid in rows if mid in embeddings]
    if not ids:
        return []
    matrix = np.asarray([embeddings[mid] for mid in ids], dtype=np.float32)
    matrix /= np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-9)
    pos = {mid: i for i, mid in enumerate(ids)}
    adjacency = {mid: set() for mid in rows}
    existing_targets = set()
    for edge in edges:
        a, b = edge["source_id"], edge["target_id"]
        if edge["edge_type"] == "updates":
            existing_targets.add(b)
        elif a in adjacency and b in adjacency:
            adjacency[a].add(b)
            adjacency[b].add(a)
    cutoff = dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=lookback_days) if lookback_days else None
    found = []
    for newer in rows.values():
        nid, text = newer["memory_id"], newer.get("text") or ""
        if nid not in pos:
            continue
        if cutoff:
            try:
                stamp = dt.datetime.fromisoformat(newer["timestamp"].replace("Z", "+00:00"))
                if stamp.tzinfo is None:
                    stamp = stamp.replace(tzinfo=dt.timezone.utc)
                if stamp < cutoff:
                    continue
            except (TypeError, ValueError):
                continue
        nt = tags(newer)
        completed, replaced, done = bool(DONE_RE.search(text)), bool(REPLACE_RE.search(text)), "action:done" in nt
        if not (completed or replaced or done):
            continue
        sims, per_new = matrix @ matrix[pos[nid]], 0
        for index in np.argsort(-sims):
            oid = ids[int(index)]
            old = rows[oid]
            similarity = float(sims[int(index)])
            if similarity < min_similarity:
                break
            if oid == nid or (old.get("timestamp") or "") >= (newer.get("timestamp") or "") or oid in existing_targets:
                continue
            ot = tags(old)
            action = done and bool(ot & ACTION_OPEN)
            lexical = completed and bool(OPEN_RE.search(old.get("text") or ""))
            if not (replaced or action or lexical):
                continue
            if not (domains(newer) & domains(old) or action):
                continue
            direct = oid in adjacency.get(nid, set())
            shared = len(adjacency.get(nid, set()) & adjacency.get(oid, set()))
            if not (direct or shared or similarity >= .84):
                continue
            signals = []
            if replaced: signals.append("replacement_phrase")
            if action: signals.append("action:done→open")
            if lexical: signals.append("completion→open")
            found.append({
                "candidate_id": pid("candidate", nid, oid), "new_id": nid, "old_id": oid,
                "similarity": round(similarity, 4), "signals": signals,
                "locality": "direct_edge" if direct else f"shared_neighbors:{shared}" if shared else "high_similarity",
                "new_time": newer.get("timestamp"), "old_time": old.get("timestamp"),
                "new_tag": newer.get("tag") or "", "old_tag": old.get("tag") or "",
                "new_summary": summary(text, 700), "old_summary": summary(old.get("text") or "", 700),
            })
            per_new += 1
            if per_new >= 2:
                break
    found.sort(key=lambda x: ("action:done→open" in x["signals"], x["locality"] == "direct_edge", x["similarity"]), reverse=True)
    return found[:max_candidates]

LLM_PROMPT = """你是 Anchor updates 关系的agent 自主复核优先级判断器。
判断新版记忆是否使旧版记忆的事实或状态过期，不是找相关、续篇或同主题。
强阳性：同一方案从想法走到最终实现；正文明确写最终、不再、完全重写、已迁移或已取代；同一 TODO 被明确做完。
强阴性：连续编号或系列记录；互补信息；同一事件的重复描述；只完成了一部分；抽象理解覆盖原始经历；仅措辞相似。
你的输出只影响agent 自主审批队列优先级，不能自动落边，也不能让确定性候选消失。
输出严格 JSON 数组；只列高置信阳性，每项格式：
{"candidate_id":"...","reason":"不超过80字","confidence":0到1}
只返回 confidence>=0.80 的项目；其余返回空数组。"""

def proposal_call(messages, max_tokens=1600):
    """W1 专用非思考调用；失败显式报错，绝不伪装成零提案。"""
    import httpx
    from taxonomy_tagger import _routes
    load_env()
    errors = []
    for url, key, model in _routes():
        endpoint = url.rstrip("/")
        if not endpoint.endswith("/chat/completions"):
            endpoint += "/chat/completions"
        body = {
            "model": model, "messages": messages, "temperature": 0,
            "max_tokens": max_tokens, "stream": False,
            "thinking": {"type": "disabled"},
        }
        try:
            with httpx.Client(timeout=45, trust_env=True) as client:
                response = client.post(
                    endpoint, headers={"Authorization": f"Bearer {key}"}, json=body
                )
            response.raise_for_status()
            content = ((((response.json().get("choices") or [{}])[0].get("message") or {})
                        .get("content") or "").strip())
            if content:
                return content
            errors.append(f"{model}:empty")
        except Exception as exc:
            errors.append(f"{model}:{type(exc).__name__}")
    raise RuntimeError("proposal LLM unavailable: " + ",".join(errors))


def parse_array(raw):
    text = (raw or "").strip()
    fence = chr(96) * 3
    if text.startswith(fence):
        text = text[len(fence):].lstrip()
        if text.startswith("json"):
            text = text[4:].lstrip()
        if text.endswith(fence):
            text = text[:-len(fence)].rstrip()
    start, end = text.find("["), text.rfind("]")
    if start < 0 or end < start:
        return []
    try:
        data = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return []
    return data if isinstance(data, list) else []

def llm_filter(candidates, call=None, batch_size=10):
    """Prioritize deterministic candidates; never use the LLM as a drop gate."""
    if not candidates:
        return []
    if call is None:
        call = proposal_call
    accepted = {}
    for offset in range(0, len(candidates), batch_size):
        batch = candidates[offset:offset + batch_size]
        raw = call([
            {"role": "system", "content": LLM_PROMPT},
            {"role": "user", "content": json.dumps(batch, ensure_ascii=False)},
        ], max_tokens=1600)
        for item in parse_array(raw):
            try:
                confidence = float(item.get("confidence", 0))
            except (TypeError, ValueError):
                continue
            if item.get("candidate_id") and confidence >= .80:
                accepted[item["candidate_id"]] = (
                    summary(str(item.get("reason") or ""), 120), round(confidence, 3)
                )
    out = []
    for candidate in candidates:
        verdict = accepted.get(candidate["candidate_id"])
        prioritized = verdict is not None
        out.append({
            "id": pid("supersede", candidate["new_id"], candidate["old_id"]),
            "kind": "supersede", "risk": "review_only",
            "priority": "normal" if prioritized else "low",
            "new_id": candidate["new_id"], "old_id": candidate["old_id"],
            "new_time": candidate["new_time"], "old_time": candidate["old_time"],
            "new_tag": candidate["new_tag"], "old_tag": candidate["old_tag"],
            "new_summary": summary(candidate["new_summary"]),
            "old_summary": summary(candidate["old_summary"]),
            "similarity": candidate["similarity"], "signals": candidate["signals"],
            "locality": candidate["locality"],
            "reason": verdict[0] if prioritized else
                      "确定性候选；LLM未判为高置信阳性，保留低优先级agent 自主复核",
            "llm_confidence": verdict[1] if prioritized else 0.0,
            "llm_rejected": not prioritized,
            "source": "heuristic+llm_proposal" if prioritized else
                      "heuristic+llm_rejected_review",
        })
    return out

def tarjan(nodes, successors):
    index, stack, on_stack, indices, low, components = 0, [], set(), {}, {}, []
    def visit(node):
        nonlocal index
        indices[node] = low[node] = index
        index += 1
        stack.append(node)
        on_stack.add(node)
        for nxt in successors.get(node, []):
            if nxt not in indices:
                visit(nxt)
                low[node] = min(low[node], low[nxt])
            elif nxt in on_stack:
                low[node] = min(low[node], indices[nxt])
        if low[node] == indices[node]:
            component = set()
            while True:
                item = stack.pop()
                on_stack.remove(item)
                component.add(item)
                if item == node:
                    break
            components.append(component)
    for node in sorted(nodes):
        if node not in indices:
            visit(node)
    return components

def structure_proposals(rows, edges):
    updates = [e for e in edges if e["edge_type"] == "updates"]
    successors, edge_map = {}, {}
    for edge in updates:
        successors.setdefault(edge["target_id"], []).append(edge["source_id"])
        edge_map[(edge["source_id"], edge["target_id"])] = edge
    for old in successors:
        successors[old].sort(key=lambda mid: rows.get(mid, {}).get("timestamp") or "", reverse=True)
    nodes = {x for e in updates for x in (e["source_id"], e["target_id"])}
    out, cyclic = [], set()
    for component in tarjan(nodes, successors):
        self_loop = any(x in successors.get(x, []) for x in component)
        if len(component) == 1 and not self_loop:
            continue
        cyclic.update(component)
        inside = [e for e in updates if e["source_id"] in component and e["target_id"] in component]
        remove = max(inside, key=lambda e: (e.get("created") or "", e["source_id"]))
        src, tgt = remove["source_id"], remove["target_id"]
        out.append({
            "id": pid("structure_cycle", *sorted(component), src, tgt),
            "kind": "structure_cycle", "risk": "review_only",
            "source_id": src, "target_id": tgt, "component": sorted(component),
            "source_summary": summary(rows.get(src, {}).get("text") or ""),
            "target_summary": summary(rows.get(tgt, {}).get("text") or ""),
            "reason": f"updates 存量环；建议确认后删最后创建边 {src}→{tgt}",
            "source": "deterministic_structure_scan",
        })
    existing = set(edge_map)
    for start in sorted(nodes):
        if start in cyclic:
            continue
        path, seen, current = [start], {start}, start
        while successors.get(current):
            nxt = successors[current][0]
            if nxt in seen:
                break
            path.append(nxt)
            seen.add(nxt)
            current = nxt
        hops, latest = len(path) - 1, path[-1]
        if hops <= MAX_HOPS or (latest, start) in existing:
            continue
        out.append({
            "id": pid("structure_overlong", latest, start),
            "kind": "structure_overlong", "risk": "review_only",
            "new_id": latest, "old_id": start, "hops": hops, "path": path,
            "new_summary": summary(rows.get(latest, {}).get("text") or ""),
            "old_summary": summary(rows.get(start, {}).get("text") or ""),
            "reason": f"updates 链 {hops} 跳超过读取护栏 {MAX_HOPS}；建议确认后加 {latest}→{start} 快捷时效边",
            "source": "deterministic_structure_scan",
        })
    return out

def supported_by_proposals(db_path=DB_PATH):
    """Build review-only proposals from structured source_raw_ids annotations."""
    conn = sqlite3.connect(str(db_path), timeout=10)
    conn.row_factory = sqlite3.Row
    memories = {row["memory_id"]: dict(row) for row in conn.execute(
        "SELECT memory_id,text,COALESCE(level,'raw') level,"
        "COALESCE(collection,'') collection FROM memories"
    )}
    existing = {(row["source_id"], row["target_id"]): row["role"]
                for row in conn.execute(
                    "SELECT source_id,target_id,role FROM semantic_edges"
                )}
    out, seen = [], set()
    annotations = conn.execute("""
        SELECT a.annotation_id,a.memory_id,a.text
        FROM annotations a JOIN memories m ON m.memory_id=a.memory_id
        WHERE m.level='understanding' ORDER BY a.annotation_id
    """).fetchall()
    conn.close()
    for annotation in annotations:
        try:
            payload = json.loads(annotation["text"])
        except (TypeError, json.JSONDecodeError):
            continue
        raw_ids = payload.get("source_raw_ids") if isinstance(payload, dict) else None
        if not isinstance(raw_ids, list) or not raw_ids:
            continue
        source_id = annotation["memory_id"]
        source = memories.get(source_id) or {}
        if source.get("level") != "understanding" or source.get("collection") == "wenku":
            continue
        for raw_id in raw_ids:
            target_id = str(raw_id or "").strip()
            pair = (source_id, target_id)
            if not target_id or pair in seen:
                continue
            target = memories.get(target_id) or {}
            if target.get("level") != "raw" or target.get("collection") == "wenku":
                continue
            occupied = existing.get(pair)
            if occupied == "SUPPORTED_BY":
                continue
            if occupied:
                continue
            seen.add(pair)
            out.append({
                "id": pid("supported_by", source_id, target_id),
                "kind": "supported_by", "risk": "review_only",
                "source_id": source_id, "target_id": target_id,
                "edge_type": "SUPPORTED_BY", "weight": 1.0,
                "replace_legacy": False,
                "occupied_edge_type": occupied,
                "annotation_id": annotation["annotation_id"],
                "source_summary": summary(source.get("text") or "", 700),
                "target_summary": summary(target.get("text") or "", 700),
                "reason": "understanding 的结构化 source_raw_ids 第一人称来源；待珩逐边确认",
                "source": "structured_source_raw_ids",
            })
    return out


def scan_supported_by(db_path=DB_PATH, write_queue=True):
    proposals = supported_by_proposals(db_path)
    result = enqueue(proposals) if write_queue else {
        "discovered": len(proposals), "enqueued": 0, "pending": None
    }
    return {**result, "understandings": len({x["source_id"] for x in proposals}),
            "zero_auto_edges": True,
            "proposal_ids": [x["id"] for x in proposals]}


def evokes_proposals(db_path=DB_PATH, chroma_path=CHROMA_PATH, embeddings=None,
                     lookback_days=7, min_similarity=.72, max_per_source=2,
                     max_candidates=40):
    """Use the existing Theseus shadow collection to suggest EVOKES only."""
    import chromadb
    import theseus_shadow_index
    conn = sqlite3.connect(str(db_path), timeout=10)
    conn.row_factory = sqlite3.Row
    rows = {row["memory_id"]: dict(row) for row in conn.execute(
        "SELECT memory_id,text,timestamp,COALESCE(level,'raw') level,"
        "COALESCE(collection,'') collection FROM memories"
    )}
    existing = {(row["source_id"], row["target_id"]) for row in conn.execute(
        "SELECT source_id,target_id FROM semantic_edges WHERE role='EVOKES'"
    )}
    conn.close()
    embeddings = embeddings if embeddings is not None else load_embeddings(chroma_path)
    shadow_name = os.environ.get(
        "THESEUS_SHADOW_COLLECTION", "theseus_shadows_voyage4_1024"
    )
    shadow = chromadb.PersistentClient(path=str(chroma_path)).get_collection(shadow_name)
    cutoff = dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=max(0, int(lookback_days)))
    sources = []
    for row in rows.values():
        if row["collection"] == "wenku" or row["memory_id"] not in embeddings:
            continue
        try:
            stamp = dt.datetime.fromisoformat((row["timestamp"] or "").replace("Z", "+00:00"))
            if stamp.tzinfo is None:
                stamp = stamp.replace(tzinfo=dt.timezone.utc)
        except (TypeError, ValueError):
            continue
        if lookback_days and stamp < cutoff:
            continue
        sources.append(row)
    sources.sort(key=lambda row: (row.get("timestamp") or "", row["memory_id"]), reverse=True)
    proposals, seen = [], set()
    for source in sources:
        got = shadow.query(
            query_embeddings=[embeddings[source["memory_id"]]],
            n_results=max(4, max_per_source * 4),
            include=["metadatas", "distances"],
        )
        per_source = 0
        for meta, distance in zip((got.get("metadatas") or [[]])[0],
                                  (got.get("distances") or [[]])[0]):
            target_id = str((meta or {}).get("parent_memory_id") or "")
            target = rows.get(target_id)
            pair = (source["memory_id"], target_id)
            similarity = 1.0 - float(distance)
            if (not target or target["collection"] != "wenku" or pair in existing
                    or pair in seen or similarity < min_similarity):
                continue
            if (meta or {}).get("source_hash") != theseus_shadow_index.source_hash(target.get("text") or ""):
                continue
            seen.add(pair)
            proposals.append({
                "id": pid("evokes", *pair), "kind": "evokes", "risk": "review_only",
                "source_id": pair[0], "target_id": pair[1], "edge_type": "EVOKES",
                "weight": 1.0, "similarity": round(similarity, 4),
                "confidence": round(similarity, 4), "replace_legacy": False,
                "source_summary": summary(source.get("text") or "", 700),
                "target_summary": summary(target.get("text") or "", 700),
                "reason": "近期 Anchor 与 Theseus 专用影子高相似；仅供AI agent判断是否为精准唤起",
                "source": "theseus_shadow_voyage_review",
            })
            per_source += 1
            if per_source >= max_per_source or len(proposals) >= max_candidates:
                break
        if len(proposals) >= max_candidates:
            break
    return proposals


def scan(db_path=DB_PATH, chroma_path=CHROMA_PATH, lookback_days=7, max_candidates=80,
         min_similarity=.74, use_llm=True, call=None, write_queue=True):
    rows, edges = rows_edges(db_path)
    structural = structure_proposals(rows, edges)
    embeddings = load_embeddings(chroma_path)
    candidates = deterministic_candidates(
        rows, edges, embeddings,
        min_similarity=min_similarity, max_candidates=max_candidates,
        lookback_days=lookback_days,
    )
    semantic = llm_filter(candidates, call=call) if use_llm else []
    evokes_error = None
    try:
        evokes = evokes_proposals(
            db_path, chroma_path, embeddings=embeddings, lookback_days=lookback_days,
        )
    except Exception as exc:
        evokes, evokes_error = [], f"{type(exc).__name__}: {str(exc)[:200]}"
    proposals = structural + semantic + evokes
    result = enqueue(proposals) if write_queue else {
        "discovered": len(proposals), "enqueued": 0, "pending": None
    }
    return {
        **result, "memories_seen": len(rows),
        "deterministic_candidates": len(candidates),
        "semantic_proposals": len(semantic),
        "structure_proposals": len(structural),
        "updates_proposals": len(structural) + len(semantic),
        "evokes_proposals": len(evokes),
        "evokes_error": evokes_error,
        "zero_auto_edges": True,
        "proposal_ids": [x["id"] for x in proposals],
    }

def main():
    parser = argparse.ArgumentParser(description="Anchor updates proposal queue")
    sub = parser.add_subparsers(dest="command", required=True)
    pscan = sub.add_parser("scan")
    pscan.add_argument("--lookback-days", type=int, default=7)
    pscan.add_argument("--max-candidates", type=int, default=80)
    pscan.add_argument("--min-similarity", type=float, default=.74)
    pscan.add_argument("--no-llm", action="store_true")
    pscan.add_argument("--dry-run", action="store_true")
    psupported = sub.add_parser("supported-by-scan")
    psupported.add_argument("--dry-run", action="store_true")
    plist = sub.add_parser("list")
    plist.add_argument("--status", default="pending")
    plist.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()
    if args.command == "scan":
        result = scan(
            lookback_days=args.lookback_days, max_candidates=args.max_candidates,
            min_similarity=args.min_similarity, use_llm=not args.no_llm,
            write_queue=not args.dry_run,
        )
    elif args.command == "supported-by-scan":
        result = scan_supported_by(write_queue=not args.dry_run)
    else:
        result = list_proposals(args.status, args.limit)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
