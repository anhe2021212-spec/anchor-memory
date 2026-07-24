"""belief.py — Belief Graph M1 (2026-06-05)

设计对齐 工作台/belief-graph/设计稿_v0.2.md
铁律: 双通道 / 单向阀(本模块只读主库,永不写) / 低置信不决策 / boost不filter / case连接由 agent 标注

confidence 现算现得,不落盘:
    case权重 = emotion_score × recency(半衰期90天, 地板0.3)
    confidence = S / (S + C + k),  k=2
pinned 不绕过计算,只绕过 routing_cutoff —— 数字永远是真的,pin 只决定参与权。
"""
import json
import os
import datetime
import threading

import belief_graph
from release_config import AnchorConfig

PROMOTE_THRESHOLD = 0.40

_CONFIG = AnchorConfig.load()
BELIEF_PATH = os.environ.get(
    "ANCHOR_BELIEFS_PATH", str(_CONFIG.data_dir / "beliefs.json")
)
_DB = None
_LOCK = threading.RLock()

DEFAULT_PARAMS = {
    "half_life_days": 90.0,
    "recency_floor": 0.3,
    "prior_k": 2.0,
    "routing_cutoff": 0.40,
    "dormant_hint": 0.25,
    "top_n": 5,
}


def configure(db, migrate: bool = True):
    """Attach the SQLite belief repository after AnchorDB has initialized."""
    global _DB
    with _LOCK:
        belief_graph.ensure_schema(db)
        result = {"imported": False, "reason": "migration disabled"}
        if migrate and os.path.exists(BELIEF_PATH):
            with open(BELIEF_PATH, encoding="utf-8") as f:
                result = belief_graph.import_legacy(db, json.load(f))
        elif migrate:
            result = {"imported": False, "reason": "no legacy snapshot"}
        _DB = db
        if result.get("imported"):
            _write_snapshot(belief_graph.export_data(db, params(_load_json_fallback())))
        if db._kuzu_conn:
            belief_graph.drain(db)
        return result


def load():
    with _LOCK:
        if _DB is not None:
            return belief_graph.export_data(_DB, params(_load_json_fallback()))
        return _load_json_fallback()


def _load_json_fallback():
    if os.path.exists(BELIEF_PATH):
        with open(BELIEF_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {"params": dict(DEFAULT_PARAMS), "beliefs": []}


def save(data):
    with _LOCK:
        if _DB is not None:
            belief_graph.replace_data(_DB, data)
            data = belief_graph.export_data(_DB, params(data))
            if _DB._kuzu_conn:
                belief_graph.drain(_DB)
        _write_snapshot(data)


def _write_snapshot(data):
    os.makedirs(os.path.dirname(os.path.abspath(BELIEF_PATH)), exist_ok=True)
    tmp = BELIEF_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, BELIEF_PATH)


def params(data=None):
    p = dict(DEFAULT_PARAMS)
    if data and isinstance(data.get("params"), dict):
        p.update(data["params"])
    return p


def _case_weight(db, case, p):
    """单向阀: 只从主库读 emotion/timestamp,绝不写。"""
    memory_id = case.get("id") or case.get("memory_id")
    row = db.get(memory_id) if memory_id else None
    if not row:
        if memory_id:
            return 0.5 * p["recency_floor"]  # 旧引用异常时留残值,不清零
        emotion = float(case.get("emotion_score", 0.5))
        ts = str(case.get("occurred_at") or case.get("added") or "").replace("Z", "")
    else:
        emotion = row.get("emotion_score") or 0.5
        ts = (row.get("timestamp") or "").replace("Z", "")
    try:
        t = datetime.datetime.fromisoformat(ts)
        age_days = max(0.0, (datetime.datetime.now() - t).total_seconds() / 86400.0)
    except Exception:
        age_days = 0.0
    recency = max(p["recency_floor"], 0.5 ** (age_days / p["half_life_days"]))
    return emotion * recency


def confidence(db, belief, p=None):
    p = p or params()
    s = sum(_case_weight(db, c, p) for c in belief.get("support_cases", []))
    c = sum(_case_weight(db, c, p) for c in belief.get("contradiction_cases", []))
    return s / (s + c + p["prior_k"])


def get_belief(data, belief_id):
    for b in data.get("beliefs", []):
        if b.get("id") == belief_id:
            return b
    return None


def _audit_status_change(belief, action, old_status, new_status, reason, conf):
    stamp = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
    belief.setdefault("notes", []).append(
        f"[{stamp}] {action} {old_status}→{new_status}: {reason} (conf={conf:.3f})"
    )
    belief["updated_at"] = datetime.date.today().isoformat()


def promote(db, belief_id, reason):
    """Promote one reviewed candidate to active and append an audit note."""
    reason = (reason or "").strip()
    if not reason:
        raise ValueError("promote reason is required")
    data = load()
    belief = get_belief(data, (belief_id or "").strip())
    if not belief:
        raise KeyError(f"belief not found: {belief_id}")
    old_status = belief.get("status")
    if old_status != "candidate":
        raise ValueError(f"promote requires candidate status; current={old_status}")
    conf = confidence(db, belief, params(data))
    if conf < PROMOTE_THRESHOLD:
        raise ValueError(
            f"promote requires conf >= {PROMOTE_THRESHOLD:.2f}; current={conf:.3f}"
        )
    belief["status"] = "active"
    _audit_status_change(belief, "promote", old_status, "active", reason, conf)
    save(data)
    return {"ok": True, "belief_id": belief["id"], "old_status": old_status,
            "new_status": "active", "confidence": round(conf, 6)}


def demote(db, belief_id, reason):
    """Demote one unpinned active belief to candidate and append an audit note."""
    reason = (reason or "").strip()
    if not reason:
        raise ValueError("demote reason is required")
    data = load()
    belief = get_belief(data, (belief_id or "").strip())
    if not belief:
        raise KeyError(f"belief not found: {belief_id}")
    old_status = belief.get("status")
    if old_status != "active":
        raise ValueError(f"demote requires active status; current={old_status}")
    if belief.get("pinned"):
        raise ValueError("pinned belief cannot be demoted; unpin it first")
    conf = confidence(db, belief, params(data))
    belief["status"] = "candidate"
    _audit_status_change(belief, "demote", old_status, "candidate", reason, conf)
    save(data)
    return {"ok": True, "belief_id": belief["id"], "old_status": old_status,
            "new_status": "candidate", "confidence": round(conf, 6)}


def routing_set(db):
    """参与路由的 belief 列表(M4 boost 用): active 且 (pinned 或 conf>=cutoff)"""
    data = load()
    p = params(data)
    out = []
    for b in data.get("beliefs", []):
        if b.get("status") != "active":
            continue
        conf = confidence(db, b, p)
        if b.get("pinned") or conf >= p["routing_cutoff"]:
            out.append((b, conf))
    return out


def render_brief(db, top_n=None):
    """briefing 顶部渲染 v2 (2026-06-19): 两段, 避免"全量active每次定调"。
    概要段——所有 active 一行一条(id+conf+statement前20字), 给全局存在感但不展开。
    展开段——最近活跃 top N(默认3)保持旧格式: statement全文 + 最近case。
    "最近活跃"优先级: ①7天内有新case ②7天内被 belief_touch 命中(last_touched) ③不足按conf降序补齐。
    其余 belief 按需用 belief_touch / belief_get 调出。
    """
    data = load()
    p = params(data)
    expand_n = top_n or 3
    today = datetime.date.today()

    actives = []
    for b in data.get("beliefs", []):
        if b.get("status") != "active":
            continue
        actives.append((b, confidence(db, b, p)))
    if not actives:
        return ""

    def _age(date_str):
        """ISO 日期字符串 → 距今天数; 解析不了返回 None。"""
        if not date_str:
            return None
        try:
            d = datetime.date.fromisoformat(str(date_str)[:10])
        except Exception:
            return None
        return (today - d).days

    def _last_case_age(b):
        ages = []
        for k in ("support_cases", "contradiction_cases", "boundary_cases"):
            for c in b.get(k, []):
                a = _age(c.get("added"))
                if a is not None:
                    ages.append(a)
        return min(ages) if ages else None

    # 概要段: 全部 active, pinned 优先 + conf 降序
    lines = ["概要(全部 active):"]
    for b, conf in sorted(actives, key=lambda x: (not x[0].get("pinned", False), -x[1])):
        pin = "📌" if b.get("pinned") else ""
        stmt = b.get("statement", "")
        head = stmt[:20] + ("…" if len(stmt) > 20 else "")
        lines.append(f"  {pin}[{b['id']}|conf {conf:.2f}] {head}")

    # 展开段: 最近活跃分三档, 取前 expand_n
    WINDOW = 7
    tier1, tier2, rest = [], [], []
    for b, conf in actives:
        case_age = _last_case_age(b)
        touch_age = _age(b.get("last_touched"))
        if case_age is not None and case_age <= WINDOW:
            tier1.append((case_age, conf, b))
        elif touch_age is not None and touch_age <= WINDOW:
            tier2.append((touch_age, conf, b))
        else:
            rest.append((conf, b))
    tier1.sort(key=lambda x: (x[0], -x[1]))
    tier2.sort(key=lambda x: (x[0], -x[1]))
    rest.sort(key=lambda x: -x[0])
    ordered = ([(b, conf) for _, conf, b in tier1]
               + [(b, conf) for _, conf, b in tier2]
               + [(b, conf) for conf, b in rest])
    expand = ordered[:expand_n]

    lines.append("")
    lines.append(f"展开(最近活跃 top {len(expand)}):")
    for b, conf in expand:
        pin_mark = "📌" if b.get("pinned") else ""
        kind = b.get("kind", "")[:4]
        sup = len(b.get("support_cases", []))
        con = len(b.get("contradiction_cases", []))
        lines.append(f"{pin_mark}[{b['id']}|{kind}|conf {conf:.2f}|+{sup}/-{con}] {b['statement']}")
        cases = b.get("support_cases", [])
        if cases:
            last = cases[-1]
            lines.append(f"    最近case: {last.get('weight_note','')} ({last.get('added','')})")
    return "\n".join(lines)
