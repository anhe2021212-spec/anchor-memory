#!/usr/bin/env python3
"""Discover overlapping relationship candidates among raw Anchor memories.

v2 is deliberately read-only with respect to SQLite and Chroma.  Its only
write mode is ``--shadow-write``, which writes under
``drafts/pattern_candidates``.  ``--dry-run`` never writes anything.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sqlite3
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

import httpx
import numpy as np

from release_config import AnchorConfig

try:
    import chromadb
except ImportError:  # pragma: no cover - exercised on the live host
    chromadb = None


_CONFIG = AnchorConfig.load()
ROOT = _CONFIG.data_dir
DB_PATH = _CONFIG.db_path
CHROMA_PATH = _CONFIG.chroma_dir or (ROOT / "chroma")
ROUTES_PATH = Path(os.environ.get("ANCHOR_MODEL_ROUTES_FILE", ROOT / "model_routes.json"))
OUTPUT_ROOT = ROOT / "drafts/pattern_candidates"
PENDING_DIR = OUTPUT_ROOT / "pending"
DONE_DIR = OUTPUT_ROOT / "done"
SKIPPED_DIR = OUTPUT_ROOT / "skipped"
ARCHIVED_DIR = OUTPUT_ROOT / "archived"
RUNS_DIR = OUTPUT_ROOT / "runs"
STATE_PATH = OUTPUT_ROOT / "state.json"

SCHEMA_VERSION = "cluster_raw_v2.0"
PROMPT_VERSION = "pattern-candidate-v1"
DEFAULT_COLLECTION = "memories_voyage4_1024"
SHADOW_COLLECTION = "memory_shadows_voyage4_1024"
ALLOWED_TYPES = {
    "same_event", "temporal_change", "cause_effect", "continuation",
    "recurring_pattern", "cross_topic_parallel", "contrast", "echo", "other",
}

SYSTEM_PROMPT = """你是 raw 记忆之间的“可能关联发现器”，不是记忆整理者，也不是理解的作者。

给你一批带编号的原始记忆。请找出其中值得放在一起看的候选组。

可能的联系包括但不限于：
- 同一件事随时间发生的变化；
- 前因、过程、结果或后续进展；
- 同一个具体问题反复出现；
- 不同主题或不同事件之间相似的行为、感受、结构或处境；
- 呼应、对照或反差。

边界：
1. 只指出可能的联系，不合并、不删除、不改写任何记忆。
2. 不生成 understanding，不替AI agent下结论，不抽象人格或概念。
3. 每个候选组 2 到 6 条。没有可信联系就不要硬分。
4. 同一条多主题记忆可以出现在多个候选组。
5. 时间接近、标签相同、主题宽泛相似都只是线索，不能单独成为理由。
6. connection_hint 只用一句话说明原文中可复述的连接，不写新记忆。
7. 只使用方括号编号，绝不输出或猜测 memory_id。

严格返回 JSON 对象，不要输出 JSON 之外的文字。"""

OUTPUT_INSTRUCTIONS = """输出字段必须严格使用下面的结构（字段名不可替换）：
{"candidate_groups":[{"members":[1,2],"connection_type":"continuation","connection_hint":"第1条之后，第2条记录了后续。"}]}
members 只能是本批方括号中的整数编号。connection_type 只能取：same_event、temporal_change、cause_effect、continuation、recurring_pattern、cross_topic_parallel、contrast、echo、other。
不要输出 memories、title、summary、delete、skipped 或任何其他字段。"""


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def parse_time(value: str) -> Optional[datetime]:
    try:
        dt = datetime.fromisoformat(value)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except (TypeError, ValueError):
        return None


def readonly_db(path: Path = DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _normalize_matrix(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def group_fingerprint(memory_ids: Iterable[str]) -> str:
    payload = "\n".join(sorted(set(memory_ids))).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def candidate_fingerprint(memory_ids: Iterable[str], connection_type: str) -> str:
    return hashlib.sha256(
        (group_fingerprint(memory_ids) + "\n" + connection_type).encode("utf-8")
    ).hexdigest()


@dataclass
class Pool:
    seed_ids: set[str]
    memory_ids: list[str]
    channels: dict[str, set[str]]

    @property
    def id_set(self) -> set[str]:
        return set(self.memory_ids)

    @property
    def lane_count(self) -> int:
        return len({lane for lanes in self.channels.values() for lane in lanes})


@dataclass
class ModelStats:
    calls: int = 0
    successes: Counter = field(default_factory=Counter)
    failures: Counter = field(default_factory=Counter)
    elapsed_seconds: float = 0.0
    usage: Counter = field(default_factory=Counter)
    json_repairs: int = 0
    errors: list[dict[str, str]] = field(default_factory=list)


def load_raw_and_embeddings() -> tuple[list[dict[str, Any]], int, int, Any]:
    with readonly_db() as conn:
        rows = conn.execute(
            "SELECT memory_id, text, tag, tier, timestamp, context, "
            "emotion_score, collection, level FROM memories "
            "WHERE COALESCE(level, 'raw') = 'raw' "
            "AND COALESCE(collection, '') != 'wenku' ORDER BY timestamp"
        ).fetchall()
    records = {str(r["memory_id"]): dict(r) for r in rows}
    if chromadb is None:
        raise RuntimeError("chromadb is not installed")
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    collection_name = os.environ.get("ANCHOR_CHROMA_COLLECTION", DEFAULT_COLLECTION)
    collection = client.get_collection(collection_name)
    embeddings: dict[str, np.ndarray] = {}
    ids = list(records)
    for start in range(0, len(ids), 500):
        result = collection.get(ids=ids[start:start + 500], include=["embeddings"])
        for mid, embedding in zip(result.get("ids", []), result.get("embeddings", [])):
            if embedding is not None:
                embeddings[str(mid)] = np.asarray(embedding, dtype=np.float32)
    items: list[dict[str, Any]] = []
    for mid in ids:
        if mid not in embeddings:
            continue
        record = records[mid]
        record.update(id=mid, embedding=embeddings[mid])
        items.append(record)
    return items, len(rows), len(rows) - len(items), client


def select_seed_ids(items: list[dict[str, Any]], args: argparse.Namespace) -> list[str]:
    by_id = {item["id"]: item for item in items}
    if args.seed_id:
        missing = [mid for mid in args.seed_id if mid not in by_id]
        if missing:
            print(f"[seed] unknown or missing-embedding IDs: {', '.join(missing)}", file=sys.stderr)
        return [mid for mid in dict.fromkeys(args.seed_id) if mid in by_id]
    if args.backfill:
        return [item["id"] for item in items]
    cutoff = datetime.now(timezone.utc) - timedelta(days=args.since_days)
    if not args.dry_run and STATE_PATH.exists():
        try:
            state_time = parse_time(json.loads(STATE_PATH.read_text(encoding="utf-8"))["last_success_at"])
            if state_time and state_time > cutoff:
                cutoff = state_time
        except (OSError, KeyError, json.JSONDecodeError):
            pass
    return [item["id"] for item in items if (parse_time(item.get("timestamp", "")) or datetime.min.replace(tzinfo=timezone.utc)) >= cutoff]


def _semantic_neighbors(seed_idx: int, sim: np.ndarray, items: list[dict[str, Any]], k: int, floor: float) -> list[str]:
    ranked = sorted(
        ((float(sim[seed_idx, idx]), idx) for idx in range(len(items)) if idx != seed_idx),
        key=lambda pair: (-pair[0], items[pair[1]]["id"]),
    )
    return [items[idx]["id"] for score, idx in ranked if score >= floor][:k]


def _temporal_neighbors(seed: dict[str, Any], items: list[dict[str, Any]], hours: int, k: int) -> list[str]:
    seed_time = parse_time(seed.get("timestamp", ""))
    if not seed_time:
        return []
    limit = hours * 3600
    candidates = []
    for item in items:
        if item["id"] == seed["id"]:
            continue
        item_time = parse_time(item.get("timestamp", ""))
        if item_time:
            delta = abs((item_time - seed_time).total_seconds())
            if delta <= limit:
                candidates.append((delta, item["id"]))
    candidates.sort(key=lambda pair: (pair[0], pair[1]))
    return [mid for _, mid in candidates[:k]]


def _shadow_neighbors(client: Any, seed: dict[str, Any], raw_ids: set[str], k: int) -> list[str]:
    try:
        collection = client.get_collection(SHADOW_COLLECTION)
        result = collection.query(
            query_embeddings=[seed["embedding"].tolist()],
            n_results=max(k * 4, 12), include=["metadatas", "distances"],
        )
        found: list[str] = []
        for meta in (result.get("metadatas") or [[]])[0]:
            parent = str((meta or {}).get("parent_id", ""))
            if parent and parent != seed["id"] and parent in raw_ids and parent not in found:
                found.append(parent)
                if len(found) >= k:
                    break
        return found
    except Exception as exc:
        print(f"[shadow] unavailable, continuing without it: {type(exc).__name__}: {exc}", file=sys.stderr)
        return []


def build_pools(items: list[dict[str, Any]], seed_ids: list[str], args: argparse.Namespace, client: Any) -> tuple[list[Pool], np.ndarray, Counter]:
    by_id = {item["id"]: item for item in items}
    index = {item["id"]: idx for idx, item in enumerate(items)}
    matrix = _normalize_matrix(np.stack([item["embedding"] for item in items]))
    sim = matrix @ matrix.T
    raw_ids = set(by_id)
    pools: list[Pool] = []
    lane_stats: Counter = Counter()
    shadow_ok = not args.no_shadow_lane
    for seed_id in seed_ids:
        seed = by_id[seed_id]
        channels: dict[str, set[str]] = defaultdict(set)
        semantic = _semantic_neighbors(index[seed_id], sim, items, args.semantic_k, args.semantic_floor)
        temporal = _temporal_neighbors(seed, items, args.temporal_hours, args.temporal_k)
        shadow = _shadow_neighbors(client, seed, raw_ids, args.shadow_k) if shadow_ok else []
        for mid in semantic:
            channels[mid].add("semantic")
        for mid in temporal:
            channels[mid].add("temporal")
        for mid in shadow:
            channels[mid].add("shadow")
        lane_stats.update(semantic=len(semantic), temporal=len(temporal), shadow=len(shadow))
        ordered = [seed_id]
        for lane_members in (semantic, temporal, shadow):
            for mid in lane_members:
                if mid not in ordered and len(ordered) < args.pool_size:
                    ordered.append(mid)
        if len(ordered) >= 2:
            pools.append(Pool({seed_id}, ordered, dict(channels)))
    return dedupe_pools(pools), sim, lane_stats


def jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def dedupe_pools(pools: list[Pool]) -> list[Pool]:
    kept: list[Pool] = []
    for pool in pools:
        exact = next((p for p in kept if p.id_set == pool.id_set), None)
        if exact:
            exact.seed_ids |= pool.seed_ids
            for mid, lanes in pool.channels.items():
                exact.channels.setdefault(mid, set()).update(lanes)
            continue
        similar_idx = next((idx for idx, p in enumerate(kept) if jaccard(p.id_set, pool.id_set) >= 0.85), None)
        if similar_idx is None:
            kept.append(pool)
            continue
        current = kept[similar_idx]
        current_key = (current.lane_count, len(current.memory_ids), tuple(current.memory_ids))
        new_key = (pool.lane_count, len(pool.memory_ids), tuple(pool.memory_ids))
        if new_key > current_key:
            pool.seed_ids |= current.seed_ids
            kept[similar_idx] = pool
        else:
            current.seed_ids |= pool.seed_ids
    return kept


def build_numbered_prompt(pool: Pool, by_id: dict[str, dict[str, Any]]) -> tuple[str, dict[int, str], int]:
    parts: list[str] = []
    id_map: dict[int, str] = {}
    truncated = 0
    for number, mid in enumerate(pool.memory_ids, 1):
        item = by_id[mid]
        text = item.get("text") or ""
        context = item.get("context") or ""
        if len(text) > 1200 or len(context) > 300:
            truncated += 1
        text = text[:1200]
        context = context[:300]
        id_map[number] = mid
        parts.append(
            f"[{number}]\n时间：{item.get('timestamp') or ''}\n标签：{item.get('tag') or ''}"
            f"\n原文：{text}\n上下文：{context}"
        )
    return OUTPUT_INSTRUCTIONS + "\n\n待发现可能关联的 raw：\n\n" + "\n\n".join(parts), id_map, truncated


def load_model_routes(route_name: str = "cluster_refine", path: Path = ROUTES_PATH) -> tuple[list[dict[str, Any]], str]:
    routes = json.loads(path.read_text(encoding="utf-8"))
    selected = route_name if routes.get(route_name) else "consolidate"
    raw = routes.get(selected)
    if isinstance(raw, dict):
        raw = [raw]
    if not isinstance(raw, list):
        return [], selected
    expanded: list[dict[str, Any]] = []
    for config in raw:
        if not isinstance(config, dict):
            continue
        models = config.get("models") or [config.get("model")]
        for model in models:
            if model:
                attempt = dict(config)
                attempt["model"] = model
                expanded.append(attempt)
    return expanded, selected


def parse_json_object(content: str) -> tuple[dict[str, Any], bool]:
    body = (content or "").strip()
    body = re.sub(r"^```(?:json)?\s*", "", body, flags=re.I)
    body = re.sub(r"\s*```$", "", body)
    try:
        obj = json.loads(body)
        if not isinstance(obj, dict):
            raise ValueError("top-level JSON is not an object")
        return obj, False
    except (json.JSONDecodeError, ValueError):
        fixed = re.sub(r",\s*([}\]])", r"\1", body)
        start, end = fixed.find("{"), fixed.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("no complete JSON object")
        obj = json.loads(fixed[start:end + 1])
        if not isinstance(obj, dict):
            raise ValueError("top-level JSON is not an object")
        return obj, True


def call_model(system: str, user: str, routes: list[dict[str, Any]], stats: ModelStats) -> tuple[Optional[dict[str, Any]], Optional[str]]:
    for config in routes:
        model = str(config.get("model") or "")
        url = str(config.get("url") or "").rstrip("/")
        key = str(config.get("key") or "")
        if not model or not url or not key:
            stats.failures[model or "invalid-route"] += 1
            continue
        budget = 8192
        for retry in range(2):
            stats.calls += 1
            started = time.monotonic()
            payload = {
                "model": model,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                "temperature": 0.0, "max_tokens": budget,
                "response_format": {"type": "json_object"},
            }
            try:
                response = httpx.post(
                    f"{url}/chat/completions",
                    headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                    json=payload, timeout=180,
                )
                if response.status_code == 400 and "response_format" in response.text.lower():
                    payload.pop("response_format", None)
                    response = httpx.post(
                        f"{url}/chat/completions",
                        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                        json=payload, timeout=180,
                    )
                response.raise_for_status()
                data = response.json()
                choice = data["choices"][0]
                content = choice.get("message", {}).get("content") or ""
                finish = choice.get("finish_reason")
                for name, value in (data.get("usage") or {}).items():
                    if isinstance(value, (int, float)):
                        stats.usage[name] += value
                if finish == "length" or not content.strip():
                    raise ValueError(f"incomplete response: finish_reason={finish!r}, empty={not bool(content.strip())}")
                parsed, repaired = parse_json_object(content)
                stats.json_repairs += int(repaired)
                stats.successes[model] += 1
                stats.elapsed_seconds += time.monotonic() - started
                return parsed, model
            except Exception as exc:
                stats.elapsed_seconds += time.monotonic() - started
                if retry == 0 and budget < 16384:
                    budget = min(16384, budget * 2)
                    continue
                stats.failures[model] += 1
                if isinstance(exc, httpx.HTTPStatusError):
                    detail = f"HTTP {exc.response.status_code}"
                elif isinstance(exc, (ValueError, KeyError, json.JSONDecodeError)):
                    detail = str(exc)[:160]
                else:
                    detail = type(exc).__name__
                stats.errors.append({"model": model, "error": f"{type(exc).__name__}: {detail}"})
    return None, None


def validate_groups(response: dict[str, Any], id_map: dict[int, str], pool: Pool, sim: np.ndarray, index: dict[str, int], model: str, order_start: int, stats: Counter) -> list[dict[str, Any]]:
    raw_groups = response.get("candidate_groups")
    if not isinstance(raw_groups, list):
        stats["invalid_response"] += 1
        return []
    valid: list[dict[str, Any]] = []
    seen: set[tuple[tuple[str, ...], str]] = set()
    for offset, group in enumerate(raw_groups):
        if not isinstance(group, dict):
            stats["invalid_group"] += 1
            continue
        members = group.get("members")
        ctype = group.get("connection_type")
        hint = group.get("connection_hint")
        if not isinstance(members, list) or not all(isinstance(n, int) and not isinstance(n, bool) for n in members):
            stats["invalid_id"] += 1
            continue
        if any(n not in id_map for n in members):
            stats["invalid_id"] += 1
            continue
        mids = [id_map[n] for n in members]
        if len(mids) != len(set(mids)) or not 2 <= len(mids) <= 6:
            stats["invalid_size_or_duplicate"] += 1
            continue
        if ctype not in ALLOWED_TYPES or not isinstance(hint, str) or not hint.strip() or len(hint.strip()) > 300:
            stats["invalid_type_or_hint"] += 1
            continue
        if not (set(mids) & pool.seed_ids):
            stats["history_only_group"] += 1
            continue
        key = (tuple(sorted(mids)), ctype)
        if key in seen:
            stats["duplicate_group"] += 1
            continue
        seen.add(key)
        pairs = [float(sim[index[a], index[b]]) for pos, a in enumerate(mids) for b in mids[pos + 1:]]
        channels = sorted({lane for mid in mids for lane in pool.channels.get(mid, set())})
        valid.append({
            "memory_ids": mids, "connection_type": ctype, "connection_hint": hint.strip(),
            "model": model, "source_channels": channels,
            "min_cosine": min(pairs) if pairs else 0.0,
            "mean_cosine": sum(pairs) / len(pairs) if pairs else 0.0,
            "model_order": order_start + offset,
        })
    return valid


def enforce_global_limits(groups: list[dict[str, Any]], maximum: int) -> list[dict[str, Any]]:
    unique: dict[tuple[tuple[str, ...], str], dict[str, Any]] = {}
    for group in groups:
        key = (tuple(sorted(group["memory_ids"])), group["connection_type"])
        unique.setdefault(key, group)
    ranked = sorted(
        unique.values(),
        key=lambda g: (-len(g["source_channels"]), -g["min_cosine"], g["model_order"], tuple(sorted(g["memory_ids"]))),
    )
    counts: Counter = Counter()
    accepted = []
    for group in ranked:
        if any(counts[mid] >= maximum for mid in group["memory_ids"]):
            continue
        accepted.append(group)
        counts.update(group["memory_ids"])
    return sorted(accepted, key=lambda g: g["model_order"])


_SOURCE_RE = re.compile(r'"source_raw_ids"\s*:\s*\[([^\]]*)\]', re.S)


def build_understanding_back_index() -> dict[str, list[str]]:
    with readonly_db() as conn:
        rows = conn.execute(
            "SELECT m.memory_id AS understanding_id, a.text AS annotation "
            "FROM memories m LEFT JOIN annotations a ON a.memory_id=m.memory_id "
            "WHERE m.level='understanding'"
        ).fetchall()
    result: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        text = row["annotation"] or ""
        ids: list[str] = []
        try:
            obj = json.loads(text)
            if isinstance(obj, dict) and isinstance(obj.get("source_raw_ids"), list):
                ids = [str(value) for value in obj["source_raw_ids"]]
        except json.JSONDecodeError:
            match = _SOURCE_RE.search(text)
            if match:
                ids = re.findall(r'"([^"]+)"', match.group(1))
        for mid in ids:
            result[mid].add(str(row["understanding_id"]))
    return {mid: sorted(values) for mid, values in result.items()}


def scan_prior_candidates() -> tuple[dict[str, str], set[str]]:
    statuses: dict[str, str] = {}
    completed: set[str] = set()
    for status, directory in (("done", DONE_DIR), ("skipped", SKIPPED_DIR), ("archived", ARCHIVED_DIR), ("pending", PENDING_DIR)):
        if not directory.exists():
            continue
        for path in directory.glob("*.json"):
            try:
                fp = json.loads(path.read_text(encoding="utf-8")).get("candidate_fingerprint")
            except (OSError, json.JSONDecodeError):
                continue
            if fp:
                statuses.setdefault(fp, status)
                if status in {"done", "skipped"}:
                    completed.add(fp)
    return statuses, completed


def build_payload(group: dict[str, Any], by_id: dict[str, dict[str, Any]], run_id: str, prior: dict[str, str], back_index: dict[str, list[str]]) -> dict[str, Any]:
    mids = group["memory_ids"]
    candidate_fp = candidate_fingerprint(mids, group["connection_type"])
    dates = [dt for dt in (parse_time(by_id[mid].get("timestamp", "")) for mid in mids) if dt]
    span_days = (max(dates) - min(dates)).days if dates else 0
    understandings = sorted({uid for mid in mids for uid in back_index.get(mid, [])})
    return {
        "schema_version": SCHEMA_VERSION,
        "candidate_id": f"pc_{candidate_fp[:16]}",
        "candidate_fingerprint": candidate_fp,
        "group_fingerprint": group_fingerprint(mids),
        "created_at": now_iso(), "run_id": run_id, "prompt_version": PROMPT_VERSION,
        "model": group["model"], "connection_type": group["connection_type"],
        "connection_hint": group["connection_hint"], "source_channels": group["source_channels"],
        "signals": {"min_cosine": round(group["min_cosine"], 6), "mean_cosine": round(group["mean_cosine"], 6), "span_days": span_days},
        "memory_ids": mids,
        "memories": [
            {"id": mid, "text": by_id[mid].get("text") or "", "tag": by_id[mid].get("tag") or "",
             "tier": by_id[mid].get("tier") or "", "date": by_id[mid].get("timestamp") or "",
             "context": by_id[mid].get("context") or ""}
            for mid in mids
        ],
        "prior_status": prior.get(candidate_fp), "prior_understanding_ids": understandings,
    }


def write_shadow_batch(payloads: list[dict[str, Any]], audit: dict[str, Any], run_id: str) -> list[str]:
    for directory in (PENDING_DIR, DONE_DIR, SKIPPED_DIR, ARCHIVED_DIR, RUNS_DIR):
        directory.mkdir(parents=True, exist_ok=True)
    tmp = OUTPUT_ROOT / f".tmp_{run_id}"
    tmp.mkdir(parents=False, exist_ok=False)
    outputs: list[tuple[Path, Path]] = []
    written_names: list[str] = []
    try:
        for payload in payloads:
            base = f"{payload['candidate_id']}.json"
            dest = PENDING_DIR / base
            if dest.exists():
                try:
                    existing = json.loads(dest.read_text(encoding="utf-8"))
                    if existing.get("candidate_fingerprint") == payload["candidate_fingerprint"]:
                        continue
                except (OSError, json.JSONDecodeError):
                    pass
                dest = PENDING_DIR / f"{payload['candidate_id']}_{int(time.time())}.json"
            src = tmp / dest.name
            src.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            json.loads(src.read_text(encoding="utf-8"))
            outputs.append((src, dest))
            written_names.append(dest.name)
        audit["output_files"] = written_names
        audit_src = tmp / f"{run_id}.json"
        audit_src.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        json.loads(audit_src.read_text(encoding="utf-8"))
        for src, dest in outputs:
            os.replace(src, dest)
        os.replace(audit_src, RUNS_DIR / audit_src.name)
        state_tmp = tmp / "state.json"
        state_tmp.write_text(json.dumps({"last_success_at": now_iso(), "run_id": run_id}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.replace(state_tmp, STATE_PATH)
        return written_names
    finally:
        try:
            tmp.rmdir()
        except OSError:
            pass


def _baseline_enforce_secondary(candidate: set[int], sim: np.ndarray, secondary: float) -> set[int]:
    members = list(candidate)
    while len(members) >= 2:
        violations: Counter = Counter()
        totals: Counter = Counter()
        bad = False
        for pos, left in enumerate(members):
            for right in members[pos + 1:]:
                score = float(sim[left, right])
                totals[left] += score
                totals[right] += score
                if score < secondary:
                    violations[left] += 1
                    violations[right] += 1
                    bad = True
        if not bad:
            return set(members)
        worst = max(members, key=lambda idx: (violations[idx], -totals[idx], idx))
        members.remove(worst)
    return set()


def vector_baseline(items: list[dict[str, Any]], floor: float) -> list[list[str]]:
    """The v1 mutually-exclusive bounded-BFS baseline, retained only for comparison."""
    if len(items) < 2:
        return []
    matrix = _normalize_matrix(np.stack([item["embedding"] for item in items]))
    sim = matrix @ matrix.T
    primary = max(0.75, floor)
    secondary = 0.65
    adjacency = [set() for _ in items]
    for left in range(len(items)):
        for right in range(left + 1, len(items)):
            if float(sim[left, right]) >= primary:
                adjacency[left].add(right)
                adjacency[right].add(left)
    remaining = {idx for idx, neighbors in enumerate(adjacency) if neighbors}
    groups: list[list[str]] = []
    while remaining:
        seed = max(remaining, key=lambda idx: (len(adjacency[idx]), -idx))
        candidate = {seed}
        frontier = {seed}
        for _ in range(2):
            next_frontier = {neighbor for idx in frontier for neighbor in adjacency[idx] if neighbor in remaining and neighbor not in candidate}
            if not next_frontier:
                break
            candidate |= next_frontier
            frontier = next_frontier
        candidate = _baseline_enforce_secondary(candidate, sim, secondary)
        if len(candidate) >= 2:
            groups.append([items[idx]["id"] for idx in sorted(candidate)])
            remaining -= candidate
        else:
            remaining.discard(seed)
    return groups


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--shadow-write", action="store_true")
    parser.add_argument("--since-days", type=float, default=7)
    parser.add_argument("--backfill", action="store_true")
    parser.add_argument("--seed-id", action="append", default=[])
    parser.add_argument("--semantic-k", type=int, default=12)
    parser.add_argument("--semantic-floor", type=float, default=0.48)
    parser.add_argument("--temporal-hours", type=int, default=72)
    parser.add_argument("--temporal-k", type=int, default=8)
    parser.add_argument("--shadow-k", type=int, default=6)
    parser.add_argument("--pool-size", type=int, default=24)
    parser.add_argument("--max-groups-per-memory", type=int, default=3)
    parser.add_argument("--model-route", default="cluster_refine")
    parser.add_argument("--max-pools", type=int, default=20)
    parser.add_argument("--no-shadow-lane", action="store_true")
    parser.add_argument("--json-report", action="store_true")
    parser.add_argument("--vector-baseline", action="store_true")
    return parser


def validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if not 2 <= args.pool_size <= 30:
        parser.error("--pool-size must be between 2 and 30")
    if args.max_pools < 1 or args.semantic_k < 0 or args.temporal_k < 0 or args.shadow_k < 0:
        parser.error("pool and lane limits must be non-negative; --max-pools must be positive")
    if args.backfill and args.seed_id:
        parser.error("--backfill and --seed-id are mutually exclusive")
    if args.backfill and args.shadow_write and "--max-pools" not in sys.argv[1:]:
        parser.error("--shadow-write --backfill requires an explicit --max-pools")


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    validate_args(args, parser)
    started = time.monotonic()
    items, raw_total, missing_embeddings, client = load_raw_and_embeddings()
    seeds = select_seed_ids(items, args)
    report: dict[str, Any] = {"raw_count": raw_total, "seed_count": len(seeds), "missing_embedding_count": missing_embeddings}
    if args.vector_baseline:
        groups = vector_baseline(items, args.semantic_floor)
        report.update(mode="vector_baseline", group_count=len(groups), groups=groups[:20])
        print(json.dumps(report, ensure_ascii=False, indent=2 if not args.json_report else None))
        return 0
    pools, sim, lane_stats = build_pools(items, seeds, args, client)
    pools = pools[:args.max_pools]
    routes, selected_route = load_model_routes(args.model_route)
    if not routes:
        print(f"[model] no usable route for {selected_route}", file=sys.stderr)
        return 2
    by_id = {item["id"]: item for item in items}
    index = {item["id"]: idx for idx, item in enumerate(items)}
    model_stats = ModelStats()
    validation_stats: Counter = Counter()
    all_groups: list[dict[str, Any]] = []
    truncated = 0
    raw_model_groups = 0
    for pool_no, pool in enumerate(pools, 1):
        prompt, id_map, count = build_numbered_prompt(pool, by_id)
        truncated += count
        response, model = call_model(SYSTEM_PROMPT, prompt, routes, model_stats)
        if response is None or model is None:
            continue
        candidate_groups = response.get("candidate_groups")
        raw_model_groups += len(candidate_groups) if isinstance(candidate_groups, list) else 0
        all_groups.extend(validate_groups(response, id_map, pool, sim, index, model, pool_no * 1000, validation_stats))
    accepted = enforce_global_limits(all_groups, args.max_groups_per_memory)
    prior, completed = scan_prior_candidates()
    back_index = build_understanding_back_index()
    run_id = "cr2_" + datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
    payloads = []
    for group in accepted:
        fp = candidate_fingerprint(group["memory_ids"], group["connection_type"])
        if fp not in completed:
            payloads.append(build_payload(group, by_id, run_id, prior, back_index))
    audit = {
        "schema_version": SCHEMA_VERSION, "run_id": run_id, "created_at": now_iso(),
        "mode": "shadow-write" if args.shadow_write else "dry-run", "prompt_version": PROMPT_VERSION,
        "input": {"raw_count": raw_total, "seed_count": len(seeds), "missing_embedding_count": missing_embeddings},
        "pools": {"count": len(pools), "lane_memberships": dict(lane_stats)},
        "model": {"route": selected_route, "calls": model_stats.calls, "successes": dict(model_stats.successes),
                  "failures": dict(model_stats.failures), "elapsed_seconds": round(model_stats.elapsed_seconds, 3),
                  "token_usage": dict(model_stats.usage), "errors": model_stats.errors},
        "groups": {"raw_model_count": raw_model_groups, "valid_count": len(all_groups),
                   "deduped_limited_count": len(accepted), "new_candidate_count": len(payloads),
                   "validation_drops": dict(validation_stats)},
        "truncated_input_count": truncated, "invalid_id_count": validation_stats["invalid_id"],
        "json_repair_count": model_stats.json_repairs,
        "parameters": {key: value for key, value in vars(args).items() if key not in {"seed_id"}},
        "elapsed_seconds": round(time.monotonic() - started, 3), "output_files": [],
    }
    if pools and not model_stats.successes:
        print(json.dumps(audit, ensure_ascii=False, indent=2), file=sys.stderr)
        return 3
    if args.shadow_write:
        write_shadow_batch(payloads, audit, run_id)
    report.update(
        mode=audit["mode"], pool_count=len(pools), model_calls=model_stats.calls,
        raw_model_groups=raw_model_groups, valid_groups=len(all_groups), candidates=len(payloads),
        truncated_input_count=truncated, json_repairs=model_stats.json_repairs,
        validation_drops=dict(validation_stats),
        output_files=audit.get("output_files", []), run_id=run_id,
    )
    print(json.dumps(report, ensure_ascii=False, indent=None if args.json_report else 2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
