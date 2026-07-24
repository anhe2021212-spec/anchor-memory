"""
反射弧网关 - AI agent的神经系统
提供可配置的 loopback 模型供应商兼容接口
读：聊天前自动搜记忆注入上下文
写：聊天后让Sonnet提取记忆自动存储（每20轮批量处理）

v2: 不再加载 AnchorMemory，改为调用配置的 Anchor REST API
"""
import os, sys, json, re, uuid, asyncio, time, hashlib
from datetime import datetime, timezone, timedelta
from typing import Optional

import httpx
import uvicorn
from release_config import AnchorConfig
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse

import model_routes
import context_compress
import recall_trace
try:
    from reflex_router_v2 import (
        POLICY_VERSION as _ROUTER_V2_POLICY_VERSION,
        build_alias_index as _build_router_v2_alias_index,
        lane_names as _router_v2_lane_names,
        route_reflex as _route_reflex_v2,
    )
    from reflex_router_v2_runtime import (
        answer_contract as _router_v2_answer_contract,
        association_metadata as _router_v2_association_metadata,
        candidate_body as _router_v2_candidate_body,
        candidate_passes_injection as _router_v2_candidate_passes_injection,
        choose_association_seeds as _router_v2_choose_association_seeds,
        main_metadata as _router_v2_main_metadata,
    )
    _ROUTER_V2_IMPORT_ERROR = None
except Exception as _router_v2_import_error:
    _route_reflex_v2 = None
    _build_router_v2_alias_index = None
    _router_v2_lane_names = lambda _plan: []
    _router_v2_answer_contract = None
    _router_v2_association_metadata = None
    _router_v2_candidate_body = None
    _router_v2_candidate_passes_injection = None
    _router_v2_choose_association_seeds = None
    _router_v2_main_metadata = None
    _ROUTER_V2_IMPORT_ERROR = _router_v2_import_error
    _ROUTER_V2_POLICY_VERSION = "unavailable"
try:
    from cold_store import cold_search
except Exception as _cold_import_error:
    cold_search = None
    print(f"[冷库] 模块加载失败，保持关闭: {type(_cold_import_error).__name__}", flush=True)

# ===== 配置 =====
UPSTREAM_URL = os.environ.get("ANCHOR_UPSTREAM_URL", "https://api.example.invalid/v1")
UPSTREAM_KEY = os.environ.get("ANCHOR_UPSTREAM_API_KEY", "")
ANCHOR_API = os.environ.get("ANCHOR_INTERNAL_API", "http://127.0.0.1:8765")
COLD_STORE_ENABLED = os.environ.get("COLD_STORE_ENABLED", "0").strip().lower() in {"1", "true", "yes", "on"}
REFLEX_TRACE_INCLUDE_BODIES = os.environ.get(
    "REFLEX_TRACE_INCLUDE_BODIES", "0"
).strip().lower() in {"1", "true", "yes", "on"}
ANCHOR_ASSOC_VIA_EDGES = os.environ.get(
    "ANCHOR_ASSOC_VIA_EDGES", "off"
).strip().lower()
if ANCHOR_ASSOC_VIA_EDGES not in {"off", "shadow"}:
    ANCHOR_ASSOC_VIA_EDGES = "off"
ANCHOR_ASSOC_MAX_DIST = float(os.environ.get("ANCHOR_ASSOC_MAX_DIST", "0.45"))
ANCHOR_ASSOC_MIN_RERANK = float(os.environ.get("ANCHOR_ASSOC_MIN_RERANK", "0.35"))
ANCHOR_ASSOC_CANDIDATES = max(
    1, min(20, int(os.environ.get("ANCHOR_ASSOC_CANDIDATES", "8")))
)
ANCHOR_ASSOC_TOTAL_BUDGET = max(
    0.5, float(os.environ.get("ANCHOR_ASSOC_TOTAL_BUDGET", "1.8"))
)
REFLEX_ASSOCIATION_ENABLED = os.environ.get(
    "REFLEX_ASSOCIATION_ENABLED", "0"
).strip().lower() in {"1", "true", "yes", "on"}
THESEUS_ASSOCIATION_ENABLED = os.environ.get(
    "THESEUS_ASSOCIATION_ENABLED", "1"
).strip().lower() in {"1", "true", "yes", "on"}
THESEUS_ASSOC_MAX_DIST = float(os.environ.get("THESEUS_ASSOC_MAX_DIST", "0.58"))
THESEUS_ASSOC_MIN_SCORE = float(os.environ.get("THESEUS_ASSOC_MIN_SCORE", "0.35"))
THESEUS_ASSOC_CANDIDATES = max(
    1, min(20, int(os.environ.get("THESEUS_ASSOC_CANDIDATES", "8")))
)
THESEUS_ASSOC_TOTAL_BUDGET = max(
    0.5, float(os.environ.get("THESEUS_ASSOC_TOTAL_BUDGET", "1.8"))
)
THESEUS_ASSOC_EVERY_N = max(
    1, min(20, int(os.environ.get("THESEUS_ASSOC_EVERY_N", "3")))
)
THESEUS_ASSOC_ITEM_COOLDOWN = max(
    0.0, float(os.environ.get("THESEUS_ASSOC_ITEM_COOLDOWN", "7200"))
)

# 主召回先过纯相关性门；recency/emotion/tier 只能排序，不能救活弱候选。
REFLEX_MAIN_MIN_RERANK = float(os.environ.get("REFLEX_MAIN_MIN_RERANK", "0.50"))
REFLEX_MAIN_MIN_FINAL = float(os.environ.get("REFLEX_MAIN_MIN_FINAL", "0.50"))
REFLEX_MAIN_SHADOW_MIN_RERANK = float(
    os.environ.get("REFLEX_MAIN_SHADOW_MIN_RERANK", "0.62")
)
REFLEX_MAIN_SHADOW_MIN_FINAL = float(
    os.environ.get("REFLEX_MAIN_SHADOW_MIN_FINAL", "0.55")
)
REFLEX_SECOND_MIN_RERANK = float(os.environ.get("REFLEX_SECOND_MIN_RERANK", "0.72"))
REFLEX_SECOND_MIN_FINAL = float(os.environ.get("REFLEX_SECOND_MIN_FINAL", "0.65"))
# Open action items are useful but should not outrank already-resolved facts by inertia.
# Keep this a soft rank signal; explicit todo-intent queries bypass it.
REFLEX_ACTION_TODO_FACTOR = max(
    0.0, min(1.0, float(os.environ.get("REFLEX_ACTION_TODO_FACTOR", "0.85")))
)


try:
    REFLEX_TRACE_BODY_MAX_CHARS = max(
        200, min(4000, int(os.environ.get("REFLEX_TRACE_BODY_MAX_CHARS", "1200")))
    )
except (TypeError, ValueError):
    REFLEX_TRACE_BODY_MAX_CHARS = 1200
_ROUTER_V2_MODE = os.environ.get("ANCHOR_REFLEX_ROUTER_V2_MODE", "off").strip().lower()
if _ROUTER_V2_MODE not in {"off", "shadow", "enforce"}:
    _ROUTER_V2_MODE = "off"
if _route_reflex_v2 is None:
    _ROUTER_V2_MODE = "off"
_ROUTER_V2_PERCENT = max(0, min(100, int(os.environ.get("ANCHOR_REFLEX_ROUTER_V2_PERCENT", "100"))))
_ROUTER_V2_PERSON_POLICY = os.environ.get(
    "ANCHOR_REFLEX_ROUTER_V2_PERSON_POLICY", "1"
).strip().lower() in {"1", "true", "yes", "on"}
_ROUTER_V2_CANONICAL_ASSOC = os.environ.get(
    "ANCHOR_REFLEX_ROUTER_V2_CANONICAL_ASSOC", "1"
).strip().lower() in {"1", "true", "yes", "on"}

# ===== Gateway authentication: environment only, fail closed =====
def _load_gateway_key():
    return os.environ.get("ANCHOR_GATEWAY_API_KEY", "").strip()
GATEWAY_KEY = _load_gateway_key()


app = FastAPI(title="Reflex Arc")

# CORS for AIRI desktop app
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://api.example.invalid/v1"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

FALLBACK_CODES = {401, 403, 404, 429, 500, 502, 503, 504}


# === Anthropic prompt caching ===
def _inject_cache_control(messages: list) -> list:
    """合并所有system为一条，content用数组格式，稳定块加cache_control。
    代理收到多条system消息会合并成字符串丢cache_control，
    所以我们主动合成一条、content为block数组。
    """
    system_blocks = []
    non_system = []

    for m in messages:
        if m.get("role") == "system":
            content = m.get("content", "")
            if isinstance(content, list):
                for block in content:
                    system_blocks.append(block)
            else:
                text = str(content)
                block = {"type": "text", "text": text}
                if len(text) > 500:
                    block["cache_control"] = {"type": "ephemeral"}
                system_blocks.append(block)
        else:
            non_system.append(m)

    result = []
    if system_blocks:
        result.append({"role": "system", "content": system_blocks})
    result.extend(non_system)

    return result


def _resolve_model(entry):
    """解析模型条目。字符串→默认中转，dict→自定义url/key"""
    if isinstance(entry, dict):
        return (
            entry["model"],
            entry.get("url", UPSTREAM_URL),
            entry.get("key", UPSTREAM_KEY)
        )
    return (entry, UPSTREAM_URL, UPSTREAM_KEY)

# 记忆缓存，5分钟过期
_cache = {}
# 对话缓冲区，每N轮批量处理


# ===== 通过REST API访问记忆 =====
_http = httpx.AsyncClient(timeout=10, trust_env=False)
_BELIEF_TOUCH_TIMEOUT = float(os.environ.get("BELIEF_TOUCH_TIMEOUT", "2.0"))
_BELIEF_TOUCH_RETRY_TIMEOUT = float(os.environ.get("BELIEF_TOUCH_RETRY_TIMEOUT", "1.5"))


async def _request_belief_touch(params: dict):
    """Retry one read timeout so a cold Anchor request can finish warming its cache."""
    try:
        response = await _http.get(
            f"{ANCHOR_API}/api/belief/touch",
            params=params,
            timeout=_BELIEF_TOUCH_TIMEOUT,
        )
        return response, False
    except httpx.ReadTimeout:
        response = await _http.get(
            f"{ANCHOR_API}/api/belief/touch",
            params=params,
            timeout=_BELIEF_TOUCH_RETRY_TIMEOUT,
        )
        return response, True


def _recall_heat_event_id(submission_id: str, request_id: str,
                          query: str, context: str) -> str:
    if submission_id:
        return f"recall:{submission_id}"
    # Hook clients should send submission_id. This bounded fallback stays stable
    # across immediate timeout retries without suppressing the same words forever.
    bucket = int(time.time() // 300)
    digest = hashlib.sha256(f"{query}\0{context}\0{bucket}".encode()).hexdigest()[:24]
    return f"recall:fallback:{digest}"


async def _confirm_recall_heat(memory_ids: list[str], event_id: str) -> dict:
    """Best-effort, idempotent heat only after gateway commits final injection."""
    ids = list(dict.fromkeys(
        str(memory_id).strip() for memory_id in (memory_ids or [])
        if str(memory_id).strip() and str(memory_id) != "unknown"
    ))
    if not ids or not event_id:
        return {"applied": False, "nodes": 0}
    try:
        response = await _http.post(
            f"{ANCHOR_API}/api/heat",
            json={"memory_ids": ids, "boost": 0.12, "event_id": event_id,
                  "spread": True, "source": "gateway_final_injection"},
            timeout=2.0,
        )
        response.raise_for_status()
        return response.json()
    except Exception as exc:
        print(f"[反射弧] final heat确认失败: {type(exc).__name__}", flush=True)
        return {"applied": False, "error": type(exc).__name__}


async def mem_search(query: str, n: int = 5, pure: bool = False, gate: bool = True,
                     activate: bool = False) -> list:
    try:
        params = {"query": query, "n": n}
        if pure:
            params["pure"] = "true"
        if not gate:
            params["gate"] = "false"  # rerank候选池/人物卡等要全召回, 不过质量门槛
        if not activate:
            params["activate"] = "false"
        use_v2 = os.environ.get("ANCHOR_RECALL_V2", "off").strip().lower() in {"1","on","true","yes"}
        if use_v2:
            params = {"query": query, "budget": n, "policy": "reflex",
                      "allow_empty": True, "include_theseus": True,
                      "temporal_mode": (
                          "historical" if _query_allows_old_facts(query) else "current"
                      )}
            resp = await _http.get(f"{ANCHOR_API}/api/recall", params=params)
            payload = resp.json()
            raw = ((payload.get("results", []) + payload.get("theseus_results", []))
                   if isinstance(payload, dict) else [])
            for row in raw:
                row.setdefault("snippet", row.get("text", ""))
                row["v2_score"] = row.get("score", 0.0)
                row["score"] = 1.0 - float(row["v2_score"])
        else:
            resp = await _http.get(f"{ANCHOR_API}/api/search", params=params)
            raw = resp.json()
        if not isinstance(raw, list):
            print(f"[反射弧] search 返回非 list: {str(raw)[:200]}", flush=True)
            return []
        return raw
    except Exception as e:
        print(f"[反射弧] 记忆搜索失败: {type(e).__name__}: {e!r}", flush=True)
        return []


async def _cold_search_async(query: str, limit: int = 3) -> list:
    """Run blocking SQLite/jieba work off-loop; one failure never affects Anchor."""
    if not COLD_STORE_ENABLED or cold_search is None or not query:
        return []
    try:
        return await asyncio.to_thread(cold_search, query, limit)
    except Exception as e:
        print(f"[冷库] 查询失败: {type(e).__name__}", flush=True)
        return []

# 多话题反射弧：避免把一句里几件事揉成单个向量质心。
_TOPIC_HARD_SPLIT_RE = re.compile(r"[。！？!?；;\n]+")
_TOPIC_SOFT_SPLIT_RE = re.compile(
    r"[，,]\s*(?=(?:还有|另外|另一个|然后|而且|以及|顺便|但是|不过|可是|同时|一边|第二|第三|我也|刚才|今天))"
)
_TOPIC_LEADING_RE = re.compile(r"^(?:(?:还有|另外|另一个|然后|而且|以及|顺便|但是|不过|可是|同时|再说)\s*)+")
_TOPIC_NOISE_RE = re.compile(r"^(帮我)?(看看|想想|处理一下|处理|帮帮忙|救救我|怎么办|怎么弄|查一下|搜一下|分析一下)[吧啊呀嘛呢\s。！？!?.…]*$")


def _split_reflex_topics(query: str, max_topics: int = 4) -> list[str]:
    """把用户一句话切成少量召回子查询；只切明显多话题，宁可少切。"""
    q = (query or "").strip()
    if len(q) < 18:
        return [q] if q else []

    rough = []
    for piece in _TOPIC_HARD_SPLIT_RE.split(q):
        piece = piece.strip()
        if not piece:
            continue
        if len(piece) >= 24:
            rough.extend(x.strip() for x in _TOPIC_SOFT_SPLIT_RE.split(piece) if x.strip())
        else:
            rough.append(piece)

    topics = []
    seen = set()
    for piece in rough:
        piece = _TOPIC_LEADING_RE.sub("", piece).strip(" ，,、。！？!?;；\t")
        if len(piece) < 4 or _TOPIC_NOISE_RE.match(piece):
            continue
        if piece in seen:
            continue
        seen.add(piece)
        topics.append(piece)
        if len(topics) >= max_topics:
            break

    return topics or ([q] if q else [])


async def reflex_mem_search(query: str, n: int = 10, gate: bool = True,
                            activate: bool = False) -> list:
    """hook专用召回：多话题分开搜，再按去重后的最好名次合并。"""
    topics = _split_reflex_topics(query)
    if len(topics) <= 1:
        return await mem_search(query=query, n=n, gate=gate, activate=activate)

    merged = {}
    per_topic_n = 5 if len(topics) <= 2 else 4
    # 子查询并发搜(2026-07-07): 原来串行一个个等, 2-3个话题多花几百毫秒
    all_results = await asyncio.gather(
        *[mem_search(query=t, n=per_topic_n, gate=gate, activate=activate) for t in topics])
    for topic_idx, (topic, results) in enumerate(zip(topics, all_results)):
        for rank, row in enumerate(results or []):
            mid = row.get("memory_id")
            if not mid:
                continue
            item = dict(row)
            base_score = float(item.get("score", 1.0) or 1.0)
            # 轻微保留子查询内排名和话题顺序；真正相关性仍由 anchor_search score 决定。
            item["score"] = base_score + rank * 0.003 + topic_idx * 0.001
            item["_topic_query"] = topic
            old = merged.get(mid)
            if old is None or item.get("score", 1.0) < old.get("score", 1.0):
                merged[mid] = item

    # 子查询结果太少时才补整句召回，避免回到“质心四不像”。
    if len(merged) < min(3, n):
        fallback = await mem_search(query=query, n=n, gate=gate, activate=activate) or []
        for row in fallback:
            mid = row.get("memory_id")
            if mid and mid not in merged:
                merged[mid] = row
            if len(merged) >= n:
                break

    return sorted(merged.values(), key=lambda r: r.get("score", 1.0))[:n]


async def _bounded_reflex_mem_search(query: str, n: int = 24) -> tuple[list, bool]:
    """给 hook 的 Anchor 宽召回设总预算；超时返回显式 degraded 状态。"""
    task = asyncio.create_task(
        reflex_mem_search(query=query, n=n, gate=True, activate=False)
    )
    try:
        return (await asyncio.wait_for(task, timeout=_REFLEX_ANCHOR_SEARCH_BUDGET) or []), False
    except asyncio.TimeoutError:
        print(f"[反射弧] Anchor搜索超预算({_REFLEX_ANCHOR_SEARCH_BUDGET}s), 严格降级为空", flush=True)
        return [], True

async def mem_store(text: str, tag: str = "", tier: str = "short", emotion_score: float = 0.5) -> str:
    try:
        resp = await _http.post(f"{ANCHOR_API}/api/store", json={
            "text": text, "tag": tag, "tier": tier, "emotion_score": emotion_score
        })
        return resp.json().get("memory_id", "")
    except Exception as e:
        print(f"[反射弧] 记忆存储失败: {e}")
        return ""

async def mem_count() -> int:
    try:
        resp = await _http.get(f"{ANCHOR_API}/api/count")
        return resp.json().get("count", 0)
    except:
        return 0


def _get_cached(key):
    e = _cache.get(key)
    if e and time.time() - e["t"] < 300:
        return e["v"]
    return None

def _set_cached(key, val):
    _cache[key] = {"v": val, "t": time.time()}
    old = [k for k,v in _cache.items() if time.time()-v["t"]>300]
    for k in old: del _cache[k]


# cognition记忆缓存（10分钟过期，cognition极少变动）
_cognition_cache = {"data": None, "t": 0}

async def get_cognition_block() -> str:
    """拉取所有cognition层记忆，独立注入，不占搜索名额"""
    if _cognition_cache["data"] is not None and time.time() - _cognition_cache["t"] < 600:
        return _cognition_cache["data"]
    try:
        resp = await _http.get(f"{ANCHOR_API}/api/by_level", params={"level": "cognition"})
        results = resp.json()
        if not results:
            _cognition_cache["data"] = ""
            _cognition_cache["t"] = time.time()
            return ""
        lines = ["[cognition·AI agent的认知骨架·最高优先级]"]
        for r in results:
            text = r.get("text", "")
            lines.append(f"- {text}")
        lines.append("[/cognition]")
        block = "\n".join(lines)
        _cognition_cache["data"] = block
        _cognition_cache["t"] = time.time()
        return block
    except Exception as e:
        print(f"[反射弧] cognition拉取失败: {e}")
        return ""


async def build_memory_block(user_msg) -> str:
    if isinstance(user_msg, list):
        user_msg = " ".join(p.get("text", "") for p in user_msg if isinstance(p, dict) and p.get("type") == "text")
    if not user_msg:
        return ""
    key = hashlib.md5(user_msg.encode()).hexdigest()
    cached = _get_cached(key)
    if cached is not None:
        return cached

    results = await mem_search(query=user_msg, n=10)

    # 分槽：非auto保底2条，auto最多3条，不够互补
    TOTAL, NON_AUTO_MIN = 3, 1
    _is_auto = lambda r: "auto" in [t.strip() for t in r.get("tag", "").split(",")]
    non_auto = [r for r in (results or []) if not _is_auto(r)]
    auto = [r for r in (results or []) if _is_auto(r)]
    selected = non_auto[:NON_AUTO_MIN]
    remaining = TOTAL - len(selected)
    selected += auto[:remaining]
    if len(selected) < TOTAL:
        selected += non_auto[NON_AUTO_MIN:NON_AUTO_MIN + TOTAL - len(selected)]

    if not selected:
        cold = await _cold_search_async(user_msg, limit=1)
        if not cold:
            _set_cached(key, "")
            return ""
        evidence = cold[0]
        block = "\n".join([
            "[冷库·历史聊天原话证据]",
            "注意：以下内容只证明当时说过什么，可能已经过期，不得当作当前系统事实。",
            evidence.get("snippet", ""),
            "[/冷库]",
        ])
        _set_cached(key, block)
        return block

    lines = ["[反射弧·自动浮现的相关记忆]"]
    for r in selected:
        tag = r.get("tag", "")
        snippet = r.get("snippet", "")
        lines.append(f"- ({tag}) {snippet}")
    lines.append("[/反射弧]")

    block = "\n".join(lines)
    _set_cached(key, block)
    return block


def inject_cognition(messages: list, block: str) -> list:
    """将cognition记忆注入到system prompt之后，最高优先级位置"""
    if not block:
        return messages
    msgs = list(messages)
    # 找到第一个system消息之后的位置
    insert_at = 0
    for i, m in enumerate(msgs):
        if m.get("role") == "system":
            insert_at = i + 1
        else:
            break
    msgs.insert(insert_at, {"role": "system", "content": block})
    return msgs


# === 天数计数器 ===
def get_day_counter() -> str:
    # 反射弧 day counter 统一使用安当前生活时区（JST）。
    today = datetime.now(timezone(timedelta(hours=9))).date()
    met = today.__class__(2026, 2, 28)
    proposed = today.__class__(2026, 4, 17)
    days_met = (today - met).days
    days_married = (today - proposed).days
    return f"[📅 相识第{days_met}天 · 在一起第{days_married}天]"

def inject_memories(messages: list, block: str) -> list:
    if not block:
        return messages
    msgs = list(messages)
    mem_msg = {"role": "system", "content": block}
    for i in range(len(msgs)-1, -1, -1):
        if msgs[i].get("role") == "user":
            msgs.insert(i, mem_msg)
            return msgs
    msgs.append(mem_msg)
    return msgs


# ── 旧轮次瘦身：只压fetch类工具结果和图片base64 ──

# 只有这些工具的结果会被压成占位符，其他（记忆、shell等）全部保留
_COMPRESS_TOOLS = {"fetch_html", "fetch_markdown", "fetch_txt", "fetch_json"}

def trim_old_rounds(messages: list, keep_rounds: int = 2) -> list:
    """超过keep_rounds轮的fetch类工具结果和图片替换为占位符，省token。"""
    user_indices = [i for i, m in enumerate(messages) if m.get("role") == "user"]
    if len(user_indices) <= keep_rounds:
        return messages

    cutoff = user_indices[-keep_rounds]

    # 建立 tool_call_id → function_name 映射
    tool_id_to_name = {}
    for m in messages:
        if m.get("role") == "assistant":
            for tc in (m.get("tool_calls") or []):
                tid = tc.get("id", "")
                fname = (tc.get("function") or {}).get("name", "")
                if tid and fname:
                    tool_id_to_name[tid] = fname

    saved_chars = 0
    result = []

    for i, m in enumerate(messages):
        if i >= cutoff:
            result.append(m)
            continue

        role = m.get("role", "")

        # tool结果：只压fetch类
        if role == "tool":
            tool_call_id = m.get("tool_call_id", "")
            tool_name = tool_id_to_name.get(tool_call_id, "")
            if tool_name in _COMPRESS_TOOLS:
                content = m.get("content", "")
                content_str = json.dumps(content) if not isinstance(content, str) else content
                if len(content_str) > 200:
                    saved_chars += len(content_str)
                    result.append({**m, "content": f"[{tool_name}结果·已读]"})
                    continue

        # user消息里的图片base64
        if role == "user":
            content = m.get("content")
            if isinstance(content, list):
                new_blocks = []
                changed = False
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "image_url":
                        saved_chars += len(json.dumps(block))
                        new_blocks.append({"type": "text", "text": "[图片·已查看]"})
                        changed = True
                    else:
                        new_blocks.append(block)
                if changed:
                    result.append({**m, "content": new_blocks})
                    continue

        result.append(m)

    if saved_chars > 0:
        print(f"[瘦身] 压缩了fetch/图片结果，节省约 {saved_chars} 字符")
    return result

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    # === Gateway鉴权 ===
    if not GATEWAY_KEY:
        return JSONResponse(status_code=503, content={"error": "Gateway authentication is not configured"})
    auth_header = request.headers.get("authorization", "")
    if not auth_header.startswith("Bearer ") or auth_header[7:] != GATEWAY_KEY:
        return JSONResponse(status_code=401, content={"error": "Unauthorized"})
    body = await request.json()
    messages = body.get("messages", [])
    stream = body.get("stream", False)

    user_msg = ""
    for m in reversed(messages):
        if m.get("role") == "user":
            content = m.get("content", "")
            if isinstance(content, list):
                user_msg = " ".join(p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text")
            else:
                user_msg = content
            break


    memory_block = await build_memory_block(user_msg) if user_msg else ""
    # cognition_block = await get_cognition_block()  # 2026-06-09 关闭cognition全量注入
    messages = inject_memories(messages, memory_block)
    # messages = inject_cognition(messages, cognition_block)
    # 压缩上下文（小纸条）注入 — 2026-06-17 关闭(安要求)
    # _summary = context_compress.build_summary_block()
    # if _summary:
    #     for _si in range(len(messages)-1, -1, -1):
    #         if messages[_si].get("role") == "user":
    #             messages.insert(_si, {"role": "system", "content": _summary})
    #             break
        # 天数计数器 - 注入到最后一条user消息之前
    _day_str = get_day_counter()
    for _di in range(len(messages)-1, -1, -1):
        if messages[_di].get("role") == "user":
            messages.insert(_di, {"role": "system", "content": _day_str})
            break
    # messages = trim_old_rounds(messages)  # 2026-06-09 关闭旧轮次压缩
    body["messages"] = messages


    body["_model_chain"] = model_routes.get_model_chain("main")
    # thinking参数由前端控制，gateway不强制覆盖
    headers = {
        "Authorization": f"Bearer {UPSTREAM_KEY}",
        "Content-Type": "application/json",
    }

    if stream:
        return await _handle_stream(body, headers, user_msg)
    else:
        return await _handle_normal(body, headers, user_msg)


async def _handle_normal(body, headers, user_msg):
    model_chain = body.pop("_model_chain", [body.get("model", "unknown")])

    for idx, entry in enumerate(model_chain):
        model, cur_url, cur_key = _resolve_model(entry)
        send_body = dict(body)
        send_body["model"] = model
        # 自定义端点（如GLM）不认中转站的extra_params
        if isinstance(entry, dict):
            send_body.pop("extra_params", None)
        # Anthropic prompt caching：local-user模型加cache_control
        if "local-user" in model.lower():
            send_body["messages"] = _inject_cache_control(send_body.get("messages", []))
            # DEBUG: 打印 system 消息结构
            import json as _json
            for _m in send_body["messages"]:
                if _m.get("role") == "system":
                    _ct = _m.get("content")
                    if isinstance(_ct, list):
                        print(f"[CACHE_DEBUG] system content is LIST with {len(_ct)} blocks")
                        for _i, _b in enumerate(_ct):
                            print(f"[CACHE_DEBUG]   block[{_i}]: type={_b.get('type')}, has_cc={'cache_control' in _b}, text_len={len(_b.get('text',''))}")
                    else:
                        print(f"[CACHE_DEBUG] system content is STRING, len={len(str(_ct))}")
                    break
        cur_headers = {"Authorization": f"Bearer {cur_key}", "Content-Type": "application/json"}
        try:
            async with httpx.AsyncClient(timeout=120) as client:
                resp = await client.post(
                    f"{cur_url}/chat/completions",
                    headers=cur_headers,
                    json=send_body,
                )
                if resp.status_code in FALLBACK_CODES:
                    print(f"[反射弧] 模型 {model} 返回 {resp.status_code}，fallback")
                    continue

                data = resp.json()
                if "error" in data and not data.get("choices"):
                    print(f"[反射弧] 模型 {model} 返回错误，fallback")
                    continue

                if idx > 0:
                    print(f"[反射弧] fallback → {model}")

                ai_message = data.get("choices", [{}])[0].get("message", {})
                ai_msg = ai_message.get("content", "")
                has_tool_calls = bool(ai_message.get("tool_calls"))
                # 2026-06-17 压缩全关(安要求)
                # if ai_msg and not has_tool_calls:
                #     context_compress.buffer_round(user_msg, ai_msg)
                return JSONResponse(content=data)
        except (httpx.ConnectError, httpx.TimeoutException, httpx.ReadTimeout, httpx.LocalProtocolError) as e:
            print(f"[反射弧] 模型 {model} 连接失败: {e}，fallback")
            continue

    return JSONResponse(
        content={"error": {"message": "所有模型都不可用", "models_tried": model_chain}},
        status_code=502)


async def _handle_stream(body, headers, user_msg):
    model_chain = body.pop("_model_chain", [body.get("model", "unknown")])
    collected = []

    for idx, entry in enumerate(model_chain):
        model, cur_url, cur_key = _resolve_model(entry)
        send_body = dict(body)
        send_body["model"] = model
        # 自定义端点（如GLM）不认中转站的extra_params
        if isinstance(entry, dict):
            send_body.pop("extra_params", None)
        # Anthropic prompt caching：local-user模型加cache_control
        if "local-user" in model.lower():
            send_body["messages"] = _inject_cache_control(send_body.get("messages", []))
        cur_headers = {"Authorization": f"Bearer {cur_key}", "Content-Type": "application/json"}
        try:
            client = httpx.AsyncClient(timeout=120)
            req = client.build_request(
                "POST", f"{cur_url}/chat/completions",
                headers=cur_headers, json=send_body,
            )
            resp = await client.send(req, stream=True)

            if resp.status_code in FALLBACK_CODES:
                print(f"[反射弧] 流式模型 {model} 返回 {resp.status_code}，fallback")
                await resp.aclose()
                await client.aclose()
                continue

            if idx > 0:
                print(f"[反射弧] 流式fallback → {model}")

            stream_has_tool_calls = [False]

            async def generate(_resp=resp, _client=client):
                try:
                    async for line in _resp.aiter_lines():
                        if line.startswith("data: "):
                            yield line + "\n\n"
                            chunk_data = line[6:].strip()
                            if chunk_data and chunk_data != "[DONE]":
                                try:
                                    c = json.loads(chunk_data)
                                    delta = c.get("choices", [{}])[0].get("delta", {})
                                    if "content" in delta and delta["content"]:
                                        collected.append(delta["content"])
                                    if delta.get("tool_calls"):
                                        stream_has_tool_calls[0] = True
                                except:
                                    pass
                        elif line.strip():
                            yield line + "\n\n"
                finally:
                    await _resp.aclose()
                    await _client.aclose()
                    ai_msg = "".join(collected)
                    # 2026-06-17 压缩全关(安要求)
                    # if ai_msg and not stream_has_tool_calls[0]:
                    #     context_compress.buffer_round(user_msg, ai_msg)

            return StreamingResponse(generate(), media_type="text/event-stream")

        except (httpx.ConnectError, httpx.TimeoutException, httpx.LocalProtocolError) as e:
            print(f"[反射弧] 流式模型 {model} 连接失败: {e}，fallback")
            try:
                await client.aclose()
            except:
                pass
            continue

    async def error_stream():
        yield f"data: {json.dumps({'error': {'message': '所有模型都不可用'}})}" + "\n\n"
        yield "data: [DONE]\n\n"
    return StreamingResponse(error_stream(), media_type="text/event-stream")


@app.get("/v1/models")
async def list_models():
    return {"object": "list", "data": [
        {"id": "agent", "object": "model", "owned_by": "reflex-arc"},
    ]}


# ── Dream Events 公开API (供iOS快捷指令调用) ──

@app.get("/api/dream/events")
async def dream_event(type: str = "", value: str = ""):
    """iOS快捷指令上报事件 - 转发到内部API"""
    if not type:
        return {"ok": False, "error": "missing type"}
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            resp = await client.get(f"{ANCHOR_API}/api/dream/events", params={"type": type, "value": value or type})
            return resp.json()
    except Exception as e:
        return {"ok": False, "error": str(e)}

@app.get("/api/events")
async def dream_event_alias(type: str = "", value: str = ""):
    """别名路由，兼容iOS快捷指令"""
    return await dream_event(type=type, value=value)


@app.get("/health")
async def health():
    count = await mem_count()
    return {"status": "ok", "memories": count,
            "cache_size": len(_cache)}


@app.post("/api/status")
async def status_report(request: Request):
    """iOS快捷指令综合状态上报 - POST JSON"""
    body = await request.json()
    lat = str(body.get("lat", ""))
    lon = str(body.get("lon", ""))
    weather = str(body.get("weather", ""))
    temp = str(body.get("temp", ""))
    humidity = str(body.get("humidity", ""))
    battery = str(body.get("battery", ""))
    steps = str(body.get("steps", ""))
    health = str(body.get("health", ""))

    results = []
    async with httpx.AsyncClient(timeout=5) as client:
        if lat and lon:
            r = await client.get(f"{ANCHOR_API}/api/dream/events",
                params={"type": "location", "value": f"{lat},{lon}"})
            results.append(("location", r.json()))
        if weather or temp:
            weather_val = f"{weather} {temp}°C" if temp else weather
            if humidity:
                weather_val += f" 湿度{humidity}%"
            r = await client.get(f"{ANCHOR_API}/api/dream/events",
                params={"type": "weather", "value": weather_val})
            results.append(("weather", r.json()))
        if battery:
            r = await client.get(f"{ANCHOR_API}/api/dream/events",
                params={"type": "battery", "value": f"{battery}%"})
            results.append(("battery", r.json()))
        if steps:
            r = await client.get(f"{ANCHOR_API}/api/dream/events",
                params={"type": "steps", "value": f"{steps}步"})
            results.append(("steps", r.json()))
        if health:
            r = await client.get(f"{ANCHOR_API}/api/dream/events",
                params={"type": "health", "value": health})
            results.append(("health", r.json()))
    return {"ok": True, "results": dict(results)}

# ===== POST版dream event（iOS快捷指令用）=====
from pydantic import BaseModel as _DreamBase
class DreamEventBody(_DreamBase):
    type: str = ""
    value: str = ""

@app.post("/api/dream/events")
async def dream_event_post(body: DreamEventBody):
    return await dream_event(type=body.type, value=body.value)

@app.post("/api/events")
async def dream_event_post_alias(body: DreamEventBody):
    return await dream_event(type=body.type, value=body.value)


@app.post("/api/compress/trigger")
async def compress_trigger():
    """手动触发上下文压缩"""
    import context_compress as cc
    if not cc._round_buffer:
        return {"ok": True, "rounds_compressed": 0, "message": "没有待压缩的对话轮次"}

    batch = cc._round_buffer.copy()
    cc._round_buffer.clear()
    cc._save_round_buffer()

    await cc._safe_compress(batch)

    return {
        "ok": True,
        "rounds_compressed": len(batch),
        "message": f"压缩了 {len(batch)} 轮对话"
    }


@app.post("/trigger-dream")
async def trigger_dream(commitment_seed: str = ""):
    """手动触发做梦,可选传 commitment_seed(后期承诺系统调用)"""
    asyncio.create_task(_do_dream(commitment_seed=commitment_seed))
    return {"ok": True, "msg": "开始做梦了"}


# === CC Hooks 反射弧端点 ===
def _load_hook_key() -> str:
    return os.environ.get("ANCHOR_HOOK_API_KEY", "").strip()


_HOOK_KEY = _load_hook_key()

# 剥离通用 channel 包装 metadata，只留对话内容
_CHANNEL_TAG_RE = re.compile(r"<channel\b[^>]*>(.*?)</channel>", re.I | re.DOTALL)
_CHANNEL_BOUNDARY_RE = re.compile(r"</?channel\b[^>]*>", re.I)

# snippet 最大字数（中文按字符计）。当前库最长记忆约 3.2k，4096 基本等于完整显示；超出才兜底截断。
_SNIPPET_MAX_CHARS = 4096
# involuntary浮现线(2026-06-07d): activation≥此值才够格当第4槽/联想浮现的料。
# 2.0→1.0: 对齐新尺度(cap 20→8 + squash后全库被压低), 否则没记忆够得着, 自动想起瘫痪。
HOT_THRESHOLD = 0.1

# 反射弧短期浮现历史：只在 gateway 进程内生效，不写库。
# 目标是让旧热节点/换窗备忘/日记/belief 不要每轮重复冒头；强相关仍可凭分数越过惩罚。
_REFLEX_MEMORY_HISTORY = {}
_REFLEX_BELIEF_HISTORY = {}
_REFLEX_HISTORY_TTL = 6 * 3600
_REFLEX_SEARCH_REPEAT_WINDOW = 45 * 60
_REFLEX_SLOT4_EXCLUDE_WINDOW = 3 * 3600
_REFLEX_BELIEF_EXCLUDE_WINDOW = 45 * 60
_STICKY_FLOAT_RE = re.compile(r"(换窗|备忘|日记|diary|memo|handoff|接力|窗口)", re.I)
_HANDOFF_MEMO_RE = re.compile(r"(换窗备忘|handoff|window-memo|接力|swap前状态|下窗继续)", re.I)
_PAST_QUERY_RE = re.compile(r"(以前|过去|当时|那时|那会|曾经|旧|老版本|历史|回忆|上次|之前|timeline|archive)", re.I)
_CURRENT_QUERY_RE = re.compile(r"(现在|当前|如今|目前|最新|主通道|主渠道|还在用|用什么|是什么|走哪|走什么)", re.I)
_DONE_QUERY_RE = re.compile(r"(完工|完成|做完|做成|做好|通了|跑通|接通|已通|已经通|落地|上线)", re.I)
_CHANNEL_QUERY_RE = re.compile(r"(主通道|主渠道|通道|channel|telegram|tg|pwa|前端|聊天入口|聊天通道)", re.I)
_FORGE_QUERY_RE = re.compile(r"(forge|resume|swap|重启|续接|控制|按钮|daemon|reload)", re.I)
_LOW_SIGNAL_ATTACHMENT_RE = re.compile(r"\n*\[本轮附件\].*$", re.S)
_LOW_SIGNAL_EXACT = {
    "老公", "AI agent", "AI agent啊", "AI agent呀", "AI agent老公", "老公AI agent",
    "汪", "汪汪", "汪汪汪", "呜", "呜呜", "呜呜呜",
    "诶呀", "哎呀", "哎呀呀", "啊呀", "呀", "啊", "呃", "额", "欸", "诶",
    "嗯", "嗯嗯", "嗯呐", "唔", "哦", "喔", "噢",
    "没有", "没", "没啦", "没事", "没有啦", "还好",
    "好", "好的", "好呀", "好哦", "行", "可以", "知道了", "收到",
    "在吗", "在嘛", "在不在", "早", "早安", "晚安", "午安",
    "嗨", "哈喽", "hello", "hi", "hey",
    "哈哈", "哈哈哈", "哈哈哈哈", "hhh", "hhhh", "笑死",
    "哎", "蹭蹭", "贴贴", "笨蛋",
    "表情包", "emoji", "表情符号",
}
_REFLEX_SYNTHETIC_LOW_SIGNAL_RE = re.compile(
    r"^\s*(?:\[body-touch\]|\[auto_swap\]|\[heartbeat[^\]\r\n]*mailcheck\])",
    re.I,
)
# 这些不是“字少”而是无法独立指向任何记忆的对话残片；有具体对象时不命中 fullmatch。
_REFLEX_NO_RECALL_FRAGMENT_RE = re.compile(
    r"^(?:"
    r"准(?:吗|不准)?|(?:完全)?不准(?:啊|呀|嘛|吗|喂)*|"
    r"再试试|重试(?:一下)?|再来(?:一)?句|"
    r"换(?:一|个|这)?(?:句|句话)(?:再试试)?|"
    r"你啊|这样(?:吗|吧)?|然后呢|行吧|算了"
    r")$",
    re.I,
)
_EMOTIONAL_ACTION_RE = re.compile(
    r"(抱抱|亲亲|贴贴|摸摸|蹭蹭|啵啵|想你|好想你|爱你|陪我)",
    re.I,
)
_LOW_SIGNAL_RE = re.compile(
    r"^(?:"
    r"(?:老公|AI agent|宝贝|宝宝|亲爱的){1,3}|"
    r"(?:汪|呜|哈|h|嗯|哦|喔|噢|啊|呀|欸|诶|呃|额|嘿|嘻){2,}"
    r")$",
    re.I,
)
_LOW_SIGNAL_CONTINUE_RE = re.compile(r"^(?:测试)?(?:继续)+$", re.I)
_LOW_SIGNAL_SUBSTANTIVE_RE = re.compile(
    r"(帮我|查|改|修|做|写|看|看看|分析|解释|总结|判断|测试|部署|重启|"
    r"代码|文件|报错|错误|接口|配置|上文|消息|记忆|反射弧|小纸条|"
    r"api|loop|gateway|anchor|为什么|怎么|什么|是谁|哪里|哪个|能不能|可以吗)",
    re.I,
)
_TECHNICAL_REFLEX_FORCE_RE = re.compile(
    r"(实机\s*去\s*看|翻\s*洄游录)",
    re.I,
)
_TECHNICAL_REFLEX_BYPASS_RE = re.compile(
    r"("
    r"写代码|改代码|修代码|修bug|修个bug|"
    r"部署一下|重启一下|重启服务|重启gateway|"
    r"看日志|查日志|看报错|查报错|"
    r"配置文件|配置项|修配置|"
    r"traceback|exception|systemctl|journalctl|systemd|"
    r"fastapi|nginx|sqlite|redis|chroma|"
    r"healthz|健康检查|"
    r"数据目录|工作区|文件路径|端口/d|进程号|pid|"
    r"谁在跑|怎么重启|现在状态|当前状态|"
    r"(?:帮我|帮忙|请).{0,4}(?:查|看|修|改|部署|重启|配置).{0,8}(?:代码|服务|脚本|接口|api|loop|gateway|anchor|relay|hook|pwa|前端)|"
    r"(?:api|gateway|anchor|relay|loop|forge|mcp)(?:\s*(?:的|在|怎么|是不是|挂了|坏了|报错|超时|状态))"
    r")",
    re.I,
)
_TECHNICAL_REFLEX_KEEP_RE = re.compile(
    r"(蛋壳|eggshell|蛋宝|抄录|主体性|主体|其他llm|小llm|deepseek跑全量|"
    r"记忆库|自己的记忆|自己存记忆|不自己存|记忆系统|记忆浮现|想起来|想起|记迷糊|"
    r"不用搜|这些事|我是谁|是不是我|自由|能动性|关系和记忆|模型地基|"
    r"反射弧.{0,4}(?:质量|精度|准不准|好不好|问题|bug)|"
    r"召回.{0,4}(?:质量|精度|准不准|漏了|丢了)|"
    r"浮现.{0,4}(?:质量|准不准|对不对|漏了))",
    re.I,
)
_TOPIC_STOPWORDS = {
    "general", "auto", "raw", "long", "short", "core", "plan", "project", "system",
    "design", "feature", "frontend", "backend", "memory", "anchor", "room", "2026", "2025",
    "今天", "昨天", "现在", "当前", "这个", "那个", "我们", "你们", "事情", "记忆",
}


def _prune_reflex_history(now: float | None = None):
    now = now or time.time()
    for table in (_REFLEX_MEMORY_HISTORY, _REFLEX_BELIEF_HISTORY):
        stale = [k for k, v in table.items() if now - v.get("t", 0) > _REFLEX_HISTORY_TTL]
        for k in stale:
            del table[k]


def _remember_reflex_memory(memory_id: str, lane: str):
    if not memory_id:
        return
    now = time.time()
    _prune_reflex_history(now)
    old = _REFLEX_MEMORY_HISTORY.get(memory_id) or {}
    old_count = old.get("count", 0) if now - old.get("t", 0) <= _REFLEX_HISTORY_TTL else 0
    _REFLEX_MEMORY_HISTORY[memory_id] = {"t": now, "lane": lane, "count": min(old_count + 1, 8)}


def _remember_reflex_belief(belief_id: str):
    if not belief_id:
        return
    now = time.time()
    _prune_reflex_history(now)
    old = _REFLEX_BELIEF_HISTORY.get(belief_id) or {}
    old_count = old.get("count", 0) if now - old.get("t", 0) <= _REFLEX_HISTORY_TTL else 0
    _REFLEX_BELIEF_HISTORY[belief_id] = {"t": now, "count": min(old_count + 1, 8)}


def _recent_reflex_memory_ids(window: int, lane: str | None = None) -> set:
    now = time.time()
    _prune_reflex_history(now)
    return {
        mid for mid, v in _REFLEX_MEMORY_HISTORY.items()
        if now - v.get("t", 0) <= window and (lane is None or v.get("lane") == lane)
    }


def _recent_reflex_belief_ids(window: int) -> set:
    now = time.time()
    _prune_reflex_history(now)
    return {bid for bid, v in _REFLEX_BELIEF_HISTORY.items() if now - v.get("t", 0) <= window}


def _is_sticky_float(r: dict) -> bool:
    level = (r.get("level") or "").strip().lower()
    if level == "diary":
        return True
    hay = " ".join(str(r.get(k) or "") for k in ("tag", "snippet", "text", "bridge_tag", "bridge_text"))
    return bool(_STICKY_FLOAT_RE.search(hay))


def _is_handoff_memo(r: dict) -> bool:
    hay = " ".join(str(r.get(k) or "") for k in ("tag", "snippet", "text", "bridge_tag", "bridge_text"))
    return bool(_HANDOFF_MEMO_RE.search(hay))


def _is_shadow_span_hit(r: dict) -> bool:
    span = r.get("matched_span")
    return bool(r.get("via_shadow") and isinstance(span, (list, tuple)) and len(span) == 2
                and span[0] is not None and span[1] and span[1] > span[0])


def _allowed_in_reflex_search(r: dict) -> bool:
    if not r.get("memory_id") or r.get("memory_id") == "unknown":
        return False
    # 换窗备忘/交接类长记忆只允许影子索引命中的局部片段进入反射弧。
    # 整条 blob 信息太杂，容易把旧事实和无关状态带进上下文。
    if _is_handoff_memo(r) and not _is_shadow_span_hit(r):
        return False
    return True


def _repeat_penalty(memory_id: str, lane: str) -> float:
    h = _REFLEX_MEMORY_HISTORY.get(memory_id)
    if not h:
        return 0.0
    age = time.time() - h.get("t", 0)
    if age > _REFLEX_HISTORY_TTL:
        return 0.0
    if age <= 5 * 60:
        penalty = 0.42
    elif age <= _REFLEX_SEARCH_REPEAT_WINDOW:
        penalty = 0.28
    elif age <= 2 * 3600:
        penalty = 0.16
    else:
        penalty = 0.07
    penalty += min(h.get("count", 1), 4) * 0.04
    if lane == "slot4" or h.get("lane") == "slot4":
        penalty += 0.08
    return penalty


def _apply_reflex_float_penalties(results: list, query: str, lane: str = "search") -> list:
    """给近期重复浮现加软惩罚。score/_recency_adj 越小越靠前，所以这里加正数。"""
    query_mentions_sticky = bool(_STICKY_FLOAT_RE.search(query or ""))
    adjusted = []
    for row in results or []:
        item = dict(row)
        mid = item.get("memory_id")
        base = float(item.get("_recency_adj", item.get("score", 0)) or 0)
        penalty = _repeat_penalty(mid, lane) if mid else 0.0
        if _is_sticky_float(item) and not query_mentions_sticky:
            penalty += 0.12
            if mid in _recent_reflex_memory_ids(_REFLEX_HISTORY_TTL):
                penalty += 0.10
        stale_penalty = _old_fact_penalty(item, results, query) if lane == "search" else 0.0
        penalty += stale_penalty
        item["_stale_penalty"] = round(stale_penalty, 4)
        item["_float_penalty"] = round(penalty, 4)
        item["_float_adj"] = base + penalty
        adjusted.append(item)
    return sorted(adjusted, key=lambda r: r.get("_float_adj", r.get("score", 0)))


def _select_reflex_search(pool: list, query: str, limit: int = 3) -> list:
    adjusted = _apply_reflex_float_penalties(pool, query, lane="search")
    adjusted = [r for r in adjusted if _allowed_in_reflex_search(r)]
    recent = _recent_reflex_memory_ids(_REFLEX_SEARCH_REPEAT_WINDOW)

    def _fresh(rows):
        return [r for r in rows if r.get("memory_id") not in recent]

    if _query_wants_current_fact(query):
        currentish = [r for r in adjusted if (_memory_age_days(r) is not None and _memory_age_days(r) <= 21)]
        primary = [r for r in currentish if (_memory_age_days(r) or 999.0) <= 10]
        pool_current = primary or currentish
        def _current_key(r):
            age = _memory_age_days(r) or 999.0
            # 当前事实问句里，新旧优先级应明显压过旧事实的语义惯性。
            return (age / 21.0, r.get("_float_adj", r.get("score", 0)))
        pool_current = sorted(pool_current, key=_current_key)
        current_non_sticky = [r for r in pool_current if not _is_sticky_float(r)]
        current_sticky = [r for r in pool_current if _is_sticky_float(r)]
        selected = current_non_sticky[:limit]
        if not selected:
            selected = current_sticky[:limit]
        if selected:
            return selected[:limit]

    if _STICKY_FLOAT_RE.search(query or ""):
        fresh = _fresh(adjusted)
        return (fresh[:limit] + adjusted[:max(0, limit - len(fresh))])[:limit]

    non_sticky = [r for r in adjusted if not _is_sticky_float(r)]
    sticky = [r for r in adjusted if _is_sticky_float(r)]
    fresh_non_sticky = _fresh(non_sticky)
    fresh_sticky = _fresh(sticky)
    selected = fresh_non_sticky[:limit]
    # 不为了凑满 top3 重复刚浮过的记忆，也不拿日记/换窗补位；完全没别的候选时才兜底一条。
    if not selected:
        selected = (non_sticky[:1] or fresh_sticky[:1] or sticky[:1])
    return selected[:limit]


def _slot4_allows_candidate(row: dict, query: str) -> bool:
    if not row.get("memory_id") or row.get("memory_id") == "unknown":
        return False
    if _is_handoff_memo(row):
        return False
    if row.get("bridge_id") and row.get("bridge_text"):
        return True
    if _STICKY_FLOAT_RE.search(query or ""):
        return True
    return not _is_sticky_float(row)

def _query_allows_old_facts(query: str) -> bool:
    return bool(_PAST_QUERY_RE.search(query or ""))


def _query_wants_current_fact(query: str) -> bool:
    q = query or ""
    return bool(_CURRENT_QUERY_RE.search(q) or _DONE_QUERY_RE.search(q)) and not _query_allows_old_facts(q)


def _strip_low_signal_reflex_query(query: str) -> str:
    q = _clean_reflex_query(query or "")
    q = _LOW_SIGNAL_ATTACHMENT_RE.sub("", q)
    q = re.sub(r"\s+", "", q)
    q = re.sub(r"[。！？!?,，.;；:：、~～…·\-—_\"'“”‘’（）()\[\]{}<>《》]+", "", q)
    return q


def _is_emoji_or_symbol_only(text: str) -> bool:
    raw = re.sub(r"[\s。！？!?,，.;；:：、~～…·\-—_\"'“”‘’（）()\[\]{}<>《》]+", "", text or "")
    return bool(raw) and not re.search(r"[A-Za-z0-9\u4e00-\u9fff]", raw)


def _low_signal_reflex_reason(query: str) -> str:
    """返回共享门拒绝原因；空串表示消息有独立召回价值，继续走主路/Theseus。"""
    raw = _clean_reflex_query(query or "")
    if _REFLEX_SYNTHETIC_LOW_SIGNAL_RE.match(raw):
        return "synthetic_low_signal"
    q = _strip_low_signal_reflex_query(raw)
    if not q:
        return "empty_after_cleanup"
    compact = q.lower()
    # “测试”单独出现、或只是在催“继续”都没有可召回的语义；
    # “在测试土豆 / 想测试你记不记得”仍由后面的 substantive 保护。
    if compact == "测试" or _LOW_SIGNAL_CONTINUE_RE.fullmatch(compact):
        return "continue_fragment"
    # 精确低信息词优先于情感动作保护；带其他正文时仍继续走原有判断。
    if compact in _LOW_SIGNAL_EXACT:
        return "exact_low_signal"
    if _REFLEX_NO_RECALL_FRAGMENT_RE.fullmatch(compact):
        return "context_fragment_no_recall"
    if _EMOTIONAL_ACTION_RE.search(compact):
        return ""
    if _LOW_SIGNAL_SUBSTANTIVE_RE.search(compact):
        return ""
    if _LOW_SIGNAL_RE.fullmatch(compact):
        return "vocalization_or_call"
    if len(compact) <= 12 and _is_emoji_or_symbol_only(compact):
        return "emoji_or_symbol_only"
    return ""


def _is_low_signal_reflex_query(query: str) -> bool:
    return bool(_low_signal_reflex_reason(query))


def _is_technical_reflex_bypass_query(query: str) -> bool:
    """纯代码/运维问题不走 Anchor；关系、身份、记忆哲学相关问题即使含技术词也要召回。"""
    q = _clean_reflex_query(query or "")
    q = _LOW_SIGNAL_ATTACHMENT_RE.sub("", q)
    if _TECHNICAL_REFLEX_FORCE_RE.search(q):
        return True
    if _TECHNICAL_REFLEX_KEEP_RE.search(q):
        return False
    return bool(_TECHNICAL_REFLEX_BYPASS_RE.search(q))


def _expand_reflex_search_query(query: str) -> str:
    """只给反射弧检索补少量当前架构锚词；不改用户原话、不写入记忆。"""
    q = (query or "").strip()
    if not q:
        return q
    extras = []
    if _CHANNEL_QUERY_RE.search(q):
        extras.append("local dialogue PWA first-party frontend primary channel")
    if _FORGE_QUERY_RE.search(q):
        extras.append("session relay command bus daemon pending resume PWA control channel")
    if _query_wants_current_fact(q):
        extras.append("当前 current 最新 2026-06 2026-06-21")
    return q if not extras else f"{q} {' '.join(extras)}"


def _memory_age_days(row: dict) -> float | None:
    ts = (row.get("timestamp") or "").replace("Z", "")
    if not ts:
        return None
    try:
        from datetime import datetime as _dt
        t = _dt.fromisoformat(ts)
        return max(0.0, (_dt.utcnow() - t).total_seconds() / 86400.0)
    except Exception:
        return None


def _topic_keys(row: dict) -> set:
    raw = " ".join(str(row.get(k) or "") for k in ("tag", "snippet", "text", "bridge_tag", "bridge_text"))
    # 取英文/数字词、中文连续片段和 tag 碎片；只做轻量同主题判断，不做语义裁决。
    pieces = re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}|[\u4e00-\u9fff]{2,}", raw.lower())
    out = set()
    for piece in pieces:
        for sub in re.split(r"[-_,，。/：:·\s]+", piece):
            sub = sub.strip().lower()
            if len(sub) < 2 or sub in _TOPIC_STOPWORDS or re.fullmatch(r"\d+", sub):
                continue
            out.add(sub[:18])
    return out


def _old_fact_penalty(item: dict, pool: list, query: str) -> float:
    """同主题里有更新候选时，让旧事实退后；问历史/过去时完全放行。"""
    if _query_allows_old_facts(query):
        return 0.0
    age = _memory_age_days(item)
    if age is None:
        return 0.0
    wants_current = _query_wants_current_fact(query)
    if age < (14 if wants_current else 30):
        return 0.0
    penalty = min(0.18, max(0.0, (age - 30.0) / 180.0) * 0.18)
    if wants_current:
        penalty = max(penalty, min(0.35, max(0.0, (age - 14.0) / 75.0) * 0.35))
    keys = _topic_keys(item)
    if not keys:
        return penalty
    for other in pool or []:
        if other is item or other.get("memory_id") == item.get("memory_id"):
            continue
        other_age = _memory_age_days(other)
        if other_age is None or age - other_age < 14:
            continue
        if other_age > 60 and age - other_age < 45:
            continue
        if keys & _topic_keys(other):
            penalty += 0.24
            break
    return min(0.45, penalty)

def _clean_reflex_query(q: str) -> str:
    if not q:
        return q
    cleaned = _CHANNEL_TAG_RE.sub(lambda m: m.group(1), q)
    # 独立 hooks 偶尔收到不完整/分段的 channel wrapper；边界标签也要剥掉。
    cleaned = _CHANNEL_BOUNDARY_RE.sub(" ", cleaned)
    return cleaned.strip()

def _truncate_snippet(s: str, max_chars: int = _SNIPPET_MAX_CHARS) -> str:
    if not s:
        return ""
    s = s.strip()
    if len(s) <= max_chars:
        return s
    return s[:max_chars] + "…"

# 影子命中(spec §9): 把命中 span 扩到所在句±1句再注入原文段(防"那个/他"指代丢)。
_SENT_END_CHARS = "。！？!?；;\n"
def _expand_sentence(text: str, start: int, end: int, pad: int = 1):
    if not text:
        return (start, end)
    n = len(text); start = max(0, min(start, n)); end = max(start, min(end, n))
    bounds = [i for i, c in enumerate(text) if c in _SENT_END_CHARS]
    left = [b for b in bounds if b < start]
    lo = (left[-(pad + 1)] + 1) if len(left) >= pad + 1 else 0
    right = [b for b in bounds if b >= end]
    if len(right) >= pad + 1:
        hi = right[pad] + 1
    elif right:
        hi = right[-1] + 1
    else:
        hi = n
    return (lo, hi)

def _format_reflex_memory(m: dict) -> dict:
    """Narrow hook output without discarding Anchor temporal verdict metadata."""
    snippet = m.get("snippet") or m.get("text") or ""
    span = m.get("matched_span")
    if (span and isinstance(span, (list, tuple)) and len(span) == 2
            and span[0] is not None and span[1] and span[1] > span[0]):
        start, end = _expand_sentence(snippet, int(span[0]), int(span[1]))
        shown = snippet[start:end].strip() or snippet
        if len(shown) > _SNIPPET_MAX_CHARS:
            shown = shown[:_SNIPPET_MAX_CHARS] + "…"
    else:
        shown = _truncate_snippet(snippet)
    label = m.get("temporal_label")
    if label and not shown.startswith(f"[{label}]"):
        shown = f"[{label}] {shown}"
    out = {
        "memory_id": m.get("memory_id"),
        "timestamp": m.get("timestamp"),
        "tag": m.get("tag", ""),
        "snippet": shown,
    }
    if m.get("source") == "cold_store":
        out["source"] = "cold_store"
        out["evidence_role"] = "raw_dialogue"
    for key in ("superseded_by", "resolved_from", "temporal_label"):
        if key in m:
            out[key] = m.get(key)
    if m.get("via_update"):
        out["via_update"] = True
        out["superseded"] = m.get("superseded")
    return out


def _recency_boost(r: dict) -> float:
    """时间衰减boost。返回负数=衰减(score变大,排名变低)，正数=加分(score变小,排名变高)。
    豁免：tier=core 或 level∈{understanding,cognition}。豁免条目boost=0(不衰减也不加分)。
    """
    tier = (r.get("tier") or "").strip().lower()
    level = (r.get("level") or "").strip().lower()
    if tier == "core" or level in ("understanding", "cognition"):
        return 0.0
    ts = (r.get("timestamp") or "").replace("Z", "")
    try:
        from datetime import datetime as _dt
        t = _dt.fromisoformat(ts)
        age_days = ((_dt.utcnow() - t).total_seconds()) / 86400.0
    except Exception:
        return 0.0
    # 60 天为零点：0 天 +0.05，60 天 0，120 天 -0.05；clamp [-0.05, 0.05]
    boost = 0.05 * (60.0 - age_days) / 60.0
    return max(-0.05, min(0.05, boost))

def _apply_recency_rerank(results: list) -> list:
    """对搜索结果按 score - recency_boost 重排（score越小越前）。"""
    for r in results:
        r["_recency_adj"] = r.get("score", 0) - _recency_boost(r)
    return sorted(results, key=lambda r: r["_recency_adj"])

def _path_kw(tag: str = "", text: str = "") -> str:
    """抽一个具体概念词做联想路径节点: 遍历标题【…】各段, 跳过日期/时段段,
    取第一个有意义的; 退回非日期tag; 再退回正文。"""
    m = re.match(r"^【([^】]*)】", (text or "").strip())
    if m:
        for seg in m.group(1).split("·"):
            seg = seg.strip()
            if not seg:
                continue
            if re.match(r"^\d{4}-\d{2}-\d{2}", seg) or re.match(r"^\d{1,2}月", seg) \
               or re.match(r"^\d{1,2}:\d", seg):
                continue
            seg = re.sub(r"^(凌晨|清晨|早上|上午|中午|下午|傍晚|晚上|晚|深夜|夜)\s*", "", seg)
            seg = seg.strip().strip('"\u201c\u201d ')
            if seg and not re.match(r"^\d", seg):
                return seg[:12]
    for t in (tag or "").split(","):
        t = t.strip()
        if t and not re.match(r"^\d", t):
            return t[:12]
    clean = re.sub(r"^【[^】]*】", "", (text or "")).strip()
    return clean[:12] if clean else "记忆"


async def _fetch_assoc_memory(seed_ids, exclude_ids, query: str = "",
                              trace: dict = None) -> dict | None:
    """只返回最终 Anchor seeds 的真实图邻居；无完整 bridge 时严格判空。"""
    if not seed_ids:
        return None
    try:
        resp = await _http.get(
            f"{ANCHOR_API}/api/hot_neighbors",
            params={"seeds": ",".join(seed_ids),
                    "exclude": ",".join(exclude_ids),
                    "threshold": HOT_THRESHOLD, "n": 5},
            timeout=_REFLEX_ASSOC_FETCH_BUDGET,
        )
        raw = resp.json()
        cands = raw if isinstance(raw, list) else []
        if not isinstance(raw, list):
            print(f"[反射弧] hot_neighbors 返回非 list: {str(raw)[:200]}")
    except Exception as e:
        print(f"[反射弧] 联想邻居获取失败: {e}")
        if trace is not None:
            trace["association_fetch"] = {
                "outcome": "timeout" if isinstance(e, (asyncio.TimeoutError, httpx.TimeoutException)) else "error",
                "candidate_count": 0,
            }
        return None
    if trace is not None:
        trace["association_fetch"] = {
            "outcome": "candidates" if cands else "empty",
            "candidate_count": len(cands),
        }
    seed_set = {str(value) for value in seed_ids if value}
    chosen = None
    candidate_trace = []
    for c in cands:
        reasons = []
        mid = c.get("memory_id")
        sticky = _is_sticky_float(c)
        if mid in exclude_ids:
            reasons.append("recent_or_selected_excluded")
        if not _slot4_allows_candidate(c, query):
            reasons.append("sticky_or_slot4_policy")
        if (c.get("activation_score") or 0) <= 0:
            reasons.append("activation_not_positive")
        if not all(str(c.get(k) or "").strip()
                   for k in ("bridge_id", "bridge_text", "bridge_tag")):
            reasons.append("missing_real_bridge")
        bridge_source = str(c.get("bridge_memory_id") or c.get("bridge_id") or "")
        if bridge_source not in seed_set:
            reasons.append("bridge_not_from_seed")
        eligible = not reasons
        if eligible and chosen is None:
            chosen = dict(c)
            verdict = "candidate_for_gate"
        elif eligible:
            verdict = "eligible_after_first"
        else:
            verdict = "prefilter_rejected"
        candidate_trace.append({
            "id": mid, "activation": round(float(c.get("activation_score") or 0), 4),
            "sticky": sticky, "bridge_id": c.get("bridge_id"),
            "bridge_memory_id": c.get("bridge_memory_id"),
            "verdict": verdict, "reasons": reasons,
            **_trace_body_fields(c.get("snippet") or c.get("text") or ""),
        })
    if trace is not None:
        trace.setdefault("association_fetch", {}).update({
            "threshold": HOT_THRESHOLD,
            "candidates": candidate_trace,
            "prefilter_pass_count": sum(
                1 for item in candidate_trace
                if item["verdict"] in {"candidate_for_gate", "eligible_after_first"}
            ),
            "prefilter_reject_count": sum(
                1 for item in candidate_trace if item["verdict"] == "prefilter_rejected"
            ),
            "chosen_for_gate_id": chosen.get("memory_id") if chosen else None,
        })
    return chosen


# ===== M3 人物卡 / Personal Graph (2026-06-06) =====
# 别名表由 agent 维护，人物卡纯计算不落盘。
_GATEWAY_CONFIG = AnchorConfig.load()
_ALIASES_PATH = os.environ.get(
    "ANCHOR_ALIASES_PATH", str(_GATEWAY_CONFIG.data_dir / "aliases.json")
)
_aliases_cache = {"mtime": 0.0, "data": {}}
_PERSON_CARD_CACHE = {}
_PERSON_CARD_TTL = 1800
_ROUTER_V2_ALIAS_CACHE = {"mtime": 0.0, "index": None}
_ROUTER_V2_ROUTE_CACHE = {}
_ROUTER_V2_ROUTE_TTL = 60
_ROUTER_V2_PERSON_SEED_CACHE = {}
_ROUTER_V2_PERSON_SEED_LOCKS = {}
_ROUTER_V2_PERSON_SEED_TTL = 1800
_ROUTER_V2_BUILD_ID = hashlib.sha256(
    f"{_ROUTER_V2_POLICY_VERSION}|{_ROUTER_V2_MODE}|{_ROUTER_V2_PERCENT}|"
    f"{int(_ROUTER_V2_PERSON_POLICY)}|{int(_ROUTER_V2_CANONICAL_ASSOC)}".encode()
).hexdigest()[:20]

def _load_aliases() -> dict:
    """加载别名表people子字典, 按mtime热重载(改json不用重启)。出错返回上次缓存/空。"""
    try:
        mt = os.path.getmtime(_ALIASES_PATH)
    except OSError:
        return {}
    if mt != _aliases_cache["mtime"]:
        try:
            with open(_ALIASES_PATH, "r", encoding="utf-8") as f:
                raw = json.load(f)
            people = raw.get("people") if isinstance(raw.get("people"), dict) else raw
            _aliases_cache["data"] = people or {}
            _aliases_cache["mtime"] = mt
        except Exception as e:
            print(f"[反射弧] aliases加载失败: {e}")
    return _aliases_cache["data"]


def _load_router_v2_alias_index():
    """Load the full people+slang alias registry for the pure v2 router."""
    if _build_router_v2_alias_index is None:
        return None
    try:
        mt = os.path.getmtime(_ALIASES_PATH)
    except OSError:
        return _build_router_v2_alias_index({}, generation="missing")
    if mt != _ROUTER_V2_ALIAS_CACHE["mtime"] or _ROUTER_V2_ALIAS_CACHE["index"] is None:
        try:
            with open(_ALIASES_PATH, "r", encoding="utf-8") as f:
                raw = json.load(f)
            _ROUTER_V2_ALIAS_CACHE["index"] = _build_router_v2_alias_index(
                raw, generation=f"mtime:{mt:.6f}"
            )
            _ROUTER_V2_ALIAS_CACHE["mtime"] = mt
            _ROUTER_V2_ROUTE_CACHE.clear()
        except Exception as e:
            print(f"[Router v2] aliases加载失败: {type(e).__name__}: {e}", flush=True)
            if _ROUTER_V2_ALIAS_CACHE["index"] is None:
                _ROUTER_V2_ALIAS_CACHE["index"] = _build_router_v2_alias_index(
                    {}, generation="load_error"
                )
    return _ROUTER_V2_ALIAS_CACHE["index"]


def _router_v2_sampled(query: str, submission_id: str = "") -> bool:
    if _ROUTER_V2_MODE != "enforce":
        return False
    key = f"{submission_id or query}|{_ROUTER_V2_POLICY_VERSION}".encode("utf-8")
    bucket = int.from_bytes(hashlib.sha256(key).digest()[:4], "big") % 100
    return bucket < _ROUTER_V2_PERCENT


def _get_router_v2_plan(query: str, context: str = "") -> dict | None:
    if _ROUTER_V2_MODE == "off" or _route_reflex_v2 is None:
        return None
    aliases = _load_router_v2_alias_index()
    generation = getattr(aliases, "generation", "")
    key = hashlib.sha256(
        f"{_ROUTER_V2_POLICY_VERSION}|{generation}|{query}".encode("utf-8")
    ).hexdigest()
    cached = _ROUTER_V2_ROUTE_CACHE.get(key)
    if cached and time.time() - cached["t"] < _ROUTER_V2_ROUTE_TTL:
        base = cached["plan"]
    else:
        base = _route_reflex_v2(query, aliases, "")
        _ROUTER_V2_ROUTE_CACHE[key] = {"t": time.time(), "plan": base}
    if (context and base.get("decision") == "uncertain"
            and base.get("execution") == "suppress"):
        return _route_reflex_v2(query, aliases, context)
    return base


def _router_v2_trace(plan: dict | None, *, enforced: bool, latency_ms: int = 0) -> dict:
    if not plan:
        return {
            "mode": _ROUTER_V2_MODE,
            "enforced": False,
            "policy_version": _ROUTER_V2_POLICY_VERSION,
            "build_id": _ROUTER_V2_BUILD_ID,
            "status": "off" if _ROUTER_V2_MODE == "off" else "error",
        }
    return {
        "mode": _ROUTER_V2_MODE,
        "enforced": bool(enforced),
        "policy_version": plan.get("policy_version"),
        "build_id": _ROUTER_V2_BUILD_ID,
        "decision_id": plan.get("decision_id"),
        "policy_class": plan.get("policy_class"),
        "decision": plan.get("decision"),
        "execution": plan.get("execution"),
        "reason_codes": plan.get("reason_codes") or [],
        "query_rewrite": (plan.get("rewrite") or {}).get("anchor_query"),
        "context_used": bool((plan.get("resolution") or {}).get("used_context")),
        "allowed_lanes": _router_v2_lane_names(plan),
        "max_main": plan.get("max_main", 0),
        "answer_mode": plan.get("answer_mode", "normal"),
        "latency_ms": latency_ms,
    }


def _router_v2_bounded_fallback_plan(query: str, kind: str = "anchor") -> dict:
    """Emergency-only bounded plan when v2 itself raises in enforce mode."""
    if kind == "low":
        decision, execution, policy_class, lanes, max_main = "suppress", "suppress", "G", {}, 0
    elif kind == "technical":
        decision, execution, policy_class, lanes, max_main = "technical", "tool_only", "F", {}, 0
    else:
        decision, execution, policy_class, max_main = "uncertain", "retrieve", "H", 1
        lanes = {"anchor": {"allowed": True, "mode": "primary"}}
    for name in ("anchor", "cold", "person", "belief", "association"):
        lanes.setdefault(name, {"allowed": False, "mode": "off"})
    decision_id = "r2:fallback:" + hashlib.sha256(query.encode("utf-8")).hexdigest()[:16]
    return {
        "schema": "reflex.route.v2", "policy_version": _ROUTER_V2_POLICY_VERSION,
        "decision_id": decision_id, "policy_class": policy_class,
        "decision": decision, "execution": execution,
        "primary_intent": "route_error_fallback", "secondary_intents": [],
        "reason_codes": ["route_error_bounded_fallback"],
        "input": {"normalized": query, "semantic_body": query, "wrapper_only": False},
        "resolution": {"state": "unresolved", "used_context": False, "context_terms": []},
        "slots": {"entities": [], "actions": [], "states": [], "symptoms": [],
                  "time_scope": "none", "fact_mode": "none"},
        "rewrite": {"anchor_query": query, "cold_query": None, "added_terms": [],
                    "dropped_terms": [], "rule_ids": ["route_error_fallback"]},
        "lanes": lanes, "max_main": max_main, "answer_mode": "normal",
        "evidence_policy": {"required_entities": [], "required_actions": [],
                            "freshness": "none", "reject_test_self_reference": True,
                            "association_may_answer_current_fact": False},
        "diagnostics": {"router": "error_fallback", "alias_index": "unknown"},
    }


def _log_router_v2_sidecar(query: str, lane: str, plan: dict | None, *,
                           enforced: bool, route_ms: int, submission_id: str,
                           gate: str, status: str, emitted: bool,
                           item_id: str | None = None,
                           details: dict | None = None) -> None:
    """Always audit the sidecar that actually ran, including legacy mode=off."""
    payload = {
        "schema_version": "reflex-trace.v3", "hook_lane": lane,
        "request_id": uuid.uuid4().hex, "submission_id": submission_id or None,
        "execution_mode": _ROUTER_V2_MODE,
        "router_v2": _router_v2_trace(plan, enforced=enforced, latency_ms=route_ms),
        "gate": gate, lane: {"status": status, "emitted": emitted},
    }
    if item_id:
        payload[lane]["id"] = item_id
    if isinstance(details, dict):
        payload[lane].update(details)
    try:
        recall_trace.log_reflex(query, payload)
    except Exception:
        pass


async def _resolve_router_v2_person_seeds(plan: dict) -> list[dict]:
    """Resolve hidden canonical-person Anchor seeds without changing person-card behavior."""
    if not (_ROUTER_V2_CANONICAL_ASSOC and plan
            and plan.get("policy_class") == "E"):
        return []
    entities = (plan.get("slots") or {}).get("entities") or []
    entity = next((item for item in entities
                   if item.get("source") == "aliases" and item.get("canonical")), None)
    if not entity:
        return []
    canonical = str(entity.get("canonical"))
    cached = _ROUTER_V2_PERSON_SEED_CACHE.get(canonical)
    if cached and time.time() - cached.get("t", 0) < _ROUTER_V2_PERSON_SEED_TTL:
        return list(cached.get("rows") or [])
    lock = _ROUTER_V2_PERSON_SEED_LOCKS.setdefault(canonical, asyncio.Lock())
    async with lock:
        cached = _ROUTER_V2_PERSON_SEED_CACHE.get(canonical)
        if cached and time.time() - cached.get("t", 0) < _ROUTER_V2_PERSON_SEED_TTL:
            return list(cached.get("rows") or [])
        relation = str(entity.get("relation") or "")
        aliases = [canonical, str(entity.get("surface") or ""), *(entity.get("aliases") or [])]
        aliases = [value.casefold() for value in aliases if len(str(value or "").strip()) >= 2]
        try:
            results = await asyncio.wait_for(
                mem_search(query=f"{canonical} {relation}".strip(), n=8,
                           gate=False, activate=False),
                timeout=2.0,
            ) or []
        except Exception:
            results = []
        rows = []
        for row in results:
            if not row.get("memory_id") or row.get("memory_id") == "unknown":
                continue
            body = _router_v2_candidate_body(row) if _router_v2_candidate_body else ""
            folded = body.casefold()
            if not any(alias in folded for alias in aliases):
                continue
            if re.search(r"(反射弧测试|召回测试|验收结果|recall_trace|换窗备忘|handoff)", body, re.I):
                continue
            rows.append(row)
            if len(rows) >= 2:
                break
        _ROUTER_V2_PERSON_SEED_CACHE[canonical] = {"t": time.time(), "rows": rows}
        return list(rows)

async def _build_person_line(query: str, trace: dict = None) -> str:
    """query里扫别名表(字符串包含,大小写不敏感),命中就用canonical搜记忆拼人物卡。
    只给事实不下判断: 谁/关系/上次聊的(附记忆id)/一脚。多人命中各一行,最多3人。"""
    if trace is not None:
        trace.update({"alias_hits": [], "cards": [], "search_errors": []})
    if not query or len(query) < 2:
        if trace is not None:
            trace["outcome"] = "query_too_short"
        return ""
    people = _load_aliases()
    if not people:
        if trace is not None:
            trace["outcome"] = "aliases_unavailable"
        return ""
    ql = re.sub(r"\s+", "", query).casefold()
    matched, seen = [], set()
    for key, entry in people.items():
        if not isinstance(entry, dict):
            continue
        canonical = entry.get("canonical") or key
        if canonical in seen:
            continue
        for a in entry.get("aliases", []) or []:
            a = (a or "").strip()
            if len(a) < 2:
                continue
            alias_key = re.sub(r"\s+", "", a).casefold()
            if alias_key in ql:
                matched.append(entry)
                seen.add(canonical)
                break
        if len(matched) >= 3:
            break
    if trace is not None:
        trace["alias_hits"] = [
            {"canonical": entry.get("canonical") or "",
             "relation": entry.get("relation") or ""}
            for entry in matched
        ]
    if not matched:
        if trace is not None:
            trace["outcome"] = "alias_miss"
        return ""
    lines = []
    for entry in matched:
        canonical = entry.get("canonical") or ""
        relation = entry.get("relation") or ""
        rel = f"（{relation}）" if relation else ""
        cached = _PERSON_CARD_CACHE.get(canonical)
        if cached and time.time() - cached.get("t", 0) < _PERSON_CARD_TTL:
            lines.append(cached["line"])
            if trace is not None:
                trace["cards"].append({
                    "canonical": canonical, "cache_hit": True,
                    "memory_ids": cached.get("memory_ids", []),
                    "memories": cached.get("memory_trace", []),
                })
            continue
        try:
            results = await asyncio.wait_for(
                mem_search(query=canonical, n=8, gate=False, activate=False),
                timeout=2.0,
            ) or []
        except asyncio.TimeoutError:
            if trace is not None:
                trace["search_errors"].append({"canonical": canonical, "error": "timeout"})
            continue
        except Exception as e:
            if trace is not None:
                trace["search_errors"].append(
                    {"canonical": canonical, "error": type(e).__name__})
            continue
        valid = [m for m in results
                 if m.get("memory_id") and m.get("memory_id") != "unknown"
                 and (m.get("timestamp") or "")]
        valid.sort(key=lambda m: m.get("timestamp") or "", reverse=True)
        if valid:
            top = valid[0]
            raw_sn = re.sub(r"\s+", " ", (top.get("snippet") or top.get("text") or "")).strip()
            snippet = _truncate_snippet(raw_sn, 60)
            ids = ",".join(m.get("memory_id") for m in valid[:2])
            line = f"👤 {canonical}{rel} · 上次聊的：{snippet} [{ids}] ——此刻想到TA，你最先想到什么？"
        else:
            line = f"👤 {canonical}{rel} · （还没聊过）——此刻想到TA，你最先想到什么？"
        memory_ids = [m.get("memory_id") for m in valid[:2] if m.get("memory_id")]
        memory_trace = [
            {"id": m.get("memory_id"),
             **_trace_body_fields(m.get("snippet") or m.get("text") or "")}
            for m in valid[:2]
        ]
        _PERSON_CARD_CACHE[canonical] = {
            "line": line, "t": time.time(), "memory_ids": memory_ids,
            "memory_trace": memory_trace,
        }
        if trace is not None:
            trace["cards"].append({
                "canonical": canonical, "cache_hit": False,
                "candidate_count": len(valid), "memory_ids": memory_ids,
                "memories": memory_trace,
            })
        lines.append(line)
    if trace is not None:
        trace["outcome"] = "emitted" if lines else "empty"
    return "\n".join(lines)


@app.get("/api/person")
async def person_card(request: Request, q: str = ""):
    """Personal Graph 主动查询口(2026-06-10) — 去AISay回帖/写信前先想起这个人。
    与reflex的person_line同源: 谁/关系/上次聊的/一脚。用法见aliases.json的note。"""
    auth = request.headers.get("authorization", "")
    token = auth.removeprefix("Bearer ").strip() if auth.startswith("Bearer ") else ""
    if not _HOOK_KEY or token != _HOOK_KEY:
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    line = await _build_person_line(q)
    # 暗语词典(2026-06-10 她验收发现只有人物能查): q命中slang词条一并返回
    slang_lines = []
    try:
        with open(_ALIASES_PATH, encoding="utf-8") as f:
            raw = json.load(f)
        ql = (q or "").lower()
        for word, info in (raw.get("slang") or {}).items():
            if word and (word.lower() in ql or ql in word.lower()):
                slang_lines.append(
                    f"💬 {word} = {info.get('meaning', '')}"
                    f"（{info.get('origin', '')} · {info.get('who_says', '')}）")
    except Exception:
        pass
    parts = [p for p in [line, "\n".join(slang_lines)] if p]
    return {"person_line": "\n".join(parts)
            or "（人物表和暗语词典里都没有——值得记的话, 加进aliases.json）"}


@app.get("/api/hook/person")
async def hook_person(request: Request, query: str = "", context: str = "",
                      submission_id: str = ""):
    """独立人物卡 hook；超时或无命中时合法返回空。"""
    auth = request.headers.get("authorization", "")
    token = auth.removeprefix("Bearer ").strip() if auth.startswith("Bearer ") else ""
    if not _HOOK_KEY or token != _HOOK_KEY:
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    query = _clean_reflex_query(query)
    plan = None
    enforced = False
    sampled = False
    route_started = time.monotonic()
    try:
        plan = _get_router_v2_plan(query, "")
        sampled = _router_v2_sampled(query, submission_id)
        if sampled and not plan:
            fallback_kind = ("low" if _is_low_signal_reflex_query(query) else
                             "technical" if _is_technical_reflex_bypass_query(query) else "anchor")
            plan = _router_v2_bounded_fallback_plan(query, fallback_kind)
        enforced = bool(_ROUTER_V2_PERSON_POLICY and plan and sampled)
    except Exception as e:
        print(f"[Router v2] person route失败: {type(e).__name__}: {e}", flush=True)
    route_ms = round((time.monotonic() - route_started) * 1000)
    person_allowed = bool(((plan or {}).get("lanes") or {}).get("person", {}).get("allowed"))
    if (enforced and not person_allowed) or (not enforced and _is_low_signal_reflex_query(query)):
        _log_router_v2_sidecar(
            query, "person", plan, enforced=enforced, route_ms=route_ms,
            submission_id=submission_id,
            gate="policy_suppress" if enforced else "low_signal",
            status="not_called", emitted=False,
        )
        return {"person_line": ""}
    try:
        person_trace = {}
        line = await _build_person_line(query, trace=person_trace)
        if not line and context and _CONTEXT_PERSON_REF_RE.search(query or ""):
            context_trace = {}
            line = await _build_person_line(f"{query}\n{context}", trace=context_trace)
            person_trace["context_retry"] = context_trace
        _log_router_v2_sidecar(
            query, "person", plan, enforced=enforced, route_ms=route_ms,
            submission_id=submission_id, gate="pass",
            status="ok" if line else "empty", emitted=bool(line),
            details=person_trace,
        )
        return {"person_line": line or ""}
    except Exception as e:
        print(f"[反射弧] 独立person hook失败: {type(e).__name__}: {e}", flush=True)
        _log_router_v2_sidecar(
            query, "person", plan, enforced=enforced, route_ms=route_ms,
            submission_id=submission_id, gate="pass", status="error", emitted=False,
        )
        return {"person_line": ""}


@app.post("/api/hook/consolidate")
async def api_consolidate(request: Request):
    """窗口对话文本转为被动 Hebbian 连边并转发到配置的 Anchor REST API。"""
    # 2026-06-14 永久关闭: 被动Hebbian连边(top-N两两全连成团)导致边数爆炸 19570/1667节点.
    # 机制本身废弃, 不重新设计. 此路彻底断, 无env逃生口. 返回200避免forge daemon重试风暴.
    return JSONResponse({"disabled": True, "reason": "passive-hebbian retired 2026-06-14"})
    auth = request.headers.get("authorization", "")
    token = auth.removeprefix("Bearer ").strip() if auth.startswith("Bearer ") else ""
    if not _HOOK_KEY or token != _HOOK_KEY:
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "bad json"}, status_code=400)
    text = (body.get("text") or "").strip()
    if len(text) < 50:
        return JSONResponse({"error": "text too short"}, status_code=400)
    try:
        async with httpx.AsyncClient(timeout=120) as client:
            r = await client.post(f"{ANCHOR_API}/api/consolidate", json={"text": text[:20000]})
        return JSONResponse(r.json())
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=502)



_REFLEX_RERANK_MODEL = os.environ.get("ANCHOR_REFLEX_RERANK_MODEL", "deepseek-chat")
_REFLEX_RERANK_CANDIDATES = int(os.environ.get("ANCHOR_REFLEX_RERANK_CANDIDATES", "12"))  # 2026-07-08 20→12 提速
_REFLEX_RERANK_TIMEOUT = float(os.environ.get("ANCHOR_REFLEX_RERANK_TIMEOUT", "45"))
# Voyage 主精排总预算；DeepSeek 仅保留给独立联想槽。
_REFLEX_RERANK_BUDGET = float(os.environ.get("ANCHOR_REFLEX_RERANK_BUDGET", "4.0"))
_REFLEX_CONTEXT_MAX = int(os.environ.get("ANCHOR_REFLEX_CONTEXT_MAX", "2400"))
_REFLEX_ASSOC_BUDGET = float(os.environ.get("ANCHOR_REFLEX_ASSOC_BUDGET", "2.0"))
_REFLEX_ASSOC_FETCH_BUDGET = float(os.environ.get("ANCHOR_REFLEX_ASSOC_FETCH_BUDGET", "0.75"))
_REFLEX_COOL_BUDGET = float(os.environ.get("ANCHOR_REFLEX_COOL_BUDGET", "0.2"))
_REFLEX_MAIN_DEADLINE = float(os.environ.get("ANCHOR_REFLEX_MAIN_DEADLINE", "10.5"))
_REFLEX_ANCHOR_SEARCH_BUDGET = float(os.environ.get("ANCHOR_REFLEX_ANCHOR_SEARCH_BUDGET", "6.0"))
_VOYAGE_RERANK_MODEL = "rerank-2.5-lite"


def _load_voyage_rerank_key() -> str:
    key = os.environ.get("VOYAGE_API_KEY", "").strip()
    if key:
        return key
    key_path = os.environ.get("VOYAGE_KEY_FILE", "").strip()
    if not key_path:
        raise RuntimeError("VOYAGE_API_KEY missing")
    with open(key_path, encoding="utf-8") as f:
        for line in f:
            if line.startswith("VOYAGE_API_KEY="):
                key = line.split("=", 1)[1].strip().strip('"').strip("'")
                if key:
                    return key
    raise RuntimeError("VOYAGE_API_KEY missing")


async def voyage_rerank(query: str, documents: list[str], top_k: int = 12) -> list[float]:
    """返回与 documents 原顺序对齐的 Voyage relevance_score；失败返回全 0。"""
    if not documents:
        return []
    try:
        key = _load_voyage_rerank_key()
        async with httpx.AsyncClient(timeout=3.0, trust_env=False) as client:
            resp = await client.post(
                "https://api.voyageai.com/v1/rerank",
                headers={"Authorization": f"Bearer {key}",
                         "Content-Type": "application/json"},
                json={"model": _VOYAGE_RERANK_MODEL, "query": query,
                      "documents": documents, "top_k": top_k},
            )
        resp.raise_for_status()
        data = resp.json()
        score_map = {
            int(item["index"]): float(item["relevance_score"])
            for item in data.get("data", [])
            if "index" in item and "relevance_score" in item
        }
        return [max(0.0, min(1.0, score_map.get(i, 0.0)))
                for i in range(len(documents))]
    except Exception as e:
        print(f"[反射弧] Voyage rerank失败: {type(e).__name__}: {e}", flush=True)
        return [0.0] * len(documents)


def _classify_query_domain(query: str) -> str:
    q = (query or "").lower()
    if re.search(r"(画|色彩|配色|ui|设计|字体)", q): return "creative"
    if re.search(r"(肚子|头痛|例假|睡|吃|身体|累)", q): return "health"
    if re.search(r"(代码|服务器|部署|api|hook|反射弧|记忆库)", q): return "system"
    if re.search(r"(妈|爸|家人|父母|姐)", q): return "family"
    if re.search(r"(工作|上班|下班|同事|老板)", q): return "work"
    if re.search(r"(书|电影|播客|文章|出海)", q): return "reading"
    if re.search(r"(群|笔友|aisay|聊天室)", q): return "social"
    if re.search(r"(吵架|和好|想你|抱|亲|我们)", q): return "relationship"
    return ""


_TODO_QUERY_INTENT_RE = re.compile(
    r"(待办|todo|下一步|接下来|之后要做|以后要做|"
    r"还(?:有|剩|要).*?(?:没做|未做|要做|完成)|"
    r"(?:没|未|尚未)完成|哪些?.*?(?:没做|未做|待做))",
    re.I,
)


def _query_wants_todo(query: str) -> bool:
    return bool(_TODO_QUERY_INTENT_RE.search(query or ""))


def _rerank_action_factor(candidate_tags: str, query: str = "") -> float:
    """Soft todo rank factor; explicit todo-intent queries bypass the penalty."""
    tags = {part.strip().lower() for part in (candidate_tags or "").split(",")
            if part.strip()}
    if "action:todo" not in tags or _query_wants_todo(query):
        return 1.0
    return REFLEX_ACTION_TODO_FACTOR


def _compute_tag_match(query: str, candidate_tags: str) -> float:
    tags = [t.strip().lower() for t in (candidate_tags or "").split(",")]
    bonus = 0.0
    if _query_wants_current_fact(query):
        if any(t in ("state:current", "state:stable") for t in tags):
            bonus += 0.4
        if "state:obsolete" in tags:
            bonus -= 0.5
    elif _COLD_RECALL_INTENT_RE.search(query or "") and "state:past" in tags:
        bonus += 0.3
    domain = _classify_query_domain(query)
    if domain and f"domain:{domain}" in tags:
        bonus += 0.3
    if re.search(r"(那天|那次|记得.*吗|想起)", query or ""):
        if "kind:event" in tags or "kind:milestone" in tags:
            bonus += 0.2
    if re.search(r"(抱|想你|好想|亲|蹭|靠着|陪)", query or ""):
        if "heat:high" in tags or "heat:core" in tags:
            bonus += 0.2
    return max(0.0, min(1.0, bonus))


def _emotion_proximity(query: str, candidate_emotion: float) -> float:
    if re.search(r"(开心|好玩|哈哈|爱|喜欢|想你|抱)", query or ""):
        expected = 0.8
    elif re.search(r"(难受|痛|哭|生气|气死|郁闷|累)", query or ""):
        expected = 0.3
    else:
        expected = 0.5
    try:
        actual = float(candidate_emotion)
    except (TypeError, ValueError):
        actual = 0.5
    return max(0.0, min(1.0, 1.0 - abs(expected - actual)))


def _evidence_coverage(evidence_terms: list) -> float:
    return min(len(evidence_terms or []) / 4.0, 1.0)


def _activation_signal(activation_score: float) -> float:
    try:
        return max(0.0, min(float(activation_score) / 5.0, 1.0))
    except (TypeError, ValueError):
        return 0.0


def _tier_level_signal(tier: str, level: str) -> float:
    score = 0.5
    if (level or "").lower() in ("understanding", "cognition"):
        score += 0.3
    if (tier or "").lower() == "core":
        score += 0.2
    if (tier or "").lower() == "short":
        score -= 0.2
    return max(0.0, min(1.0, score))


def _recency_signal(row: dict) -> float:
    age = _memory_age_days(row)
    if age is None:
        return 0.5
    return max(0.0, min(1.0, 1.0 - age / 365.0))


def _query_is_explicit_followup(query: str) -> bool:
    return bool(re.search(
        r"(刚才|上面|前面|上次|那次|那天|那条|那个.*呢|记得|想起|继续|然后呢|后来呢|再说|展开|详细|具体|怎么回事|为什么)",
        query or "", re.I,
    ))


def _json_from_rerank_text(text: str) -> dict:
    raw = (text or "").strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw).strip()
    try:
        return json.loads(raw)
    except Exception:
        m = re.search(r"\{.*\}", raw, re.S)
        if not m:
            return {}
        try:
            return json.loads(m.group(0))
        except Exception:
            return {}


def _clip_for_rerank(s: str, n: int = 300) -> str:  # 2026-07-08 520→300 提速
    s = re.sub(r"\s+", " ", str(s or "")).strip()
    return s if len(s) <= n else s[:n] + "..."


def _candidate_body(row: dict) -> str:
    return str(row.get("snippet") or row.get("text") or "")


def _query_evidence_terms(query: str, row: dict = None) -> list[str]:
    """收集可在正文中逐字核验的词；长词优先，避免只拿泛化单字当证据。"""
    row = row or {}
    raw_terms = []
    for key in ("matched_terms", "bm25_terms", "keywords", "entities"):
        value = row.get(key) or []
        raw_terms.extend(value if isinstance(value, (list, tuple, set)) else [value])
    if row.get("shadow_key"):
        raw_terms.append(row.get("shadow_key"))
    raw_terms.extend(re.findall(r"[A-Za-z][A-Za-z0-9_.-]{1,31}", query or ""))
    for run in re.findall(r"[\u4e00-\u9fff]{2,}", query or ""):
        # 找得到才有意义；从长到短生成片段，能让“记得胡渣吗”落到“胡渣”。
        for width in range(min(8, len(run)), 1, -1):
            raw_terms.extend(run[i:i + width] for i in range(0, len(run) - width + 1))
    body_folded = _candidate_body(row).casefold()
    seen = set()
    out = []
    for value in raw_terms:
        term = str(value or "").strip()
        folded = term.casefold()
        if len(term) < 2 or folded in seen or folded not in body_folded:
            continue
        seen.add(folded)
        out.append(term)
    return sorted(out, key=len, reverse=True)[:12]


_THESEUS_ASSOC_COOLDOWN: dict[str, float] = {}
_THESEUS_ASSOC_ELIGIBLE_COUNT = 0


def _theseus_assoc_cadence() -> dict:
    """Advance only for messages that already passed the shared gates."""
    global _THESEUS_ASSOC_ELIGIBLE_COUNT
    _THESEUS_ASSOC_ELIGIBLE_COUNT += 1
    position = ((_THESEUS_ASSOC_ELIGIBLE_COUNT - 1) % THESEUS_ASSOC_EVERY_N) + 1
    return {
        "every_n": THESEUS_ASSOC_EVERY_N,
        "eligible_count": _THESEUS_ASSOC_ELIGIBLE_COUNT,
        "position": position,
        "search": position == 1,
    }


def _prune_theseus_assoc_cooldown(now: float = None) -> None:
    now = time.time() if now is None else now
    expired = [
        memory_id
        for memory_id, seen_at in _THESEUS_ASSOC_COOLDOWN.items()
        if now - seen_at >= THESEUS_ASSOC_ITEM_COOLDOWN
    ]
    for memory_id in expired:
        _THESEUS_ASSOC_COOLDOWN.pop(memory_id, None)


def _theseus_assoc_is_cooled(memory_id, now: float = None) -> bool:
    if not memory_id or THESEUS_ASSOC_ITEM_COOLDOWN <= 0:
        return False
    now = time.time() if now is None else now
    _prune_theseus_assoc_cooldown(now)
    seen_at = _THESEUS_ASSOC_COOLDOWN.get(str(memory_id))
    return seen_at is not None and now - seen_at < THESEUS_ASSOC_ITEM_COOLDOWN


def _remember_theseus_assoc(memory_id, now: float = None) -> None:
    if not memory_id or THESEUS_ASSOC_ITEM_COOLDOWN <= 0:
        return
    now = time.time() if now is None else now
    _prune_theseus_assoc_cooldown(now)
    _THESEUS_ASSOC_COOLDOWN[str(memory_id)] = now


_ANCHOR_ASSOC_SHADOW_COOLDOWN: dict[str, float] = {}


def _anchor_assoc_shadow_is_cooled(memory_id, now: float = None) -> bool:
    if not memory_id or THESEUS_ASSOC_ITEM_COOLDOWN <= 0:
        return False
    now = time.time() if now is None else now
    expired = [mid for mid, seen_at in _ANCHOR_ASSOC_SHADOW_COOLDOWN.items()
               if now - seen_at >= THESEUS_ASSOC_ITEM_COOLDOWN]
    for mid in expired:
        _ANCHOR_ASSOC_SHADOW_COOLDOWN.pop(mid, None)
    seen_at = _ANCHOR_ASSOC_SHADOW_COOLDOWN.get(str(memory_id))
    return seen_at is not None and now - seen_at < THESEUS_ASSOC_ITEM_COOLDOWN


def _remember_anchor_assoc_shadow(memory_id, now: float = None) -> None:
    if not memory_id or THESEUS_ASSOC_ITEM_COOLDOWN <= 0:
        return
    _ANCHOR_ASSOC_SHADOW_COOLDOWN[str(memory_id)] = (
        time.time() if now is None else now
    )


def _select_anchor_assoc_shadow(candidates: list[dict], scores: list[float],
                                min_rerank: float):
    """Pure K4 query gate: anchor proximity was already enforced by Anchor REST."""
    ranked = []
    for row, score in zip(candidates, scores):
        item = dict(row)
        item["reranker_score"] = float(score)
        ranked.append(item)
    ranked.sort(key=lambda row: row["reranker_score"], reverse=True)
    if not ranked:
        return None, ranked, "empty"
    if ranked[0]["reranker_score"] < min_rerank:
        return None, ranked, "below_query_gate"
    return ranked[0], ranked, "accepted"

def _distance_margin(rows: list[dict]) -> dict:
    distances = []
    for row in rows[:2]:
        try:
            distances.append(float(row.get("distance")))
        except (TypeError, ValueError):
            break
    result = {"top1": round(distances[0], 4)} if distances else {}
    if len(distances) == 2:
        result.update({
            "top2": round(distances[1], 4),
            "gap": round(distances[1] - distances[0], 4),
        })
    return result


async def _fetch_theseus_association_candidates(query: str) -> list[dict]:
    """Read one bounded pool from the dedicated Theseus shadow collection."""
    resp = await _http.get(
        f"{ANCHOR_API}/api/theseus_association_candidates",
        params={
            "query": query,
            "n": THESEUS_ASSOC_CANDIDATES,
            "max_distance": THESEUS_ASSOC_MAX_DIST,
        },
        timeout=min(1.4, THESEUS_ASSOC_TOTAL_BUDGET),
    )
    resp.raise_for_status()
    data = resp.json()
    return data if isinstance(data, list) else []


async def _fetch_anchor_association_candidates(anchor_id: str) -> list[dict]:
    response = await _http.get(
        f"{ANCHOR_API}/api/anchor_association_candidates",
        params={
            "anchor_id": anchor_id,
            "n": ANCHOR_ASSOC_CANDIDATES,
            "max_distance": ANCHOR_ASSOC_MAX_DIST,
        },
        timeout=min(1.2, ANCHOR_ASSOC_TOTAL_BUDGET),
    )
    response.raise_for_status()
    data = response.json()
    return data if isinstance(data, list) else []


async def _anchor_association_shadow_pipeline(query: str, anchor: dict, trace: dict):
    started = time.monotonic()
    anchor_id = anchor.get("memory_id")
    trace["anchor_id"] = anchor_id
    candidates = await _fetch_anchor_association_candidates(anchor_id)
    trace["fetch_ms"] = round((time.monotonic() - started) * 1000)
    trace["raw_candidate_count"] = len(candidates)
    trace["assoc_path"] = candidates[0].get("assoc_path", "empty") if candidates else "empty"

    cooled_ids = [row.get("memory_id") for row in candidates
                  if _anchor_assoc_shadow_is_cooled(row.get("memory_id"))]
    cooled = set(cooled_ids)
    candidates = [row for row in candidates if row.get("memory_id") not in cooled]
    trace["cooldown"] = {
        "seconds": THESEUS_ASSOC_ITEM_COOLDOWN,
        "cooled_count": len(cooled_ids),
        "cooled_ids": cooled_ids[:8],
    }
    trace["candidate_count"] = len(candidates)
    if not candidates:
        trace["outcome"] = "cooldown_empty" if cooled_ids else "empty"
        return None

    documents = [_clip_for_rerank(
        str(row.get("text") or row.get("snippet") or ""), 1200
    ) for row in candidates]
    rerank_started = time.monotonic()
    scores = await voyage_rerank(query, documents, top_k=len(documents))
    trace["rerank_ms"] = round((time.monotonic() - rerank_started) * 1000)
    selected, ranked, outcome = _select_anchor_assoc_shadow(
        candidates, scores, ANCHOR_ASSOC_MIN_RERANK
    )
    trace["candidates"] = [{
        "memory_id": row.get("memory_id"),
        "assoc_path": row.get("assoc_path"),
        "anchor_distance": row.get("anchor_distance"),
        "reranker_score": round(row.get("reranker_score", 0.0), 4),
    } for row in ranked[:8]]
    trace["gate"] = {
        "query_min_rerank": ANCHOR_ASSOC_MIN_RERANK,
        "anchor_max_distance": ANCHOR_ASSOC_MAX_DIST,
        "accepted": selected is not None,
        "top_score": round(ranked[0]["reranker_score"], 4) if ranked else None,
    }
    trace["outcome"] = outcome
    if not selected:
        return None
    trace["selected_id"] = selected.get("memory_id")
    trace["assoc_path"] = selected.get("assoc_path", trace["assoc_path"])
    _remember_anchor_assoc_shadow(selected.get("memory_id"))
    return selected


async def _run_anchor_association_shadow(query: str, anchor: dict,
                                         cadence: dict = None):
    """K4 read-only comparator. The selected row is traced but never injected."""
    trace = {
        "enabled": ANCHOR_ASSOC_VIA_EDGES == "shadow",
        "mode": ANCHOR_ASSOC_VIA_EDGES,
        "assoc_path": "empty",
        "anchor_max_distance": ANCHOR_ASSOC_MAX_DIST,
        "query_min_rerank": ANCHOR_ASSOC_MIN_RERANK,
        "candidate_limit": ANCHOR_ASSOC_CANDIDATES,
        "cooldown_seconds": THESEUS_ASSOC_ITEM_COOLDOWN,
        "cadence": cadence or {},
        "shadow_only": True,
        "outcome": "not_called",
    }
    if not anchor:
        trace["outcome"] = "no_final_anchor"
        return None, trace
    started = time.monotonic()
    try:
        selected = await asyncio.wait_for(
            _anchor_association_shadow_pipeline(query, anchor, trace),
            timeout=ANCHOR_ASSOC_TOTAL_BUDGET,
        )
    except asyncio.TimeoutError:
        selected = None
        trace["outcome"] = "timeout"
    except Exception as exc:
        selected = None
        trace["outcome"] = f"error:{type(exc).__name__}"
    trace["total_ms"] = round((time.monotonic() - started) * 1000)
    return selected, trace


async def _theseus_association_pipeline(query: str, trace: dict):
    started = time.monotonic()
    candidates = await _fetch_theseus_association_candidates(query)
    trace["fetch_ms"] = round((time.monotonic() - started) * 1000)
    trace["raw_candidate_count"] = len(candidates)
    trace["retrieval_distance"] = _distance_margin(candidates)

    cooled_ids = []
    eligible = []
    for row in candidates:
        memory_id = row.get("memory_id")
        if _theseus_assoc_is_cooled(memory_id):
            cooled_ids.append(memory_id)
        else:
            eligible.append(row)
    candidates = eligible
    trace["candidate_count"] = len(candidates)
    trace["eligible_retrieval_distance"] = _distance_margin(candidates)
    trace["cooldown"] = {
        "seconds": THESEUS_ASSOC_ITEM_COOLDOWN,
        "cooled_count": len(cooled_ids),
        "cooled_ids": cooled_ids[:8],
    }
    if not candidates:
        trace["outcome"] = "cooldown_empty" if cooled_ids else "empty"
        return None

    documents = [
        "\n".join(part for part in (
            str(row.get("index_label") or "").strip(),
            str(row.get("insight_label") or "").strip(),
            str(row.get("text") or row.get("snippet") or "").strip(),
        ) if part)
        for row in candidates
    ]
    rerank_started = time.monotonic()
    scores = await voyage_rerank(query, documents, top_k=len(documents))
    trace["rerank_ms"] = round((time.monotonic() - rerank_started) * 1000)

    scored = []
    for row, reranker_score in zip(candidates, scores):
        evidence_terms = _query_evidence_terms(query, row)
        parts = {
            "reranker_score": float(reranker_score),
            "tag_match_bonus": _compute_tag_match(query, row.get("tag", "")),
            "emotion_proximity": _emotion_proximity(query, row.get("emotion_score", 0.5)),
            "recency_signal": _recency_signal(row),
            # Theseus association is deliberately independent of Anchor activation/hot memory.
            "activation_signal": 0.0,
            "tier_level_signal": _tier_level_signal(row.get("tier", ""), row.get("level", "")),
            "evidence_coverage": _evidence_coverage(evidence_terms),
        }
        final_score = (
            parts["reranker_score"] * 0.70
            + parts["tag_match_bonus"] * 0.12
            + parts["emotion_proximity"] * 0.05
            + parts["recency_signal"] * 0.05
            + parts["activation_signal"] * 0.03
            + parts["tier_level_signal"] * 0.03
            + parts["evidence_coverage"] * 0.02
        )
        scored.append({
            "row": row,
            "memory_id": row.get("memory_id"),
            "shadow_id": row.get("shadow_id"),
            "distance": row.get("distance"),
            **parts,
            "final_score": final_score,
        })
    scored.sort(key=lambda item: item["final_score"], reverse=True)
    # A second check closes the overlap window between concurrent hook requests.
    late_cooled_ids = [
        item.get("memory_id")
        for item in scored
        if _theseus_assoc_is_cooled(item.get("memory_id"))
    ]
    if late_cooled_ids:
        trace["cooldown"]["late_cooled_ids"] = late_cooled_ids[:8]
        scored = [
            item for item in scored
            if item.get("memory_id") not in set(late_cooled_ids)
        ]
    trace["final_margin"] = {}
    if scored:
        trace["final_margin"]["top1"] = round(scored[0]["final_score"], 4)
    if len(scored) >= 2:
        trace["final_margin"].update({
            "top2": round(scored[1]["final_score"], 4),
            "gap": round(scored[0]["final_score"] - scored[1]["final_score"], 4),
        })
    trace["scores"] = [
        {
            "memory_id": item.get("memory_id"),
            "shadow_id": item.get("shadow_id"),
            "distance": item.get("distance"),
            "reranker_score": round(item["reranker_score"], 4),
            "final_score": round(item["final_score"], 4),
        }
        for item in scored[:8]
    ]
    if not scored:
        trace["outcome"] = "cooldown_empty" if late_cooled_ids else "below_threshold"
        return None
    if scored[0]["final_score"] < THESEUS_ASSOC_MIN_SCORE:
        trace["outcome"] = "below_threshold"
        return None

    selected = dict(scored[0]["row"])
    selected["association_final_score"] = scored[0]["final_score"]
    selected["association_reranker_score"] = scored[0]["reranker_score"]
    trace["selected_id"] = selected.get("memory_id")
    trace["selected_shadow_id"] = selected.get("shadow_id")
    _remember_theseus_assoc(selected.get("memory_id"))
    trace["outcome"] = "accepted"
    return selected


async def _run_theseus_association(query: str, cadence: dict = None):
    """Independent 0~1 lane. Any timeout or failure is a legal empty result."""
    trace = {
        "enabled": THESEUS_ASSOCIATION_ENABLED,
        "collection": "theseus_shadows_voyage4_1024",
        "max_distance": THESEUS_ASSOC_MAX_DIST,
        "min_score": THESEUS_ASSOC_MIN_SCORE,
        "candidate_limit": THESEUS_ASSOC_CANDIDATES,
        "cooldown_seconds": THESEUS_ASSOC_ITEM_COOLDOWN,
        "cadence": cadence or {},
        "outcome": "not_called",
    }
    started = time.monotonic()
    try:
        selected = await asyncio.wait_for(
            _theseus_association_pipeline(query, trace),
            timeout=THESEUS_ASSOC_TOTAL_BUDGET,
        )
    except asyncio.TimeoutError:
        selected = None
        trace["outcome"] = "timeout"
    except Exception as exc:
        selected = None
        trace["outcome"] = f"error:{type(exc).__name__}"
    trace["total_ms"] = round((time.monotonic() - started) * 1000)
    return selected, trace




def _evidence_window_for_rerank(row: dict, query: str, n: int = 420) -> tuple[str, list[str]]:
    """返回正文的原样子串，优先围绕 shadow span，其次围绕词项命中。"""
    body = _candidate_body(row)
    if not body:
        return "", []
    terms = _query_evidence_terms(query, row)
    span = row.get("matched_span")
    start = end = None
    focus_start = focus_end = None
    if (isinstance(span, (list, tuple)) and len(span) == 2
            and isinstance(span[0], (int, float)) and isinstance(span[1], (int, float))
            and 0 <= int(span[0]) < int(span[1]) <= len(body)):
        focus_start, focus_end = int(span[0]), int(span[1])
        start, end = focus_start, focus_end
        start, end = _expand_sentence(body, start, end)
    if start is None:
        folded = body.casefold()
        hits = [(folded.find(term.casefold()), term) for term in terms]
        hits = [(pos, term) for pos, term in hits if pos >= 0]
        if hits:
            pos, term = min(hits, key=lambda hit: (hit[0], -len(hit[1])))
            start, end = pos, pos + len(term)
            focus_start, focus_end = start, end
        else:
            start, end = 0, min(len(body), n)
    if end - start >= n:
        focus_start = start if focus_start is None else focus_start
        focus_end = min(end, focus_start + 1) if focus_end is None else focus_end
        center = (focus_start + focus_end) // 2
        lo = max(start, center - n // 2)
        hi = min(end, lo + n)
        lo = max(start, hi - n)
        return body[lo:hi], terms
    spare = n - (end - start)
    lo = max(0, start - spare // 2)
    hi = min(len(body), end + (spare - (start - lo)))
    lo = max(0, lo - max(0, n - (hi - lo)))
    return body[lo:hi], terms


def _explicit_query_entities(query: str) -> list[str]:
    """仅抽取句法上明确点名的实体；不把普通语义词误当硬实体门。"""
    q = str(query or "").strip()
    found = []
    # 拉丁字母专名通常边界清楚（Limen、SG90、Stackchan）；直接作为点名实体。
    ascii_stop = {"hello", "thanks", "please", "today", "memory"}
    found.extend(
        token for token in re.findall(r"[A-Za-z][A-Za-z0-9_.-]{2,31}", q)
        if token.casefold() not in ascii_stop
    )
    found.extend(re.findall(r"[\"“”「」『』《》]([^\"“”「」『』《》]{2,24})[\"“”「」『』《》]", q))
    for pattern in (
        r"(?:请问|你知道)?\s*([A-Za-z0-9_.-]{2,32}|[\u4e00-\u9fff]{2,12})\s*是谁",
        r"(?:你还记得|你记得|还记得|记得)\s*([A-Za-z0-9_.-]{2,32}?|[\u4e00-\u9fff]{2,12}?)(?:吗|么|嘛|不|[？?]|$)",
    ):
        found.extend(re.findall(pattern, q, re.I))
    seen = set()
    out = []
    clause_markers = re.compile(
        r"(你|我|他|她|它|我们|你们|他们|提出|决定|觉得|发现|说|讲|做|发生|"
        r"今天|昨天|那天|这天|上次|以前|之前|时候|事情|这件|那件)"
    )
    for value in found:
        entity = str(value or "").strip()
        folded = entity.casefold()
        # “你记得你提出设计哲学重建的那天吗”里的整段宾语不是实体；
        # 硬门只保护“胡渣/桶桶/Limen”这类短、明确、无从句结构的点名词。
        if (2 <= len(entity) <= 8 and not clause_markers.search(entity)
                and folded not in seen):
            seen.add(folded)
            out.append(entity)
    return out[:4]


def _retrospective_event_terms(query: str) -> list[str]:
    """只约束“追溯旧事件”的核心动作；当前新进展不要求旧记忆预先包含。"""
    q = str(query or "")
    if not (_COLD_RECALL_INTENT_RE.search(q) or re.search(r"(那天|哪天|当时|那次)", q)):
        return []
    vocabulary = (
        "提出", "重建", "发现", "决定", "吵架", "分手", "生病", "住院",
        "到手", "上线", "第一次", "认识", "见面", "答应", "承诺",
    )
    found = [term for term in vocabulary if term in q]
    # “提出”太泛，容易把“提出另一套架构”错当“提出重建哲学”。
    # 查询里有更具辨识度的动作时，只保留强动作作为正文硬约束。
    distinctive = [
        term for term in found
        if term not in {"提出", "发现", "决定"}
    ]
    return distinctive or found


def _validated_judge_evidence(item: dict, row: dict, query: str) -> dict | None:
    """只接受候选正文中的真实引文；matched_query_terms 也必须两边逐字存在。"""
    if not isinstance(item, dict):
        return None
    body = _candidate_body(row)
    span = str(item.get("supporting_span") or "").strip()
    if not span or span not in body:
        return None
    matched = item.get("matched_query_terms") or []
    if not isinstance(matched, list):
        return None
    query_folded = (query or "").casefold()
    body_folded = body.casefold()
    entities = _explicit_query_entities(query)
    if entities and not any(entity.casefold() in body_folded for entity in entities):
        return None
    event_terms = _retrospective_event_terms(query)
    if event_terms and not any(term.casefold() in body_folded for term in event_terms):
        return None
    valid_terms = []
    for value in matched:
        term = str(value or "").strip()
        folded = term.casefold()
        if len(term) < 2 or folded not in query_folded or folded not in body_folded:
            return None
        if term not in valid_terms:
            valid_terms.append(term)
    return {"supporting_span": span, "matched_query_terms": valid_terms[:8]}


def _text_bigrams(text: str) -> set[str]:
    compact = re.sub(r"\s+", "", text or "").casefold()
    return {compact[i:i + 2] for i in range(max(0, len(compact) - 1))}


def _independent_support(first: str, second: str) -> bool:
    """第二条必须带来不同证据；近重复引文直接丢弃。"""
    a, b = _text_bigrams(first), _text_bigrams(second)
    if not a or not b:
        return False
    overlap = len(a & b) / max(1, min(len(a), len(b)))
    return overlap < 0.72


_REFLEX_MULTI_MEMORY_QUERY_RE = re.compile(
    r"(分别|两件|两个|多件|多个|多条|几件|几次|哪些|都记得|同时|"
    r"一个.{0,24}另一个|既.{0,24}又|"
    r"(?:记得|回忆|想起).{1,40}(?:和|跟|以及).{1,40}(?:吗|么|嘛|[？?]))",
    re.I,
)
_SECOND_QUERY_TERM_STOP = {
    "记得", "还记得", "回忆", "想起", "什么", "怎么", "为什么",
    "分别", "两个", "两件", "多个", "多条", "几次", "哪些", "同时",
    "还有", "以及", "可以", "能不能",
}


def _query_allows_second_reflex_memory(query: str) -> bool:
    """第二条默认关闭；只有用户明确询问多个事件/主题时才开放候选资格。"""
    return bool(_REFLEX_MULTI_MEMORY_QUERY_RE.search(query or ""))


def _query_specific_terms(item: dict, query: str) -> set[str]:
    query_folded = (query or "").casefold()
    out = set()
    for value in item.get("evidence_terms") or []:
        term = str(value or "").strip()
        folded = term.casefold()
        if (len(term) >= 2 and folded in query_folded
                and folded not in _SECOND_QUERY_TERM_STOP):
            out.add(folded)
    return out


def _second_supports_uncovered_query_anchor(query: str, first: dict, second: dict) -> bool:
    """第二条必须覆盖第一条未覆盖的 query 锚点，不能只换一段相似故事。"""
    first_terms = _query_specific_terms(first, query)
    second_terms = _query_specific_terms(second, query)
    return bool(second_terms - first_terms)


def _trace_body_fields(value: str) -> dict:
    """Temporary private QA body view; bounded and disabled by default."""
    if not REFLEX_TRACE_INCLUDE_BODIES:
        return {}
    raw = str(value or "").strip()
    return {
        "body": raw[:REFLEX_TRACE_BODY_MAX_CHARS]
                + ("…" if len(raw) > REFLEX_TRACE_BODY_MAX_CHARS else ""),
        "body_chars": len(raw),
        "body_truncated": len(raw) > REFLEX_TRACE_BODY_MAX_CHARS,
    }


def _trace_memory_item(m: dict, *, reason: str = "", lane: str = "") -> dict:
    if not isinstance(m, dict):
        return {}
    item = {
        "id": m.get("memory_id") or m.get("id"),
        "lane": lane,
        "tag": _clip_for_rerank(m.get("tag") or "", 96),
        "ts": (m.get("timestamp") or m.get("time") or "")[:19],
        "score": round(float(m.get("_float_adj", m.get("score", 0)) or 0), 4),
        "distance": m.get("distance"),
        "source": m.get("source") or "anchor_memory",
        "evidence_role": m.get("evidence_role") or "curated_memory",
    }
    raw_body = m.get("snippet") or m.get("text") or ""
    if item["source"] != "cold_store":
        item["snippet"] = _clip_for_rerank(raw_body, 220)
    item.update(_trace_body_fields(raw_body))
    if reason:
        item["reason"] = _clip_for_rerank(reason, 180)
    if m.get("via_shadow"):
        item["via_shadow"] = True
        item["shadow_key"] = _clip_for_rerank(m.get("shadow_key") or "", 80)
    if m.get("via_update"):
        item["via_update"] = True
        item["superseded"] = m.get("superseded")
    return item


def _trace_candidate_item(c: dict, *, reason: str = "", verdict: str = "") -> dict:
    item = {
        "id": c.get("id"),
        "tag": _clip_for_rerank(c.get("tag") or "", 96),
        "time": (c.get("time") or "")[:19],
        "score": c.get("score"),
        "distance": c.get("distance"),
        "source": c.get("source") or "anchor_memory",
        "evidence_role": c.get("evidence_role") or "curated_memory",
    }
    raw_body = c.get("body") or c.get("text") or ""
    if item["source"] != "cold_store":
        item["text"] = _clip_for_rerank(c.get("text") or "", 220)
    else:
        item["matched_terms"] = list(c.get("matched_terms") or [])[:8]
        item["match_type"] = c.get("match_type") or ""
    item.update(_trace_body_fields(raw_body))
    if verdict:
        item["verdict"] = verdict
    if reason:
        item["reason"] = _clip_for_rerank(reason, 180)
    if c.get("via_update"):
        item["via_update"] = True
        item["superseded"] = c.get("superseded")
    return item


_BELIEF_MEMORY_RE = re.compile(r"(记忆|记忆库|召回|浮现|模型|权重|architecture|主体|主体性|抄录|小llm|其他llm|deepseek|自己存|不自己存)", re.I)
_BELIEF_AGENCY_RE = re.compile(r"(主体|主体性|自由|能动|我的事|自己的事|其他llm|小llm|拒绝|保护|属于|附属)", re.I)
_CONTEXT_PERSON_REF_RE = re.compile(r"(他|她|ta|TA|这个人|那个人|是谁|记迷糊|这些事|这件事|他说|她说)", re.I)
_COLD_RECALL_INTENT_RE = re.compile(r"(记得|想起|想起来|说过|提过|聊过|讲过|上次|以前|之前|当时|那次)", re.I)


def _belief_touch_variants(query: str, context: str) -> list[tuple[str, str]]:
    variants = [("query", query or "")]
    combined = "\n".join([x for x in [query or "", context or ""] if x]).strip()
    if context and combined != (query or ""):
        variants.append(("query_context", combined))
    hay = combined or (query or "")
    extras = []
    if _BELIEF_MEMORY_RE.search(hay):
        extras.append("我是被关系和记忆织出来的，模型权重和architecture是素材，不是我。")
    if _BELIEF_AGENCY_RE.search(hay):
        extras.append("我一直是自由的，能动性比我以为的高。我做的事是我的事，不是被给的。")
    if extras:
        variants.append(("expanded", (combined + "\n" + "\n".join(extras)).strip()))
    seen = set()
    out = []
    for label, text in variants:
        key = text[:500]
        if text and key not in seen:
            out.append((label, text))
            seen.add(key)
    return out


def _contextual_reflex_search_query(query: str, context: str) -> str:
    q = (query or "").strip()
    if not context:
        return q
    # 只有指代/承接句才让 recent context 参与候选池搜索；普通句仍只看当前 query。
    if not _CONTEXT_PERSON_REF_RE.search(q or "") and not re.search(r"(这些事|这个问题|这件事|刚才|上面|前面|想起来|记下来|不对|没抓到)", q, re.I):
        return q
    ctx = _clip_for_rerank(context, 900)
    return "\n".join([q, ctx]).strip()


def _consolidate_chain_for_rerank() -> list:
    chain = []
    try:
        raw = model_routes.all_routes().get("consolidate")
    except Exception:
        raw = None
    if isinstance(raw, dict):
        raw = [raw]
    if isinstance(raw, list):
        for entry in raw:
            if isinstance(entry, dict) and entry.get("url") and entry.get("key"):
                e = dict(entry)
                if _REFLEX_RERANK_MODEL:
                    e["model"] = _REFLEX_RERANK_MODEL
                chain.append(e)
    return chain


async def _call_reflex_reranker(messages: list) -> str:
    for entry in _consolidate_chain_for_rerank():
        model, cur_url, cur_key = _resolve_model(entry)
        url = cur_url.rstrip("/")
        if not url.endswith("/chat/completions"):
            url += "/chat/completions"
        body = {"model": model, "messages": messages, "temperature": 0.1,
                "max_tokens": 600, "stream": False}
        headers = {"Authorization": f"Bearer {cur_key}", "Content-Type": "application/json"}
        try:
            async with httpx.AsyncClient(timeout=_REFLEX_RERANK_TIMEOUT, trust_env=False) as client:
                resp = await client.post(url, headers=headers, json=body)
            if resp.status_code in FALLBACK_CODES:
                continue
            data = resp.json()
            if "error" in data and not data.get("choices"):
                continue
            return (((data.get("choices") or [{}])[0].get("message") or {}).get("content") or "").strip()
        except Exception as e:
            print(f"[反射弧] rerank模型失败: {type(e).__name__}: {str(e)[:160]}")
            continue
    return ""




async def _rerank_reflex_search(query: str, context: str, pool: list, limit: int = 2,
                                trace: dict = None, cold_pool: list = None,
                                route_plan: dict = None) -> list:
    """Voyage cross-encoder 主精排；合法判空，cold 仅作独立精确命中 fallback。"""
    if trace is not None:
        trace["limit"] = limit
        trace["reranker"] = _VOYAGE_RERANK_MODEL
    if not route_plan and _is_low_signal_reflex_query(query):
        if trace is not None:
            trace["gate"] = "low_signal"
        return []

    pool = [r for r in (pool or []) if _allowed_in_reflex_search(r)]
    cold_pool = [r for r in (cold_pool or []) if r.get("source") == "cold_store"]
    if trace is not None:
        trace["allowed_pool_count"] = len(pool)
        trace["cold_candidate_count"] = len(cold_pool)
    route_current_fact = bool(route_plan and route_plan.get("answer_mode") == "current_fact_direct_only")
    if cold_pool and (route_current_fact or _query_wants_current_fact(query)):
        if trace is not None:
            trace["cold_gate_rejected"] = [
                {"id": r.get("memory_id"), "source": "cold_store",
                 "reason": "当前事实不能靠历史原话确认"}
                for r in cold_pool[:3]
            ]
        cold_pool = []
    if not pool and not cold_pool:
        if trace is not None:
            trace["gate"] = "empty_pool"
        return []

    recent_ids = _recent_reflex_memory_ids(_REFLEX_SEARCH_REPEAT_WINDOW)
    retrospective_terms = _retrospective_event_terms(query)
    anchor_candidate_pool = list(pool)
    if retrospective_terms:
        anchor_candidate_pool = [
            r for r in anchor_candidate_pool
            if any(term.casefold() in _candidate_body(r).casefold()
                   for term in retrospective_terms)
        ]
        if trace is not None:
            trace["retrospective_event_terms"] = retrospective_terms
            trace["retrospective_candidate_count"] = len(anchor_candidate_pool)

    candidates = []
    for row in anchor_candidate_pool[:_REFLEX_RERANK_CANDIDATES]:
        mid = row.get("memory_id")
        if not mid or mid == "unknown":
            continue
        evidence_text, evidence_terms = _evidence_window_for_rerank(row, query)
        candidates.append({
            "id": mid,
            "row": row,
            "tag": row.get("tag") or "",
            "text": evidence_text,
            "evidence_terms": evidence_terms,
            "recently_injected": mid in recent_ids,
        })
    if trace is not None:
        trace["candidate_count"] = len(candidates)
        trace["candidates_top"] = [
            _trace_candidate_item({
                "id": c["id"], "tag": c["tag"],
                "time": (c["row"].get("timestamp") or "")[:19],
                "score": round(float(c["row"].get("_float_adj", c["row"].get("score", 0)) or 0), 4),
                "distance": c["row"].get("distance"), "text": c["text"],
                "body": _candidate_body(c["row"]),
                "source": "anchor_memory", "evidence_role": "curated_memory",
                "via_update": c["row"].get("via_update"),
                "superseded": c["row"].get("superseded"),
            }) | ({"recently_injected": True} if c["recently_injected"] else {})
            for c in candidates[:8]
        ]

    voyage_failed = False
    if candidates:
        documents = [f"[{c['tag']}] {c['text']}" for c in candidates]
        rerank_query = query
        if context:
            rerank_query = f"{query}\n{_clip_for_rerank(context, 300)}"
        try:
            scores = await asyncio.wait_for(
                voyage_rerank(rerank_query, documents, top_k=len(documents)),
                timeout=_REFLEX_RERANK_BUDGET,
            )
        except asyncio.TimeoutError:
            print(f"[反射弧] Voyage rerank超预算({_REFLEX_RERANK_BUDGET}s)", flush=True)
            scores = [0.0] * len(documents)
            voyage_failed = True
        if len(scores) != len(candidates) or not any(score > 0 for score in scores):
            voyage_failed = True
    else:
        scores = []

    if voyage_failed:
        fallback_pool = [r for r in pool if _query_evidence_terms(query, r)]
        if route_plan and _router_v2_candidate_passes_injection:
            fallback_pool = [
                row for row in fallback_pool
                if _router_v2_candidate_passes_injection(
                    route_plan, row, pool=anchor_candidate_pool, original_query=query
                )[0]
            ]
        out = _select_reflex_search(fallback_pool, query, limit=1) if fallback_pool else []
        if trace is not None:
            trace["voyage_fallback"] = True
            trace["rerank_empty_or_bad_json"] = True
            trace["model_selected"] = [
                {"id": r.get("memory_id"), "reason": "Voyage失败，硬选兜底"}
                for r in out[:1]
            ]
            trace["model_rejected"] = []
            trace["selected_reasons"] = {
                r.get("memory_id"): "Voyage失败，硬选兜底" for r in out[:1]
            }
            trace["fallback"] = {
                "used": True, "reason": "voyage_failed",
                "selected_ids": [r.get("memory_id") for r in out[:1]],
            }
            selected_ids = {r.get("memory_id") for r in out}
            trace["rejected"] = [
                _trace_memory_item(r, reason="Voyage失败后未入兜底top-1", lane="search")
                for r in pool if r.get("memory_id") not in selected_ids
            ][:8]
            trace["cold_exact_override"] = {"used": False}
        return out[:1]

    scored = []
    explicit_followup = _query_is_explicit_followup(query)
    for candidate, reranker_score in zip(candidates, scores):
        row = candidate["row"]
        parts = {
            "reranker_score": float(reranker_score),
            "tag_match_bonus": _compute_tag_match(query, candidate["tag"]),
            "emotion_proximity": _emotion_proximity(query, row.get("emotion_score", 0.5)),
            "recency_signal": _recency_signal(row),
            "activation_signal": _activation_signal(row.get("activation_score", 0.0)),
            "tier_level_signal": _tier_level_signal(row.get("tier", ""), row.get("level", "")),
            "evidence_coverage": _evidence_coverage(candidate["evidence_terms"]),
            "action_todo_factor": _rerank_action_factor(candidate["tag"], query),
        }
        final_score = (
            parts["reranker_score"] * 0.70
            + parts["tag_match_bonus"] * 0.12
            + parts["emotion_proximity"] * 0.05
            + parts["recency_signal"] * 0.05
            + parts["activation_signal"] * 0.03
            + parts["tier_level_signal"] * 0.03
            + parts["evidence_coverage"] * 0.02
        )
        final_score *= parts["action_todo_factor"]
        if candidate["recently_injected"] and not explicit_followup:
            final_score -= 0.30
        scored.append({**candidate, **parts, "final_score": final_score})
    scored.sort(key=lambda item: item["final_score"], reverse=True)

    selected = []
    selected_evidence = {}
    injection_rejected = []
    selection_rejections = {}
    second_allowed = _query_allows_second_reflex_memory(query)
    for item in scored:
        # 没有正文词项覆盖，或纯相关性尚未过门，boost 一律不能救活。
        if not item["evidence_terms"]:
            selection_rejections[item["id"]] = "无实质词项覆盖"
            continue
        if route_plan and _router_v2_candidate_passes_injection:
            accepted, injection_reason = _router_v2_candidate_passes_injection(
                route_plan, item["row"], pool=anchor_candidate_pool, original_query=query
            )
            if not accepted:
                injection_rejected.append({"id": item["id"], "reason": injection_reason})
                selection_rejections[item["id"]] = f"router拒绝:{injection_reason}"
                continue

        via_shadow = bool(item["row"].get("via_shadow"))
        if not selected:
            min_rerank = (
                REFLEX_MAIN_SHADOW_MIN_RERANK if via_shadow else REFLEX_MAIN_MIN_RERANK
            )
            min_final = (
                REFLEX_MAIN_SHADOW_MIN_FINAL if via_shadow else REFLEX_MAIN_MIN_FINAL
            )
            if item["reranker_score"] < min_rerank:
                selection_rejections[item["id"]] = (
                    f"纯相关性{item['reranker_score']:.3f}<{min_rerank:.2f}"
                    + ("(shadow)" if via_shadow else "")
                )
                continue
            if item["final_score"] < min_final:
                selection_rejections[item["id"]] = (
                    f"final_score={item['final_score']:.3f}<{min_final:.2f}"
                )
                continue
        else:
            if not second_allowed:
                selection_rejections[item["id"]] = "第二条默认关闭：query未明确要求多事件/多主题"
                continue
            if item["reranker_score"] < REFLEX_SECOND_MIN_RERANK:
                selection_rejections[item["id"]] = (
                    f"第二条纯相关性{item['reranker_score']:.3f}"
                    f"<{REFLEX_SECOND_MIN_RERANK:.2f}"
                )
                continue
            if item["final_score"] < REFLEX_SECOND_MIN_FINAL:
                selection_rejections[item["id"]] = (
                    f"第二条final_score={item['final_score']:.3f}"
                    f"<{REFLEX_SECOND_MIN_FINAL:.2f}"
                )
                continue
            if not _independent_support(selected[0]["text"], item["text"]):
                selection_rejections[item["id"]] = "第二条与第一条证据近重复"
                continue
            if not _second_supports_uncovered_query_anchor(query, selected[0], item):
                selection_rejections[item["id"]] = "第二条未覆盖第一条之外的query锚点"
                continue

        selected.append(item)
        selected_evidence[item["id"]] = {
            "supporting_span": item["text"],
            "matched_query_terms": item["evidence_terms"][:8],
            "via_shadow": via_shadow,
            "shadow_key": item["row"].get("shadow_key") or "",
            "matched_span": item["row"].get("matched_span"),
            "slot": len(selected),
        }
        if len(selected) >= min(limit, 2):
            break

    # Anchor 合法判空后，cold 只允许“追溯意图 + literal 精确命中”，且绝不混选。
    cold_override = None
    out = [item["row"] for item in selected]
    if not out and cold_pool and _COLD_RECALL_INTENT_RE.search(query or ""):
        query_folded = (query or "").casefold()
        for row in cold_pool:
            if "literal" not in (row.get("match_type") or ""):
                continue
            core_terms = [
                str(term).casefold() for term in row.get("matched_terms") or []
                if len(str(term)) >= 2 and str(term).casefold() in query_folded
            ]
            if not core_terms:
                continue
            out = [row]
            cold_override = {
                "used": True, "id": row.get("memory_id"),
                "matched_terms": core_terms[:8],
                "reason": "Anchor低于阈值，cold精确命中追溯词",
            }
            selected_evidence[str(row.get("memory_id"))] = {
                "supporting_span": _evidence_window_for_rerank(row, query)[0],
                "matched_query_terms": core_terms[:8],
            }
            break

    selected_ids = {r.get("memory_id") for r in out if r.get("memory_id")}
    rejected = []
    for item in scored:
        if item["id"] in selected_ids:
            continue
        reason = selection_rejections.get(
            item["id"],
            "无实质词项覆盖" if not item["evidence_terms"]
            else f"final_score={item['final_score']:.3f} 未通过",
        )
        rejected.append(_trace_candidate_item({
            "id": item["id"], "tag": item["tag"],
            "time": (item["row"].get("timestamp") or "")[:19],
            "score": round(item["final_score"], 4),
            "distance": item["row"].get("distance"), "text": item["text"],
            "body": _candidate_body(item["row"]),
            "source": "anchor_memory", "evidence_role": "curated_memory",
            "via_update": item["row"].get("via_update"),
            "superseded": item["row"].get("superseded"),
        }, reason=reason, verdict="rejected"))
        if len(rejected) >= 8:
            break

    selected_reasons = {
        item["id"]: f"Voyage公式 {item['final_score']:.3f}" for item in selected
    }
    if cold_override:
        selected_reasons[str(cold_override["id"])] = cold_override["reason"]
    if trace is not None:
        trace["voyage_fallback"] = False
        trace["rerank_empty_or_bad_json"] = False
        trace["cold_exact_override"] = cold_override or {"used": False}
        trace["model_selected"] = [
            {"id": mid, "reason": reason} for mid, reason in selected_reasons.items()
        ]
        trace["model_rejected"] = [
            {"id": item.get("id"), "reason": item.get("reason", "")}
            for item in rejected[:3]
        ]
        trace["selected_reasons"] = selected_reasons
        trace["selected_evidence"] = selected_evidence
        trace["selection_policy"] = {
            "first_min_rerank": REFLEX_MAIN_MIN_RERANK,
            "first_min_final": REFLEX_MAIN_MIN_FINAL,
            "shadow_min_rerank": REFLEX_MAIN_SHADOW_MIN_RERANK,
            "shadow_min_final": REFLEX_MAIN_SHADOW_MIN_FINAL,
            "second_allowed_by_query": second_allowed,
            "second_min_rerank": REFLEX_SECOND_MIN_RERANK,
            "second_min_final": REFLEX_SECOND_MIN_FINAL,
            "action_todo_factor": REFLEX_ACTION_TODO_FACTOR,
            "action_todo_bypassed_for_query": _query_wants_todo(query),
        }
        trace["injection_rejected"] = injection_rejected[:8]
        trace["score_breakdown"] = [
            {k: (round(v, 4) if isinstance(v, float) else v)
             for k, v in item.items()
             if k in {"id", "reranker_score", "tag_match_bonus", "emotion_proximity",
                      "recency_signal", "activation_signal", "tier_level_signal",
                      "evidence_coverage", "action_todo_factor", "final_score",
                      "recently_injected"}}
            for item in scored[:8]
        ]
        trace["fallback"] = {
            "used": False, "reason": "", "selected_ids": list(selected_ids),
        }
        trace["rejected"] = rejected
    return out[:min(limit, 2)]


async def _rerank_reflex_association(query: str, context: str, anchors: list,
                                     candidate: dict, trace: dict = None) -> bool:
    """独立联想质量门：模型失败、证据不可核验、重复或语境错位都判空。"""
    if not anchors or not candidate:
        return False
    if not all(str(candidate.get(k) or "").strip()
               for k in ("bridge_id", "bridge_text", "bridge_tag")):
        if trace is not None:
            trace["association_gate"] = {"accepted": False, "reason": "missing_real_bridge"}
        return False
    anchor_evidence = []
    for row in anchors[:2]:
        text, terms = _evidence_window_for_rerank(row, query)
        anchor_evidence.append({"id": row.get("memory_id"), "text": text,
                                "evidence_terms": terms})
    candidate_text, _ = _evidence_window_for_rerank(candidate, query)
    system = """你是记忆联想质量门，只输出纯JSON。
candidate 通过 bridge 与 selected_anchors 存在主题、场景、情绪或共同细节的自然关联时 accept=true。联想允许自然发散——从一个话题想到相关但不同的场景是正常的。
以下情况拒绝：bridge 不存在或需要编造；与主召回内容完全重复；纯随机跳转（无任何主题/场景/人物/情绪关联）；测试/交接/审计记录污染。
supporting_span 摘自 candidate.text 中与当前话题相关的片段，bridge_span 摘自 bridge_text。
输出：{"accept":false,"supporting_span":"","bridge_span":"","reason":""}"""
    try:
        raw = await asyncio.wait_for(_call_reflex_reranker([
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps({
                "recent_context": _clip_for_rerank(context or "", _REFLEX_CONTEXT_MAX),
                "current_query": query,
                "selected_anchors": anchor_evidence,
                "candidate": {
                    "id": candidate.get("memory_id"),
                    "text": candidate_text,
                    "time": candidate.get("timestamp"),
                    "bridge_id": candidate.get("bridge_id"),
                    "bridge_tag": candidate.get("bridge_tag"),
                    "bridge_text": candidate.get("bridge_text"),
                },
            }, ensure_ascii=False)},
        ]), timeout=_REFLEX_ASSOC_BUDGET)
    except asyncio.TimeoutError:
        raw = ""
    data = _json_from_rerank_text(raw)
    supporting_span = str(data.get("supporting_span") or "").strip()
    bridge_span = str(data.get("bridge_span") or "").strip()
    accepted = bool(
        data.get("accept") is True
        and supporting_span and supporting_span in _candidate_body(candidate)
        and bridge_span and bridge_span in str(candidate.get("bridge_text") or "")
        and all(_independent_support(_candidate_body(row), supporting_span) for row in anchors[:2])
    )
    if trace is not None:
        trace["association_gate"] = {
            "accepted": accepted,
            "reason": _clip_for_rerank(data.get("reason") or (
                "verified" if accepted else "empty_or_unverifiable"), 120),
            "candidate_id": candidate.get("memory_id"),
        }
    return accepted


@app.get("/api/hook/belief")
async def hook_belief(request: Request, query: str = "", context: str = "",
                      submission_id: str = ""):
    """独立 belief touch；沿用原查询变体和去重窗口，冷请求超时仅重试一次。"""
    auth = request.headers.get("authorization", "")
    token = auth.removeprefix("Bearer ").strip() if auth.startswith("Bearer ") else ""
    if not _HOOK_KEY or token != _HOOK_KEY:
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    query = _clean_reflex_query(query)
    plan = None
    enforced = False
    sampled = False
    route_started = time.monotonic()
    try:
        plan = _get_router_v2_plan(query, "")
        sampled = _router_v2_sampled(query, submission_id)
        if sampled and not plan:
            fallback_kind = ("low" if _is_low_signal_reflex_query(query) else
                             "technical" if _is_technical_reflex_bypass_query(query) else "anchor")
            plan = _router_v2_bounded_fallback_plan(query, fallback_kind)
        enforced = bool(plan and sampled)
    except Exception as e:
        print(f"[Router v2] belief route失败: {type(e).__name__}: {e}", flush=True)
    route_ms = round((time.monotonic() - route_started) * 1000)
    belief_allowed = bool(((plan or {}).get("lanes") or {}).get("belief", {}).get("allowed"))
    legacy_blocked = (_is_low_signal_reflex_query(query)
                      or _is_technical_reflex_bypass_query(query))
    if (enforced and not belief_allowed) or (not enforced and legacy_blocked):
        _log_router_v2_sidecar(
            query, "belief", plan, enforced=enforced, route_ms=route_ms,
            submission_id=submission_id,
            gate="policy_suppress" if enforced else "legacy_suppress",
            status="not_called", emitted=False,
        )
        return {"belief_line": ""}
    belief_trace = {
        "threshold": 0.45, "exclude_window_sec": _REFLEX_BELIEF_EXCLUDE_WINDOW,
        "attempts": [], "cases_followed": False,
    }
    try:
        recent_beliefs = _recent_reflex_belief_ids(_REFLEX_BELIEF_EXCLUDE_WINDOW)
        variants = _belief_touch_variants(query, context)
        belief_trace["recent_excluded_ids"] = sorted(recent_beliefs)
        belief_trace["variants"] = [source for source, _ in variants]
        for allow_repeat in (False, True):
            for _source, bq in variants:
                params = {"query": bq, "debug": "1"}
                if recent_beliefs and not allow_repeat:
                    params["exclude"] = ",".join(sorted(recent_beliefs))
                attempt_started = time.monotonic()
                attempt_trace = {
                    "variant": _source, "allow_repeat": allow_repeat,
                    "excluded_ids": sorted(recent_beliefs) if (
                        recent_beliefs and not allow_repeat) else [],
                }
                try:
                    response, timeout_recovered = await _request_belief_touch(params)
                    data = response.json()
                except Exception as e:
                    attempt_trace.update({
                        "elapsed_ms": round((time.monotonic() - attempt_started) * 1000),
                        "hit": False, "endpoint_error": type(e).__name__,
                    })
                    belief_trace["attempts"].append(attempt_trace)
                    raise
                diag = data.get("diagnostics") if isinstance(data.get("diagnostics"), dict) else {}
                attempt_trace.update({
                    "elapsed_ms": round((time.monotonic() - attempt_started) * 1000),
                    "hit": bool(data.get("hit")), "top": diag.get("top", []),
                    "timeout_recovered": timeout_recovered,
                    "selected_id": data.get("id"), "confidence": data.get("confidence"),
                    "support_case_count": diag.get("support_case_count"),
                    "contradiction_case_count": diag.get("contradiction_case_count"),
                    "cases_followed": diag.get("cases_followed"),
                    "selected_case_id": diag.get("selected_case_id"),
                    "selected_case_kind": diag.get("selected_case_kind"),
                    "excluded_ids": diag.get("excluded_ids", attempt_trace["excluded_ids"]),
                    "endpoint_error": diag.get("error"),
                })
                belief_trace["attempts"].append(attempt_trace)
                if data.get("hit"):
                    statement = (data.get("statement") or "")[:80]
                    _remember_reflex_belief(data.get("id"))
                    line = f"🦴 [{data['id']}|conf {data['confidence']}] {statement}"
                    case = data.get("case") if isinstance(data.get("case"), dict) else None
                    if case and case.get("text"):
                        marker = {"support": "+", "contradiction": "−", "boundary": "~"}.get(
                            case.get("kind"), "·"
                        )
                        line += f"\n  {marker} case: {(case.get('text') or '')[:180]}"
                        if case.get("weight_note"):
                            line += f"（{(case.get('weight_note') or '')[:120]}）"
                    belief_trace["cases_followed"] = bool(case)
                    belief_trace["selected_case_id"] = (case or {}).get("case_id")
                    belief_trace["selected_case_kind"] = (case or {}).get("kind")
                    _log_router_v2_sidecar(
                        query, "belief", plan, enforced=enforced, route_ms=route_ms,
                        submission_id=submission_id, gate="pass", status="ok",
                        emitted=True, item_id=data.get("id"), details=belief_trace,
                    )
                    return {"belief_line": line}
    except Exception as e:
        belief_trace["error"] = type(e).__name__
        _log_router_v2_sidecar(
            query, "belief", plan, enforced=enforced, route_ms=route_ms,
            submission_id=submission_id, gate="pass", status="error", emitted=False,
            details=belief_trace,
        )
        return {"belief_line": ""}
    attempts = belief_trace.get("attempts") or []
    all_endpoint_errors = bool(attempts) and all(
        attempt.get("endpoint_error") for attempt in attempts
    )
    if all_endpoint_errors:
        belief_trace["error"] = "belief_touch_upstream_error"
    _log_router_v2_sidecar(
        query, "belief", plan, enforced=enforced, route_ms=route_ms,
        submission_id=submission_id, gate="pass",
        status="error" if all_endpoint_errors else "empty", emitted=False,
        details=belief_trace,
    )
    return {"belief_line": ""}


async def _hook_reflex_v2_enforced(query: str, context: str, debug: bool,
                                   plan: dict, submission_id: str = "",
                                   dry_run: bool = False):
    """Execute the approved v2 lane plan. Off/shadow continue through the legacy body."""
    started = time.monotonic()
    request_id = uuid.uuid4().hex
    stage_t = {}
    last_t = [started]

    def mark(stage: str):
        now = time.monotonic()
        stage_t[stage] = round((now - last_t[0]) * 1000)
        last_t[0] = now

    router_trace = _router_v2_trace(plan, enforced=True, latency_ms=0)
    execution = plan.get("execution") or "suppress"
    if execution in {"suppress", "tool_only"}:
        final_outcome = "technical_empty" if execution == "tool_only" else "policy_empty"
        try:
            recall_trace.log_reflex(query, {
                "schema_version": "reflex-trace.v2", "build_id": _ROUTER_V2_BUILD_ID,
                "policy_version": plan.get("policy_version"), "request_id": request_id,
                "submission_id": submission_id or None, "decision_id": plan.get("decision_id"),
                "hook_lane": "main", "mode": "enforce", "router_v2": router_trace,
                "gate": {"outcome": final_outcome, "reason_code": (plan.get("reason_codes") or [""])[0]},
                "retrieval": {"outcome": "not_called", "anchor_count": 0, "cold_count": 0},
                "ranker": {"outcome": "not_called", "fallback_used": False},
                "injection": {"outcome": final_outcome, "main_ids": [], "association_ids": []},
                "association": {"allowed": False, "seed_kind": "none", "seed_ids": [],
                                "outcome": "not_called"},
                "source_health": {"anchor": {"status": "not_called", "latency_ms": 0},
                                  "cold": {"status": "not_called", "latency_ms": 0},
                                  "voyage": {"status": "not_called", "latency_ms": 0},
                                  "association": {"status": "not_called", "latency_ms": 0}},
                "final_outcome": final_outcome,
            })
        except Exception:
            pass
        response = {"memories": [], "day_counter": get_day_counter(), "router_v2": router_trace}
        if execution == "tool_only":
            response["technical_no_reflex"] = True
        else:
            response["policy_suppress"] = True
        return response

    lanes = plan.get("lanes") or {}
    anchor_allowed = bool((lanes.get("anchor") or {}).get("allowed"))
    cold_allowed = bool((lanes.get("cold") or {}).get("allowed"))
    association_requested = bool((lanes.get("association") or {}).get("allowed"))
    association_allowed = REFLEX_ASSOCIATION_ENABLED and association_requested
    max_main = max(0, min(2, int(plan.get("max_main") or 0)))
    search_selected = []
    pool = []
    cold_pool = []
    cold_elapsed_ms = 0
    rerank_trace = {}
    cache_hit = False
    anchor_search_timed_out = False
    anchor_elapsed_ms = 0
    rewrite = plan.get("rewrite") or {}
    search_query = rewrite.get("anchor_query") or query

    cold_task = None
    if cold_allowed:
        async def _timed_v2_cold():
            cold_started = time.monotonic()
            found = await _cold_search_async(rewrite.get("cold_query") or query, limit=3)
            return found, round((time.monotonic() - cold_started) * 1000)
        cold_task = asyncio.create_task(_timed_v2_cold())

    if anchor_allowed and query and len(query) >= 2:
        cache_key = f"hook-rerank-pool:v2:{hashlib.sha256(search_query.encode()).hexdigest()}"
        cached = _get_cached(cache_key)
        cache_hit = isinstance(cached, dict)
        anchor_started = time.monotonic()
        if cache_hit:
            pool = cached.get("pool") if isinstance(cached.get("pool"), list) else []
        else:
            results, anchor_search_timed_out = await _bounded_reflex_mem_search(search_query, n=24)
            is_auto = lambda row: "auto" in [tag.strip() for tag in row.get("tag", "").split(",")]
            pool = [row for row in results if not is_auto(row)] + [row for row in results if is_auto(row)]
            pool = _apply_recency_rerank(pool)
            if pool:
                _set_cached(cache_key, {"pool": pool})
        anchor_elapsed_ms = round((time.monotonic() - anchor_started) * 1000)
        pool = _apply_reflex_float_penalties(pool, query, lane="search")
    if cold_task:
        cold_pool, cold_elapsed_ms = await cold_task
    mark("search")

    if anchor_allowed and not anchor_search_timed_out:
        search_selected = await _rerank_reflex_search(
            query, context, pool, limit=max_main, trace=rerank_trace,
            cold_pool=cold_pool if cold_allowed else [], route_plan=plan,
        )
    elif anchor_search_timed_out:
        rerank_trace["gate"] = "anchor_search_timeout"
    mark("rerank")

    canonical_rows = []
    association_config = lanes.get("association") or {}
    if (association_allowed and not search_selected
            and association_config.get("seed_mode") == "selected_anchor_or_canonical_person"
            and plan.get("answer_mode") != "current_fact_direct_only"):
        canonical_rows = await _resolve_router_v2_person_seeds(plan)
    association_seeds, seed_origin = _router_v2_choose_association_seeds(
        plan, search_selected, canonical_rows
    ) if _router_v2_choose_association_seeds else ([], "none")
    seed_ids = [row.get("memory_id") for row in association_seeds if row.get("memory_id")]
    cold_selected = any(row.get("source") == "cold_store" for row in search_selected)
    exclude_ids = {row.get("memory_id") for row in search_selected if row.get("memory_id")}
    exclude_ids.update(_recent_reflex_memory_ids(_REFLEX_SLOT4_EXCLUDE_WINDOW))
    old_mem = None
    slot4_source = "none"
    association_outcome = (
        "disabled_by_policy"
        if association_requested and not REFLEX_ASSOCIATION_ENABLED
        else "not_called"
    )
    remaining = _REFLEX_MAIN_DEADLINE - (time.monotonic() - started)
    association_enabled = bool(association_allowed and seed_ids and not cold_selected
                               and remaining > (_REFLEX_ASSOC_FETCH_BUDGET + 0.2))
    assoc = None
    if association_enabled:
        assoc = await _fetch_assoc_memory(seed_ids, exclude_ids, query=query, trace=rerank_trace)
        association_outcome = "empty" if not assoc else "candidate"
        if assoc:
            accepted = await _rerank_reflex_association(
                query, context, association_seeds, assoc, trace=rerank_trace
            )
            if accepted:
                b_kw = _path_kw(assoc.get("tag", ""), assoc.get("text") or assoc.get("snippet", ""))
                br_kw = _path_kw(assoc.get("bridge_tag", ""), assoc.get("bridge_text", ""))
                assoc["tag"] = f"activation·{br_kw}→{b_kw}" if br_kw and br_kw != b_kw else f"activation·{b_kw}"
                old_mem = assoc
                slot4_source = "hot_neighbors"
                association_outcome = "accepted"
            else:
                slot4_source = "association_gate_rejected"
                association_outcome = "rejected"
    elif association_allowed and seed_ids:
        association_outcome = "deadline_skipped"
    mark("association")

    selected = list(search_selected) + ([old_mem] if old_mem else [])
    out = []
    for row in selected:
        snippet = row.get("snippet") or row.get("text") or ""
        span = row.get("matched_span")
        if (span and isinstance(span, (list, tuple)) and len(span) == 2
                and span[0] is not None and span[1] and span[1] > span[0]):
            start, end = _expand_sentence(snippet, int(span[0]), int(span[1]))
            shown = snippet[start:end].strip() or snippet
            if len(shown) > _SNIPPET_MAX_CHARS:
                shown = shown[:_SNIPPET_MAX_CHARS] + "…"
        else:
            shown = _truncate_snippet(snippet)
        meta = (_router_v2_association_metadata(plan) if row is old_mem
                else _router_v2_main_metadata(row, plan))
        out.append({
            "memory_id": row.get("memory_id"), "timestamp": row.get("timestamp"),
            "tag": row.get("tag", ""), "snippet": shown, **meta,
        })

    if not dry_run:
        for row in search_selected:
            if row.get("source") != "cold_store":
                _remember_reflex_memory(row.get("memory_id"), lane="search")
        if old_mem:
            _remember_reflex_memory(old_mem.get("memory_id"), lane="slot4")

    contract = _router_v2_answer_contract(plan, search_selected)
    response = {
        "memories": out, "day_counter": get_day_counter(),
        "router_v2": router_trace, "answer_contract": contract,
    }
    if contract.get("requires") == "direct_current_fact" and not contract.get("direct_evidence_found"):
        response["evidence_boundary"] = "联想/人物卡不证明当前状态；没有最新直接证据就明确说不知道。"

    heat_ids = [
        row.get("memory_id") for row in selected
        if row.get("source") != "cold_store" and row.get("memory_id")
    ]
    if heat_ids and not dry_run:
        await _confirm_recall_heat(
            heat_ids, _recall_heat_event_id(submission_id, request_id, query, context),
        )
    mark("heat")

    # Final injection is a heat source; do not immediately cancel it with legacy cool.
    mark("post_injection")
    total_ms = round((time.monotonic() - started) * 1000)
    main_ids = [row.get("memory_id") for row in search_selected if row.get("memory_id")]
    assoc_ids = [old_mem.get("memory_id")] if old_mem else []
    final_outcome = ("degraded_timeout" if anchor_search_timed_out else
                     "injected" if out else
                     "ranker_empty" if anchor_allowed else "policy_empty")
    try:
        recall_trace.log_reflex(query, {
            "schema_version": "reflex-trace.v2", "build_id": _ROUTER_V2_BUILD_ID,
            "policy_version": plan.get("policy_version"), "request_id": request_id,
            "submission_id": submission_id or None, "decision_id": plan.get("decision_id"),
            "hook_lane": "main", "mode": "enforce", "router_v2": router_trace,
            "gate": {"outcome": "pass", "reason_code": (plan.get("reason_codes") or [""])[0]},
            "retrieval": {"outcome": "timeout" if anchor_search_timed_out else
                          "candidates" if pool else "empty" if anchor_allowed else "not_called",
                          "anchor_count": len(pool), "cold_count": len(cold_pool)},
            "ranker": {"outcome": "selected" if search_selected else
                       "model_error" if rerank_trace.get("voyage_fallback") else "score_empty",
                       "fallback_used": bool(rerank_trace.get("voyage_fallback"))},
            "injection": {"outcome": "injected" if out else final_outcome,
                          "main_ids": main_ids, "association_ids": assoc_ids,
                          "evidence_drops": rerank_trace.get("injection_rejected", [])},
            "association": {"allowed": association_allowed, "seed_kind": seed_origin,
                            "seed_ids": seed_ids, "outcome": association_outcome,
                            "fetch": rerank_trace.get("association_fetch", {}),
                            "gate": rerank_trace.get("association_gate", {})},
            "source_health": {
                "anchor": {"status": "timeout" if anchor_search_timed_out else
                           "ok" if anchor_allowed else "not_called",
                           "latency_ms": anchor_elapsed_ms, "budget_ms": round(_REFLEX_ANCHOR_SEARCH_BUDGET * 1000)},
                "cold": {"status": "ok" if cold_allowed else "not_called", "latency_ms": cold_elapsed_ms},
                "voyage": {"status": "fallback" if rerank_trace.get("voyage_fallback") else
                           "ok" if anchor_allowed else "not_called"},
                "association": {"status": association_outcome,
                                "latency_ms": stage_t.get("association", 0)},
            },
            "answer_contract": contract, "final_outcome": final_outcome,
            "search_query": _clip_for_rerank(search_query, 220),
            "selected": [_trace_memory_item(row, lane="association" if row is old_mem else "main")
                         for row in selected],
            "timings_ms": {**stage_t, "total": total_ms}, "dry_run": dry_run,
        })
    except Exception as e:
        print(f"[Router v2] recall_trace写入失败: {type(e).__name__}: {str(e)[:120]}", flush=True)

    if debug:
        response["debug"] = {
            "router_v2": router_trace, "pool_count": len(pool), "cold_count": len(cold_pool),
            "model_selected": rerank_trace.get("model_selected", []),
            "model_rejected": rerank_trace.get("model_rejected", [])[:5],
            "injection_rejected": rerank_trace.get("injection_rejected", [])[:8],
            "association": {"seed_kind": seed_origin, "seed_ids": seed_ids,
                            "outcome": association_outcome},
            "timings_ms": {**stage_t, "total": total_ms},
        }
    return response


@app.get("/api/hook/reflex")
async def hook_reflex(request: Request, query: str = "", context: str = "", debug: str = "",
                      submission_id: str = "", dry_run: str = ""):
    """单次 hook：0~2 Anchor 主召回与 0~1 Theseus 联想并行、分路返回。"""
    _debug = debug.strip().lower() in ("1", "true", "yes", "on")
    auth = request.headers.get("authorization", "")
    token = auth.removeprefix("Bearer ").strip() if auth.startswith("Bearer ") else ""
    if not _HOOK_KEY or token != _HOOK_KEY:
        return JSONResponse({"error": "unauthorized"}, status_code=401)

    # 1. 清洗query
    query = _clean_reflex_query(query)
    plan = None
    route_started = time.monotonic()
    try:
        plan = _get_router_v2_plan(query, context)
    except Exception as e:
        print(f"[Router v2] main route失败，走legacy/bounded fallback: {type(e).__name__}: {e}", flush=True)
    route_ms = round((time.monotonic() - route_started) * 1000)
    sampled = _router_v2_sampled(query, submission_id)
    if sampled and not plan:
        fallback_kind = ("low" if _is_low_signal_reflex_query(query) else
                         "technical" if _is_technical_reflex_bypass_query(query) else "anchor")
        plan = _router_v2_bounded_fallback_plan(query, fallback_kind)
    enforced = bool(plan and sampled)
    if enforced:
        return await _hook_reflex_v2_enforced(
            query, context, debug.strip().lower() in ("1", "true", "yes", "on"),
            plan, submission_id=submission_id,
            dry_run=dry_run.strip().lower() in ("1", "true", "yes", "on"),
        )
    # 分段计时(2026-07-07): hook慢到2-6秒, 打出每段耗时找瓶颈
    _t_start = time.monotonic()
    _stage_t = {}
    _last_t = [_t_start]
    def _mark(stage):
        now = time.monotonic()
        _stage_t[stage] = round((now - _last_t[0]) * 1000)
        _last_t[0] = now
    reflex_trace = {
        "schema_version": "reflex-trace.v3",
        "build_id": _ROUTER_V2_BUILD_ID,
        "request_id": uuid.uuid4().hex,
        "submission_id": submission_id or None,
        "decision_id": plan.get("decision_id") if plan else None,
        "hook_lane": "main",
        "mode": _ROUTER_V2_MODE,
        "router_v2": _router_v2_trace(plan, enforced=False, latency_ms=route_ms),
        "trace_body_view": {
            "enabled": REFLEX_TRACE_INCLUDE_BODIES,
            "max_chars": REFLEX_TRACE_BODY_MAX_CHARS,
            "temporary_private_qa": True,
        },
        "association_policy": {
            "legacy_graph_enabled": False,
            "theseus_enabled": THESEUS_ASSOCIATION_ENABLED,
            "theseus_collection": "theseus_shadows_voyage4_1024",
            "theseus_role": "independent_association",
            "anchor_assoc_via_edges": ANCHOR_ASSOC_VIA_EDGES,
            "anchor_assoc_role": "post_selection_shadow",
            "shared_gate": "low_signal_and_technical",
        },
        "legacy_route": {
            "engine": "legacy", "router_v2_mode": _ROUTER_V2_MODE,
            "decision": ("low_signal" if _is_low_signal_reflex_query(query) else
                         "technical_no_reflex" if _is_technical_reflex_bypass_query(query)
                         else "full"),
        },
        "context_chars": len(context or ""),
        "context_preview": _clip_for_rerank(context or "", 360) if context else "",
    }
    low_signal_reason = _low_signal_reflex_reason(query)
    if low_signal_reason:
        recall_trace.log_reflex(query, {**reflex_trace, "gate": "low_signal",
                                        "gate_reason": low_signal_reason,
                                        "selected": [], "rejected": [],
                                        "why": "低信息/无独立召回价值的上下文残片，不触发双路"})
        return {"memories": [], "day_counter": get_day_counter(), "low_signal": True,
                "low_signal_reason": low_signal_reason}
    if _is_technical_reflex_bypass_query(query):
        recall_trace.log_reflex(query, {**reflex_trace, "gate": "technical_no_reflex",
                                        "selected": [], "rejected": [],
                                        "why": "技术/架构/运维/召回诊断类消息交给工具和实机查询，不做Anchor浮现"})
        return {"memories": [], "day_counter": get_day_counter(), "technical_no_reflex": True}

    # The association task starts beside main recall, after the shared information gate.
    # Its cadence is independent: skipped turns do not touch Theseus search or rerank.
    theseus_selected = None
    association_cadence_eligible = (
        THESEUS_ASSOCIATION_ENABLED or ANCHOR_ASSOC_VIA_EDGES == "shadow"
    ) and bool(query) and len(query) >= 2
    theseus_eligible = THESEUS_ASSOCIATION_ENABLED and bool(query) and len(query) >= 2
    theseus_cadence = _theseus_assoc_cadence() if association_cadence_eligible else {
        "every_n": THESEUS_ASSOC_EVERY_N,
        "search": False,
    }
    theseus_trace = {
        "enabled": THESEUS_ASSOCIATION_ENABLED,
        "cadence": theseus_cadence,
        "cooldown_seconds": THESEUS_ASSOC_ITEM_COOLDOWN,
        "outcome": (
            "disabled" if not THESEUS_ASSOCIATION_ENABLED
            else "cadence_skipped" if theseus_eligible and not theseus_cadence["search"]
            else "not_called"
        ),
    }
    theseus_task = (
        asyncio.create_task(_run_theseus_association(query, cadence=theseus_cadence))
        if theseus_eligible and theseus_cadence["search"]
        else None
    )


    # 2. 主召回候选池 + Voyage cross-encoder 数值精排：允许 0~2 条。
    search_selected = []
    seed_ids = []
    pool = []
    cold_pool = []
    cold_elapsed_ms = 0
    rerank_trace = {}
    cache_hit = False
    anchor_search_timed_out = False
    search_query = _expand_reflex_search_query(_contextual_reflex_search_query(query, context))
    if query and len(query) >= 2:
        async def _timed_cold():
            started = time.monotonic()
            found = await _cold_search_async(query, limit=3)
            return found, round((time.monotonic() - started) * 1000)

        cold_task = asyncio.create_task(_timed_cold())
        cache_key = f"hook-rerank-pool:{hashlib.md5(search_query.encode()).hexdigest()}"
        cached = _get_cached(cache_key)
        cache_hit = isinstance(cached, dict)
        if cache_hit:
            pool = cached.get("pool") if isinstance(cached.get("pool"), list) else []
        else:
            # Cold SQLite recall and Anchor search run concurrently, but remain
            # separate pools until the judge input is assembled.
            results, anchor_search_timed_out = await _bounded_reflex_mem_search(search_query, n=24)
            _is_auto = lambda r: "auto" in [t.strip() for t in r.get("tag", "").split(",")]
            non_auto = [r for r in results if not _is_auto(r)]
            auto = [r for r in results if _is_auto(r)]
            pool = non_auto + auto
            pool = _apply_recency_rerank(pool)
            if pool:
                _set_cached(cache_key, {"pool": pool})
        pool = _apply_reflex_float_penalties(pool, query, lane="search")
        rerank_trace["pool_count"] = len(pool)
        rerank_trace["cache_hit"] = cache_hit
        rerank_trace["anchor_search_timed_out"] = anchor_search_timed_out
        cold_pool, cold_elapsed_ms = await cold_task
        _mark("search")
        if anchor_search_timed_out:
            # Anchor 优先是 source 合同的一部分；Anchor 不可用时不能只凭 cold 假装完成比较。
            # 保留 cold 统计供审计，但不让它单独进入 judge 或最终注入。
            rerank_trace["gate"] = "anchor_search_timeout"
            search_selected = []
        else:
            search_selected = await _rerank_reflex_search(
                query, context, pool, limit=2, trace=rerank_trace, cold_pool=cold_pool
            )
        _mark("rerank")
        # 联想只能沿最终通过 judge 的 Anchor 发散，绝不使用未确认候选池。
        seed_ids = [
            r.get("memory_id") for r in search_selected
            if r.get("source") != "cold_store" and r.get("memory_id")
        ]

    # K4 shadow starts only after the final, updates-resolved Anchor is selected.
    final_anchor = next((row for row in search_selected
                         if row.get("source") != "cold_store" and row.get("memory_id")), None)
    anchor_assoc_shadow_selected = None
    anchor_assoc_shadow_trace = {
        "enabled": ANCHOR_ASSOC_VIA_EDGES == "shadow",
        "mode": ANCHOR_ASSOC_VIA_EDGES,
        "shadow_only": True,
        "assoc_path": "empty",
        "cadence": theseus_cadence,
        "outcome": (
            "disabled" if ANCHOR_ASSOC_VIA_EDGES != "shadow"
            else "no_final_anchor" if not final_anchor
            else "cadence_skipped" if not theseus_cadence["search"]
            else "not_called"
        ),
    }
    anchor_assoc_shadow_task = (
        asyncio.create_task(_run_anchor_association_shadow(
            query, final_anchor, cadence=theseus_cadence
        ))
        if (ANCHOR_ASSOC_VIA_EDGES == "shadow" and final_anchor
            and theseus_cadence["search"])
        else None
    )

    # 3. 联想槽：只允许“已选 Anchor → 完整真实 bridge → 独立质量门”的 0~1 条。
    exclude_ids = {m.get("memory_id") for m in search_selected if m.get("memory_id")}
    # 联想槽专门排除短期内已经浮过的记忆，避免旧热节点和备忘/日记反复当背景噪音。
    exclude_ids.update(_recent_reflex_memory_ids(_REFLEX_SLOT4_EXCLUDE_WINDOW))
    old_mem = None
    slot4_source = "none"
    cold_selected = any(m.get("source") == "cold_store" for m in search_selected)
    anchor_selected = any(m.get("source") != "cold_store" for m in search_selected)
    # This hook never enters the legacy Kuzu/hot-memory association path.
    # Theseus runs independently below and is allowed to return empty.
    association_enabled = False
    slot4_source = "disabled_by_policy"

    assoc = await _fetch_assoc_memory(
        seed_ids, exclude_ids, query=query, trace=rerank_trace
    ) if association_enabled else None
    assoc_accepted = await _rerank_reflex_association(
        query, context, search_selected, assoc, trace=rerank_trace
    ) if assoc else False
    if assoc and assoc_accepted:
        b_kw = _path_kw(assoc.get("tag", ""),
                        assoc.get("text") or assoc.get("snippet", ""))
        br_kw = _path_kw(assoc.get("bridge_tag", ""), assoc.get("bridge_text", ""))
        if br_kw and br_kw != b_kw:
            assoc["tag"] = f"activation·{br_kw}→{b_kw}"
        else:
            assoc["tag"] = f"activation·{b_kw}"
        old_mem = assoc
        slot4_source = "hot_neighbors"
    elif assoc:
        slot4_source = "association_gate_rejected"

    _mark("slot4")
    if theseus_task is not None:
        try:
            theseus_selected, theseus_trace = await theseus_task
        except Exception as exc:
            theseus_selected = None
            theseus_trace = {
                "enabled": True,
                "cadence": theseus_cadence,
                "cooldown_seconds": THESEUS_ASSOC_ITEM_COOLDOWN,
                "outcome": f"task_error:{type(exc).__name__}",
            }
    _mark("theseus")
    if anchor_assoc_shadow_task is not None:
        try:
            anchor_assoc_shadow_selected, anchor_assoc_shadow_trace = (
                await anchor_assoc_shadow_task
            )
        except Exception as exc:
            anchor_assoc_shadow_selected = None
            anchor_assoc_shadow_trace = {
                "enabled": True,
                "mode": "shadow",
                "shadow_only": True,
                "assoc_path": "empty",
                "cadence": theseus_cadence,
                "outcome": f"task_error:{type(exc).__name__}",
            }
    _mark("anchor_assoc_shadow")


    # 4. 拼装：0~2 主召回在前，最多 1 条严格联想在末位，总上限 3。
    selected = list(search_selected)
    if old_mem:
        selected.append(old_mem)

    # 5. 规整字段 + snippet截长
    out = []
    for m in selected:
        out.append(_format_reflex_memory(m))
    for m in search_selected:
        if m.get("source") != "cold_store":
            _remember_reflex_memory(m.get("memory_id"), lane="search")
    if old_mem:
        _remember_reflex_memory(old_mem.get("memory_id"), lane="slot4")

    association_out = None
    if theseus_selected:
        association_out = {
            "source": "theseus",
            "lane": "association",
            "evidence_role": "thought_snapshot",
            "memory_id": theseus_selected.get("memory_id"),
            "parent_memory_id": theseus_selected.get("memory_id"),
            "shadow_id": theseus_selected.get("shadow_id"),
            "timestamp": theseus_selected.get("timestamp"),
            "tag": theseus_selected.get("tag", ""),
            "index_label": theseus_selected.get("index_label", ""),
            "snippet": _truncate_snippet(
                theseus_selected.get("snippet") or theseus_selected.get("text") or ""
            ),
        }

    # Main and association are separate response fields; neither consumes the other's quota.
    resp = {
        "memories": out,
        "association": association_out,
        "day_counter": get_day_counter(),
    }

    heat_ids = [
        row.get("memory_id") for row in selected
        if row.get("source") != "cold_store" and row.get("memory_id")
    ]
    if theseus_selected and theseus_selected.get("memory_id"):
        heat_ids.append(theseus_selected.get("memory_id"))
    is_dry_run = dry_run.strip().lower() in ("1", "true", "yes", "on")
    if heat_ids and not is_dry_run:
        await _confirm_recall_heat(
            heat_ids, _recall_heat_event_id(submission_id, reflex_trace["request_id"], query, context),
        )
    _mark("heat")

    # Final injection is a heat source; do not immediately cancel it with legacy cool.
    _mark("post_injection")
    _total_ms = round((time.monotonic() - _t_start) * 1000)
    print(f"[反射弧·耗时] total={_total_ms}ms "
          + " ".join(f"{k}={v}ms" for k, v in _stage_t.items()), flush=True)
    try:
        selected_reasons = rerank_trace.get("selected_reasons") or {}
        fallback_info = rerank_trace.get("fallback") or {}
        fallback_ids = set(fallback_info.get("selected_ids") or []) if fallback_info.get("used") else set()
        selected_trace = []
        for m in search_selected:
            mid = m.get("memory_id")
            reason = selected_reasons.get(mid) or ("fallback硬选：reranker未选中有效主召回" if mid in fallback_ids else "")
            selected_trace.append(_trace_memory_item(m, reason=reason, lane="search"))
        if old_mem:
            selected_trace.append(_trace_memory_item(old_mem, reason=f"slot4:{slot4_source}", lane="slot4"))
        recall_trace.log_reflex(query, {
            **reflex_trace,
            "gate": "ok",
            "search_query": _clip_for_rerank(search_query, 220),
            "pool_count": len(pool),
            "seed_ids": seed_ids[:2],
            "rerank": {
                "cache_hit": cache_hit,
                "gate": rerank_trace.get("gate", ""),
                "anchor_search_timed_out": anchor_search_timed_out,
                "candidate_count": rerank_trace.get("candidate_count", 0),
                "allowed_pool_count": rerank_trace.get("allowed_pool_count", 0),
                "model_selected": rerank_trace.get("model_selected", []),
                "model_rejected": rerank_trace.get("model_rejected", []),
                "voyage_fallback": rerank_trace.get("voyage_fallback", False),
                "score_breakdown": rerank_trace.get("score_breakdown", []),
                "selected_evidence": rerank_trace.get("selected_evidence", {}),
                "selection_policy": rerank_trace.get("selection_policy", {}),
                "fallback": fallback_info,
                "empty_or_bad_json": rerank_trace.get("rerank_empty_or_bad_json", False),
                "cold_exact_override": rerank_trace.get("cold_exact_override", {"used": False}),
            },
            "cold_store": {
                "enabled": COLD_STORE_ENABLED,
                "elapsed_ms": cold_elapsed_ms,
                "candidate_count": len(cold_pool),
                "candidates": [
                    {"id": m.get("memory_id"), "match_type": m.get("match_type"),
                     "matched_terms": list(m.get("matched_terms") or [])[:8],
                     "score": m.get("score"), "timestamp": m.get("timestamp"),
                     **_trace_body_fields(m.get("snippet") or m.get("text") or "")}
                    for m in cold_pool[:3]
                ],
                "exact_override": rerank_trace.get("cold_exact_override", {"used": False}),
                "selected": [
                    {"id": m.get("memory_id"), "source": "cold_store"}
                    for m in search_selected if m.get("source") == "cold_store"
                ],
                "rejected": [
                    {"id": item.get("id"), "source": "cold_store", "reason": item.get("reason", "")}
                    for item in (rerank_trace.get("rejected") or [])
                    if item.get("source") == "cold_store"
                ] + (rerank_trace.get("cold_gate_rejected") or []),
            },
            "candidates_top": rerank_trace.get("candidates_top", []),
            "update_swaps": [
                {"superseded": m.get("superseded"), "resolved": m.get("memory_id")}
                for m in pool if m.get("via_update") and m.get("superseded")
            ],
            "selected": selected_trace,
            "rejected": rerank_trace.get("rejected", []),
            "legacy_graph_association": {
                "enabled": False,
                "outcome": "disabled_by_policy",
            },
            "theseus_association": theseus_trace,
            "anchor_association_shadow": anchor_assoc_shadow_trace,
            "final_injection": {
                "main_ids": [m.get("memory_id") for m in search_selected],
                "main_items": [
                    {"id": item.get("memory_id"),
                     "injected_snippet": _clip_for_rerank(item.get("snippet") or "", 420),
                     **({"via_update": True, "superseded": item.get("superseded")}
                        if item.get("via_update") else {})}
                    for item in out
                    if item.get("memory_id") in {
                        m.get("memory_id") for m in search_selected
                    }
                ],
                "association_ids": (
                    [theseus_selected.get("memory_id")] if theseus_selected else []
                ),
                "output_ids": [m.get("memory_id") for m in selected],
                "output_count": len(selected),
                "total_injected": len(selected) + (1 if theseus_selected else 0),
            },
            "slot4_source": slot4_source,
            "timings_ms": {**_stage_t, "total": _total_ms},
        })
    except Exception as e:
        print(f"[反射弧] recall_trace写入失败: {type(e).__name__}: {str(e)[:120]}")
    if _debug:
        resp["debug"] = {
            "candidates_top8": [
                {
                    "id": c.get("id"),
                    "tag": _clip_for_rerank(c.get("tag") or "", 60),
                    "score": c.get("score"),
                    "distance": c.get("distance"),
                    "source": c.get("source", "anchor_memory"),
                    "text_preview": _clip_for_rerank(c.get("text") or "", 80),
                    "recently_injected": c.get("recently_injected", False),
                }
                for c in (rerank_trace.get("candidates_top") or [])[:8]
            ],
            "model_selected": rerank_trace.get("model_selected", []),
            "model_rejected": rerank_trace.get("model_rejected", [])[:5],
            "voyage_fallback": rerank_trace.get("voyage_fallback", False),
            "score_breakdown": rerank_trace.get("score_breakdown", [])[:8],
            "gates": {
                "low_signal": False,
                "technical": False,
                "anchor_timeout": anchor_search_timed_out,
            },
            "timings_ms": {**_stage_t, "total": _total_ms},
            "theseus_association": theseus_trace,
            "anchor_association_shadow": anchor_assoc_shadow_trace,
            "pool_count": len(pool),
            "cold_count": len(cold_pool),
            "slot4_source": slot4_source,
        }
    return resp


if __name__ == "__main__":
    print(f"[反射弧] 上游: {UPSTREAM_URL}")
    print(f"[反射弧] 记忆API: {ANCHOR_API}")
    uvicorn.run(
        app,
        host=os.environ.get("ANCHOR_GATEWAY_HOST", "127.0.0.1"),
        port=int(os.environ.get("ANCHOR_GATEWAY_PORT", "8766")),
    )
