"""
影子索引 (Shadow Index) — AI agent记忆检索层的"话题钥匙"派生层。

一条长记忆 = 一个被摊平的质心向量, 对任何单一话题都"不够近"(实测多话题 blob
对纯 query "肚子疼" cosine=0.503, 卡在 0.50 门槛外被丢)。影子索引给长记忆用
DeepSeek 产几把"检索钥匙"(短检索短语 key + 指回原文的逐字 quote→span), 只嵌 key、
不嵌原文质心; 查询撞 key → 拉原文那段。

铁律 (spec §2, 不可违背):
  - 原文永远是正典, 一字不改。
  - DeepSeek 只产"检索钥匙", 派生、可丢、可重生成。
  - 召回返回给AI agent的是【原文那一段】, 不是 DeepSeek 写的字。
  - DeepSeek 的输出只能进 shadow 索引层(向量 + span 映射),
    永远不写回 memories.text, 永远不作为召回内容直接呈现给AI agent。
  - 即使 key 写错, 最坏只是"这把钥匙开错门"= 一次检索 miss, 错的东西进不了本体。

RAM 纪律 (实机 3.6G 总 / ~1.9G 可用, anchor-sse 已常驻 ~400MB bge 模型):
  本模块【绝不加载第二个 bge 模型】。所有 embedding 走 anchor-sse 进程里那个常驻的
  mem._embedder —— 通过传入的 embed_fn (upsert_shadows / shadow_search 都收 embedding
  或 embed_fn, 不自己 new SentenceTransformer)。backfill 脚本只做 DeepSeek(纯网络) +
  span 定位(纯 python), embedding 交给常驻服务的 /api/shadow/upsert。

Initial production calibration. 纯 additive: SHADOW_ENABLED=False → search 第三路完全旁路 = 今天行为。
prompt 由安手写, 放同目录 shadow_prompt.txt, 改词不动代码。
"""
import os
import re
import json
import sqlite3
from datetime import datetime

import httpx

_DIR = os.path.dirname(os.path.abspath(__file__))

# ─────────────────────────── 开关 / 常量 ───────────────────────────
SHADOW_ENABLED = True              # False = search() 第三路完全不跑 = 逐字回到今天的检索行为
SHADOW_COLLECTION = "memory_shadows"
SHADOW_VERSION = 1                 # 递增 → 跑 backfill 整批重生成 (旧 version 先删后写, spec §10)

# 检索门槛: shadow 命中本就该很近 (key 常含 query 原词)。None = 沿用调用方的 max_distance(0.50);
# 设具体值(如 0.40)= 给 shadow 单独更严门槛(spec §14.4)。先 None 保持纯 additive, 按留痕再调。
SHADOW_MAX_DIST = None

# 输入门控 (spec §5 + 生产约束): 只给 raw-level、多facet 记忆产钥匙。判据是 facet 数, 不是长度。
# understanding/cognition 等提炼层【不碰】(已是聚焦认知, whole-vector 够用, 不让派生层插手认知层)。
LONG_MULTI_MIN_SENTS = 4           # "长多句"兜底: 无显式结构但句数≥此
LONG_MULTI_MIN_LEN = 150           # 且 字数≥此, 才算实质多陈述

# DeepSeek 调用 (v4-pro 是 reasoning 模型, 实测 reasoning 1569~2967+ tok 不可控, 17~33s/条)
DS_MAX_TOKENS = 4096               # 给足, 防 reasoning 吃满后 content 空 (finish_reason=length)
DS_MAX_TOKENS_CAP = 16384           # 撞 length 时翻倍上限
DS_TEMPERATURE = 0.3
DS_TIMEOUT = 150
DS_MAX_KEYS = 5                    # 硬截断, 对齐 prompt "1~5 把"

_PROMPT_FILE = os.path.join(_DIR, "shadow_prompt.txt")
_ROUTES_FILE = os.path.join(_DIR, "model_routes.json")
_SENT_END = "。！？!?；;\n"


def enabled() -> bool:
    return SHADOW_ENABLED


def _utcnow() -> str:
    return datetime.utcnow().isoformat()


# ─────────────────────────── 输入门控 ───────────────────────────
def should_index(text: str, tag: str = "", collection: str = "", level: str = "raw") -> bool:
    """只给 raw-level、多facet 记忆产钥匙(spec §5 + 安: 不碰 understanding/cognition 提炼层)。
    判据是 facet 数(≥2个独立可检的点), 不是长度——长但单话题质心本就聚焦, 走 whole-vector。
    多facet信号: 枚举①/编号≥2/项目符≥2/【】≥2/→链≥2/、列≥3 / 或 长多句(≥4句且≥150字)。"""
    if not text:
        return False
    if (collection or "") == "wenku":
        return False                       # 文库自有 TOC/检索, 不在此仗
    if (level or "raw") != "raw":
        return False                       # 只索引 raw; understanding/cognition/refined/polished/diary 不碰
    t = text.strip()
    if re.search(r"[①②③④⑤⑥⑦⑧⑨⑩]", t):                          # 枚举
        return True
    if len(re.findall(r"(?m)^\s*\d+[\)\.、]", t)) >= 2:             # 编号列表
        return True
    if len(re.findall(r"(?m)^\s*[-•・*●]\s", t)) >= 2:              # 项目符
        return True
    if len(re.findall(r"【[^】]*】", t)) >= 2:                       # 多【…】段
        return True
    if (t.count("→") + t.count("->")) >= 2:                         # 箭头链/时间线
        return True
    if t.count("、") >= 3:                                          # 顿号罗列
        return True
    if (sum(t.count(c) for c in _SENT_END) >= LONG_MULTI_MIN_SENTS
            and len(t) >= LONG_MULTI_MIN_LEN):                      # 长多句兜底
        return True
    return False


# ─────────────────────────── prompt / 路由 ───────────────────────────
def load_prompt() -> str:
    """读安手写的 system prompt。读不到就抛, 让调用方明确失败, 绝不用空 prompt 污染生成。"""
    with open(_PROMPT_FILE, "r", encoding="utf-8") as f:
        return f.read().strip()


def build_messages(memory_text: str) -> list:
    """system = 安的规则 prompt; user = 记忆原文(prompt 结尾"下方用户提供的记忆原文"即指此)。"""
    return [
        {"role": "system", "content": load_prompt()},
        {"role": "user", "content": memory_text},
    ]


def load_routes(routes_path: str = None) -> list:
    """读取 consolidate 模型链；兼容旧 dict 和当前 list[dict] 配置。"""
    routes_path = routes_path or _ROUTES_FILE
    with open(routes_path, "r", encoding="utf-8") as f:
        raw = json.load(f).get("consolidate", {})
    configs = [raw] if isinstance(raw, dict) else (
        [cfg for cfg in raw if isinstance(cfg, dict)]
        if isinstance(raw, list) else []
    )
    return [{
        "key": cfg.get("key", ""),
        "model": cfg.get("model", "deepseek-chat"),
        "url": cfg.get("url", "https://api.deepseek.com/v1").rstrip("/"),
    } for cfg in configs]


def load_route(routes_path: str = None) -> dict:
    """向后兼容：返回 consolidate 模型链的第一路。"""
    routes = load_routes(routes_path)
    return routes[0] if routes else {
        "key": "",
        "model": "deepseek-chat",
        "url": "https://api.deepseek.com/v1",
    }


# ─────────────────────────── DeepSeek 生成 ───────────────────────────
def call_deepseek(memory_text: str, route: dict, retries: int = 1, return_usage: bool = False):
    """调 DeepSeek 产钥匙, 返回 [{"key","quote"}](return_usage=True 时返回 (keys, usage_dict))。
    失败抛异常(调用方决定跳过/记失败重试)。
    reasoning 模型: finish_reason=length 且 content 空(reasoning 吃满 max_tokens) → 翻倍重试。"""
    url = route["url"]
    if not url.endswith("/chat/completions"):
        url = url + "/chat/completions"
    headers = {"Authorization": f"Bearer {route['key']}", "Content-Type": "application/json"}
    body = {
        "model": route["model"],
        "messages": build_messages(memory_text),
        "temperature": DS_TEMPERATURE,
        "max_tokens": DS_MAX_TOKENS,
        "stream": False,
        "response_format": {"type": "json_object"},
    }
    last_err = None
    for _ in range(retries + 2):           # 至少给一次 length 翻倍的机会
        try:
            resp = httpx.post(url, headers=headers, json=body, timeout=DS_TIMEOUT)
            if resp.status_code != 200:
                last_err = f"http {resp.status_code}: {resp.text[:160]}"
                continue
            j = resp.json()
            ch = j["choices"][0]
            content = (ch.get("message") or {}).get("content") or ""
            if ch.get("finish_reason") == "length" and not content.strip():
                last_err = "finish_reason=length, empty content"
                if body["max_tokens"] < DS_MAX_TOKENS_CAP:
                    body["max_tokens"] = min(body["max_tokens"] * 2, DS_MAX_TOKENS_CAP)
                    continue
                break
            data = json.loads(content)
            keys = data.get("keys") if isinstance(data, dict) else data
            keys = _clean_keys(keys or [])
            return (keys, j.get("usage", {})) if return_usage else keys
        except Exception as e:             # 网络/JSON 等, 兜底重试
            last_err = str(e)
    raise RuntimeError(f"deepseek shadow-gen failed: {last_err}")


def call_deepseek_any(memory_text: str, routes: list, retries: int = 1,
                      return_usage: bool = False):
    """按配置顺序尝试 consolidate 模型链；一路失败才切下一路。"""
    errors = []
    for route in routes:
        if not route.get("key"):
            errors.append(f"{route.get('model', 'unnamed')}: missing key")
            continue
        try:
            return call_deepseek(memory_text, route, retries=retries,
                                 return_usage=return_usage)
        except Exception as exc:
            errors.append(f"{route.get('model', 'unnamed')}: {exc}")
    raise RuntimeError("all consolidate routes failed: " + "; ".join(errors))


def _clean_keys(keys: list) -> list:
    """清洗 + 硬截断到 DS_MAX_KEYS。只留 key/quote 双非空。key 去重(同 quote 下同 key 没意义)。"""
    out, seen = [], set()
    for k in keys[:DS_MAX_KEYS * 2]:       # 多给点缓冲再截, 防去重后不足
        if not isinstance(k, dict):
            continue
        key = (k.get("key") or "").strip()
        quote = k.get("quote") or ""        # quote 不 strip: 逐字定位以原样为准
        if not key or not quote.strip():
            continue
        sig = (key, quote)
        if sig in seen:
            continue
        seen.add(sig)
        out.append({"key": key, "quote": quote})
        if len(out) >= DS_MAX_KEYS:
            break
    return out


# ─────────────────────────── span 定位 / 句扩展 ───────────────────────────
def locate_span(text: str, quote: str):
    """逐字定位 quote 在原文的 [start,end)。找不到 → None(该钥匙 span 指向整条, 降级不阻塞)。
    spec §4B: span = quote 在 text 的字符起止(authoritative)。先原样 find, 再退一步 strip 后 find。"""
    if not text or not quote:
        return None
    pos = text.find(quote)
    if pos >= 0:
        return (pos, pos + len(quote))
    q = quote.strip()
    if q and q != quote:
        pos = text.find(q)
        if pos >= 0:
            return (pos, pos + len(q))
    return None


def expand_to_sentence(text: str, start: int, end: int, pad_sentences: int = 1):
    """把 [start,end) 扩到所在句边界, 再前后各含 pad_sentences 句(spec §14.3: 句范围±1句,
    防"那个/他"指代丢)。纯 python, 反射弧注入时用。空/越界自愈。"""
    if not text:
        return (start, end)
    n = len(text)
    start = max(0, min(start, n))
    end = max(start, min(end, n))
    bounds = [i for i, c in enumerate(text) if c in _SENT_END]   # 句末标点下标
    # 左界: start 之前第 (pad+1) 个句末标点之后; 不够则到 0
    left = [b for b in bounds if b < start]
    lo = (left[-(pad_sentences + 1)] + 1) if len(left) >= pad_sentences + 1 else 0
    # 右界: end 之后(含)第 (pad+1) 个句末标点之后; 不够则到 n
    right = [b for b in bounds if b >= end]
    if len(right) >= pad_sentences + 1:
        hi = right[pad_sentences] + 1
    elif right:
        hi = right[-1] + 1
    else:
        hi = n
    return (lo, hi)


# ─────────────────────────── SQLite: shadows 表 ───────────────────────────
def ensure_shadow_table(db_path: str):
    """在 memories.db 建 shadows 表(权威映射 + 可溯源, spec §4B)。幂等。"""
    conn = sqlite3.connect(db_path, timeout=5.0)
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS shadows (
                shadow_id   TEXT PRIMARY KEY,
                parent_id   TEXT NOT NULL,
                shadow_key  TEXT NOT NULL,    -- DeepSeek 产的检索短语
                quote       TEXT,             -- 原文逐字引用(可溯源/可重校验)
                span_start  INTEGER,          -- 在原文 text 里的字符起止(可空=指向整条)
                span_end    INTEGER,
                kind        TEXT DEFAULT 'topic',
                version     INTEGER DEFAULT 1,
                created_at  TEXT
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_shadows_parent ON shadows(parent_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_shadows_version ON shadows(version)")
        conn.commit()
    finally:
        conn.close()


def has_current_shadow(conn, parent_id: str, version: int = SHADOW_VERSION) -> bool:
    """该 parent 是否已有当前 version 的 shadow(backfill 幂等/续跑用)。"""
    row = conn.execute(
        "SELECT 1 FROM shadows WHERE parent_id=? AND version=? LIMIT 1",
        (parent_id, version)).fetchone()
    return row is not None


# ─────────────────────────── 写入 (在 anchor-sse 进程内调, 用常驻 embedder) ───────────────────────────
def _as_list(vec):
    return vec.tolist() if hasattr(vec, "tolist") else list(vec)


def delete_shadows(collection, conn, parent_id: str) -> int:
    """删某 parent 的全部 shadow(SQLite 为准再同步 Chroma, spec §4B/§10)。返回删除条数。"""
    rows = conn.execute("SELECT shadow_id FROM shadows WHERE parent_id=?", (parent_id,)).fetchall()
    ids = [r[0] for r in rows]
    if ids and collection is not None:
        try:
            collection.delete(ids=ids)
        except Exception:
            pass
    conn.execute("DELETE FROM shadows WHERE parent_id=?", (parent_id,))
    conn.commit()
    return len(ids)


def upsert_shadows(collection, conn, embed_fn, parent_id: str, parent_text: str,
                   keys: list, version: int = SHADOW_VERSION,
                   parent_tag: str = "", parent_ts: str = "") -> int:
    """把一条记忆的钥匙写进 shadow 层。embed_fn = mem._embedder.encode(常驻模型, 不另加载)。
    先删该 parent 旧 shadow(drop-then-write), 再写新。返回写入条数。
    keys: [{"key","quote"}](来自 DeepSeek)。span 在此用 parent_text 现算。
    嵌的是 key(spec §4A: embedding = encode(shadow_key), 不加 prefix, 与 memories 同约定)。"""
    delete_shadows(collection, conn, parent_id)
    now = _utcnow()
    ids, embs, docs, metas = [], [], [], []
    for k, item in enumerate(keys):
        key = item["key"]
        quote = item["quote"]
        span = locate_span(parent_text, quote)
        ss, se = span if span else (None, None)
        sid = f"{parent_id}#s{k}"
        conn.execute(
            "INSERT OR REPLACE INTO shadows "
            "(shadow_id,parent_id,shadow_key,quote,span_start,span_end,kind,version,created_at) "
            "VALUES (?,?,?,?,?,?,?,?,?)",
            (sid, parent_id, key, quote, ss, se, "topic", version, now))
        ids.append(sid)
        embs.append(_as_list(embed_fn(key)))
        docs.append(key)
        metas.append({
            "parent_id": parent_id,
            "span_start": ss if ss is not None else -1,   # Chroma metadata 不收 None
            "span_end": se if se is not None else -1,
            "kind": "topic",
            "shadow_version": version,
            "parent_tag": parent_tag or "",
            "parent_ts": parent_ts or "",
        })
    if ids and collection is not None:
        collection.upsert(ids=ids, embeddings=embs, documents=docs, metadatas=metas)
    conn.commit()
    return len(ids)


# ─────────────────────────── 检索 (第三路, 在 mem.search 内调) ───────────────────────────
def shadow_search(collection, query_embedding, fetch_n: int, version: int = SHADOW_VERSION) -> dict:
    """第三路: 拿 query 向量撞 memory_shadows。返回 {parent_id: {"dist","span","shadow_id","key"}},
    同一 parent 多个 shadow 命中取【最小距离】(最相关 facet 代表该记忆, spec §8.3)。
    query_embedding 来自 mem._embedder(query 本就要嵌一次, 零额外模型)。任何异常 → {} 不阻塞主检索。"""
    if collection is None:
        return {}
    try:
        res = collection.query(
            query_embeddings=[query_embedding],
            n_results=fetch_n,
            include=["metadatas", "distances", "documents"],
        )
    except Exception:
        return {}
    ids = (res.get("ids") or [[]])[0]
    metas = (res.get("metadatas") or [[]])[0]
    dists = (res.get("distances") or [[]])[0]
    docs = (res.get("documents") or [[]])[0]
    best = {}
    for sid, meta, dist, doc in zip(ids, metas, dists, docs):
        meta = meta or {}
        if version is not None and meta.get("shadow_version") != version:
            continue
        pid = meta.get("parent_id")
        if not pid:
            continue
        if pid not in best or dist < best[pid]["dist"]:
            ss = meta.get("span_start", -1)
            se = meta.get("span_end", -1)
            span = (ss, se) if (isinstance(ss, int) and isinstance(se, int)
                                and ss >= 0 and se > ss) else None
            best[pid] = {"dist": dist, "span": span, "shadow_id": sid, "key": doc}
    return best
