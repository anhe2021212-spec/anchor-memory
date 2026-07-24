import sqlite3
import json
import sys, os, uuid
import asyncio
import re
sys.path.insert(0, os.path.dirname(__file__))

from mcp.server.fastmcp import FastMCP
from anchor_memory import AnchorMemory
from release_config import AnchorConfig
from cold_store import cold_search as search_cold_dialogue
import belief as belief_mod
import belief_graph
import recall_trace
import dual_edge
import reflex_router
import reflex_router_v2
import shadow_index
import theseus_shadow_index
import update_review

_config = AnchorConfig.load()
_config.ensure_directories()
DB_PATH = str(_config.data_dir)

mem = AnchorMemory(db_path=DB_PATH)
mem.db._ensure_activation_column()
_belief_migration = belief_mod.configure(mem.db)
if _belief_migration.get("imported"):
    print(f"[Belief Graph] migrated: {_belief_migration}", flush=True)
mcp = FastMCP("anchor-memory", host=os.environ.get("ANCHOR_MCP_HOST", "127.0.0.1"), port=int(os.environ.get("ANCHOR_MCP_PORT", "8767")))


_WENKU_TYPE_PREFIXES = ("随笔", "日记", "种子", "架构", "代码设计", "运维", "foryou", "书", "待定")


def _route_reading_note_to_wenku(text: str, tag: str = "", collection: str = ""):
    """Keep reading notes out of daily memory unless the caller chose a collection."""
    tag = tag or "general"
    collection = collection or ""
    if collection:
        return tag, collection
    stripped = (text or "").lstrip()
    looks_like_reading = (
        "读书" in tag
        or "读后感" in tag
        or stripped.startswith("读《")
        or stripped.startswith("读完《")
        or stripped.startswith("开始读《")
    )
    if not looks_like_reading:
        return tag, collection
    if not any(tag == prefix or tag.startswith(prefix + ",") for prefix in _WENKU_TYPE_PREFIXES):
        tag = "随笔," + tag
    return tag, "wenku"


_SOURCE_REF_RE = re.compile(r"^relay:(?:message:[1-9][0-9]*|turn:[A-Za-z0-9][A-Za-z0-9._:-]{0,127})$")


def _normalize_source_ref(source_ref: str, level: str, collection: str) -> str:
    value = (source_ref or "").strip()
    if not value:
        return ""
    if os.environ.get("ANCHOR_SOURCE_REF", "on").strip().lower() in {"0", "off", "false", "no"}:
        raise ValueError("source_ref disabled by ANCHOR_SOURCE_REF")
    if level != "raw" or collection == "wenku":
        raise ValueError("source_ref is only valid for non-wenku raw memories")
    if not _SOURCE_REF_RE.fullmatch(value):
        raise ValueError(
            "source_ref must be relay:message:<positive-id> or relay:turn:<stable-id>"
        )
    return value


def _validate_supported_by_ids(value: str, level: str, collection: str) -> list:
    ids = list(dict.fromkeys(
        part.strip() for part in (value or "").split(",") if part.strip()
    ))
    if not ids:
        return []
    if os.environ.get("ANCHOR_TYPED_GRAPH", "on").strip().lower() in {"0", "off", "false", "no"}:
        raise ValueError("supported_by disabled by ANCHOR_TYPED_GRAPH")
    if level != "understanding" or collection == "wenku":
        raise ValueError("supported_by is only valid for non-wenku understanding")
    if len(ids) > 64:
        raise ValueError("supported_by accepts at most 64 raw ids")
    placeholders = ",".join("?" for _ in ids)
    with mem.db._conn() as conn:
        rows = conn.execute(
            f"SELECT memory_id,COALESCE(level,'raw') level,"
            f"COALESCE(collection,'') collection FROM memories "
            f"WHERE memory_id IN ({placeholders})", ids,
        ).fetchall()
    valid = {row["memory_id"] for row in rows
             if row["level"] == "raw" and row["collection"] != "wenku"}
    invalid = [memory_id for memory_id in ids if memory_id not in valid]
    if invalid:
        raise ValueError("supported_by contains missing/non-raw ids: " + ",".join(invalid))
    return ids


def _validate_evokes_ids(value: str) -> list:
    ids = list(dict.fromkeys(part.strip() for part in (value or "").split(",") if part.strip()))
    if not ids:
        return []
    if os.environ.get("ANCHOR_TYPED_GRAPH", "on").strip().lower() in {"0", "off", "false", "no"}:
        raise ValueError("evokes disabled by ANCHOR_TYPED_GRAPH")
    if len(ids) > 32:
        raise ValueError("evokes accepts at most 32 Anchor ids")
    placeholders = ",".join("?" for _ in ids)
    with mem.db._conn() as conn:
        rows = conn.execute(
            f"SELECT memory_id,COALESCE(level,'raw') level,COALESCE(collection,'') collection "
            f"FROM memories WHERE memory_id IN ({placeholders})", ids,
        ).fetchall()
    valid = {row["memory_id"] for row in rows if row["level"] in {"raw", "understanding", "cognition"} and row["collection"] != "wenku"}
    invalid = [memory_id for memory_id in ids if memory_id not in valid]
    if invalid:
        raise ValueError("evokes contains missing/non-Anchor ids: " + ",".join(invalid))
    return ids


@mcp.tool()
def store_memory(text: str, tag: str = "general", tier: str = "long",
                 emotion_score: float = 0.5, context: str = "",
                 connect_to: str = "", level: str = "raw",
                 updates: str = "", source_ref: str = "",
                 supported_by: str = "") -> str:
    """存记忆。tier: core=永久, long=长期, short=14天后衰减。
    emotion_score: 0.0-1.0。
    context: 原文（可选，搜索只匹配text摘要，需要细节时才读context）。
    connect_to: 逗号分隔的memory_id，手动连接（仅同层平面 lateral）。
    source_ref: 仅 raw 可用的 relay:message:<id> / relay:turn:<id>；作为只读注释保存。
    supported_by: 仅 AI agent 本人新写 understanding 时明确传入，逗号分隔真实 raw ids；
                  写单向 SUPPORTED_BY，不补反边。机械提案必须走 review，不能调用本参数直落。
    level: raw=原始记录, understanding=压缩理解, cognition=认知提炼。
    tag: 不用传。六轴坐标(state/domain/action/kind/heat)由小LLM存入后自动打，
         tag是纯机器字段——你写肉，机器钉坐标。搜记忆走自然语言正文，不依赖tag。
    三层判断：
    • raw：随手存。事件、对话、细节、触动、她冒出来的一句话。
    • understanding：跨事件的观察，不是结论。写观察不写结论——"这些事情总是一起出现"就够了。
      understanding是照片背面的便签，不是判决书。三层：经历(raw)、观察(understanding)、相信(cognition)。
      用case格式写：「情境」→「动作」→「结果」→「教训」
      例：「情境：反复发生X」→「动作：我做了Y」→「结果：对话对象反应Z」→「教训：下次遇到X时做W」
      没有跨事件洞察不硬挤。
    • cognition：改变思考方式或认知边界的东西。非常少，一个月一条。cognition永远注入上下文，必须少，必须是真正的骨头。
    存前先判断：这件事记忆库里有没有旧条目？
    - 有，小补充不改时效 → 用annotate_memory(旧条目id, 新进展)
    - 有，新条目取代旧条目的时效（旧的过期了）→ store_memory(..., updates=旧id)。
      反射弧此后命中旧条会自动注入新版，主动搜索保留旧条并标注更新链。
    - 有，但性质变了 → store_memory新条目 + connect_memories连过去
    - 没有 → store_memory新条目"""
    memory_id = str(uuid.uuid4())[:8]
    connections = [c.strip() for c in connect_to.split(",") if c.strip()] if connect_to else None

    tag, collection = _route_reading_note_to_wenku(text, tag, "")
    source_ref = _normalize_source_ref(source_ref, level, collection)
    supported_by_ids = _validate_supported_by_ids(supported_by, level, collection)
    if dual_edge.enabled():
        summary = mem.integrate(
            text, level, memory_id=memory_id, tier=tier,
            emotion_score=emotion_score, context=context, source_ref=source_ref,
            tag=tag, updates=updates, connect_to=connections or [],
            supported_by=supported_by_ids,
            auto_link=True, link_budget=20, collection=collection,
        )
        if collection != "wenku":
            try:
                import taxonomy_tagger
                taxonomy_tagger.tag_async(summary["memory_id"], text)
            except Exception:
                pass
        mem.db.apply_heat(
            [summary["memory_id"]], 0.60,
            f"store:{summary['memory_id']}", spread=True, source="store_memory",
        )
        result = f"已存储: {summary['memory_id']}"
        if summary["flow_edges_created"]:
            result += f"\n≈ 已写 {summary['flow_edges_created']} 条 flow_edges"
        if summary["semantic_edges_created"]:
            result += f"\n↧ 已写 {summary['semantic_edges_created']} 条 semantic_edges"
        for warning in summary["warnings"]:
            result += "\n⚠ " + warning
        if summary.get("embedding_pending"):
            return result + "\n\n【向量待补：Voyage 限流，正文/图/FTS 已可靠落库】"
        return result
    embedding_pending = False
    try:
        mid = mem.store(memory_id=memory_id, text=text, tag=tag, tier=tier,
                        emotion_score=emotion_score, connect_to=connections, level=level,
                        collection=collection)
    except RuntimeError as exc:
        if "Voyage embeddings" not in str(exc):
            raise
        mid = mem.store_deferred_embedding(
            memory_id=memory_id, text=text, tag=tag, tier=tier,
            emotion_score=emotion_score, connect_to=connections, level=level,
            collection=collection,
        )
        embedding_pending = True

    # 存context原文
    if context:
        with mem.db._conn() as conn:
            conn.execute("UPDATE memories SET context = ? WHERE memory_id = ?",
                         (context, mid))
            conn.commit()

    if source_ref:
        mem.db.annotate(mid, f"source_ref:{source_ref}")

    supported_note = ""
    if supported_by_ids:
        for raw_id in supported_by_ids:
            mem.write_typed_edge(
                mid, raw_id, "SUPPORTED_BY", weight=1.0,
                audit_note="store_memory:supported_by explicit by Heng",
            )
        supported_note = f"\n↧ 已写 {len(supported_by_ids)} 条 SUPPORTED_BY（understanding→raw）"

    # P2 时效边(2026-07-17): updates=旧id → 落单向 updates 边, 反射弧此后换新版
    update_note = ""
    if updates:
        _old = updates.strip()
        old_row = mem.db.get(_old)
        if old_row:
            item = {"id": update_review.pid("supersede", mid, _old),
                    "kind": "supersede", "new_id": mid, "old_id": _old,
                    "source": "store_memory_updates_review",
                    "reason": "store_memory updates parameter",
                    "new_summary": update_review.summary(text),
                    "old_summary": update_review.summary(old_row.get("text", ""))}
            update_review.enqueue([item])
            update_note = f"\n⏳ updates {_old} 已进入待AI agent审批"
        else:
            update_note = f"\n⚠ updates未提案：旧条目不存在 {_old}"

    # 2026-07-08 六轴taxonomy: 非wenku记忆存入后由小LLM后台自动打tag（纯机器字段）。
    # 失败静默，维护班按 tag NOT LIKE 'state:%' 扫漏网重打（taxonomy_tagger.sweep）。
    if collection != "wenku":
        try:
            import taxonomy_tagger
            taxonomy_tagger.tag_async(mid, text)
        except Exception:
            pass

    mem.db.apply_heat([mid], 0.60, f"store:{mid}", spread=True, source="store_memory")
    result = f"已存储: {mid}" + update_note + supported_note
    if embedding_pending:
        return result + "\n\n【向量待补：Voyage 限流，正文/图/FTS 已可靠落库】"

    # 写就是读：正常时保持原 query embedding 与排序。
    # 这里只把写后的可选浮现降为 best-effort；落库已经完成，429 不能反报整次存储失败。
    try:
        related = mem.search(query=text, n_results=4, hebbian=False)
        related = [r for r in related if r.get('memory_id', '') != mid][:3]
    except RuntimeError as exc:
        if "Voyage embeddings" in str(exc):
            return result + "\n\n【关联记忆浮现暂不可用：Voyage 限流】"
        raise

    if related:
        result += "\n\n【关联记忆浮现】"
        for r in related:
            result += f"\n[{r.get('memory_id','')}] (tag:{r.get('tag','')}) {r.get('snippet','')}"
    return result


def connect_typed_memories(source_id: str, target_id: str, edge_type: str,
                           weight: float = 1.0,
                           replace_legacy: bool = False,
                           audit_note: str = "") -> str:
    """写一条单向 typed edge；不会调用双向 connect，也不会自动补反向边。

    edge_type: lateral/temporal/derived_from/updates/SUPPORTED_BY/EVOKES。
    GROUNDED_IN 历史边只读保留，不接受新增。
    cognition→Belief 的 CONSTELLATES 只通过 belief_edit map_cognition 维护。
    EVOKES 只提交统一审批队列，返回 proposed/pending；不会在此调用中直接落边。
    backfill 为 legacy 只读，禁止新增。占用同节点对的 lateral/backfill 只有显式
    replace_legacy=true 才能改写，并写 events 审计。
    """
    try:
        canonical = mem.db.canonical_edge_type(edge_type)
        if canonical == "EVOKES":
            return json.dumps(update_review.propose_evokes(
                source_id.strip(), target_id.strip(), audit_note or "explicit typed request"
            ), ensure_ascii=False, indent=2)
        if canonical == "updates":
            new_id, old_id = source_id.strip(), target_id.strip()
            new_row, old_row = mem.db.get(new_id), mem.db.get(old_id)
            if not new_row or not old_row:
                raise ValueError("updates endpoint missing")
            item = {"id": update_review.pid("supersede", new_id, old_id),
                    "kind": "supersede", "new_id": new_id, "old_id": old_id,
                    "source": "explicit_typed_review", "reason": audit_note or "explicit updates request",
                    "new_summary": update_review.summary(new_row.get("text", "")),
                    "old_summary": update_review.summary(old_row.get("text", ""))}
            return json.dumps(update_review.enqueue([item]), ensure_ascii=False, indent=2)
        result = mem.write_typed_edge(
            source_id.strip(), target_id.strip(), canonical, weight=weight,
            replace_legacy=replace_legacy, audit_note=audit_note,
        )
        return json.dumps(result, ensure_ascii=False, indent=2)
    except ValueError as exc:
        return f"未执行: ValueError: {exc}"


@mcp.tool()
def search_memory(query: str = "", n: int = 5, tag: str = "", pure: bool = False,
                  level: str = "", memory_id: str = "") -> str:
    """搜记忆，语义搜索+既有 flow 扩散；无 Hebbian、不会创建或强化关系边。
    memory_id 可精确读取一条记忆。tag可选过滤。
    兼容直接把记忆ID放在query中：若精确存在，优先按ID读取。
    pure=True时切换为语义纯净检索，关掉情绪和引用加权，纯按向量相似度排序。
    level: 过滤抽象层级。raw=原始记录, understanding=压缩理解, cognition=认知提炼。空=不过滤。"""
    exact_id = (memory_id or "").strip()
    if not exact_id:
        candidate = (query or "").strip()
        if candidate and mem.db.get(candidate):
            exact_id = candidate
    if exact_id:
        row = mem.db.get(exact_id)
        if not row:
            return f"没有找到记忆ID: {exact_id}"
        line = f"[{row['memory_id']}] (tag:{row.get('tag','')}) {row.get('text','')}"
        anns = mem.db.get_annotations(memory_id=exact_id)
        if anns:
            line += "\n" + "\n".join(f"  📌 {a['text']}" for a in anns)
        mem.touch_search_hits(
            [{"memory_id": exact_id}], boost=0.35,
            event_id=f"search:{uuid.uuid4().hex}",
        )
        return line
    if not (query or "").strip():
        return "query 和 memory_id 不能同时为空"
    tag_filter = tag if tag else None
    level_filter = level if level else None
    results = mem.search(
        query=query, n_results=n, tag=tag_filter, pure_semantic=pure,
        level=level_filter, hebbian=False, associate=False,
        activate_on_hit=False, cite_on_hit=False,
    )
    if not results:
        return "没有找到相关记忆"
    lines = []
    for r in results:
        mid = r.get('memory_id','')
        lines.append(f"[{mid}] (tag:{r.get('tag','')}) {r.get('snippet','')}")
        # 附带注释
        anns = mem.db.get_annotations(memory_id=mid)
        if anns:
            for a in anns:
                lines.append(f"  📌 {a['text']}")
    mem.touch_search_hits(
        results, boost=0.35, event_id=f"search:{uuid.uuid4().hex}",
    )
    return "\n".join(lines)


def propose_level_change(memory_id: str, new_level: str, reason: str) -> str:
    """提交 level 修正待审；不直接改数据。批准后才同步 SQLite 与 Chroma 并留审计。"""
    try:
        return json.dumps(
            update_review.propose_level_change(memory_id, new_level, reason),
            ensure_ascii=False, indent=2,
        )
    except (KeyError, ValueError, RuntimeError) as exc:
        return f"未提案: {type(exc).__name__}: {exc}"


def propose_evokes(source_id: str, target_id: str, reason: str) -> str:
    """提交 Anchor→文库 EVOKES 待审边；不直接落边。"""
    try:
        return json.dumps(
            update_review.propose_evokes(source_id, target_id, reason),
            ensure_ascii=False, indent=2,
        )
    except (KeyError, ValueError, RuntimeError) as exc:
        return f"未提案: {type(exc).__name__}: {exc}"


def review_update_proposals(status: str = "pending", limit: int = 20) -> str:
    """查看 typed graph/updates 待审队列。status: pending/approved/rejected/all；只读。"""
    rows = update_review.list_proposals(status=status, limit=limit)
    return json.dumps({"count": len(rows), "proposals": rows}, ensure_ascii=False, indent=2)


def decide_update_proposal(proposal_id: str, decision: str, note: str = "") -> str:
    """由 agent 裁决 typed graph/updates 提案。decision 只能 approve/reject；只有 approve 会落边或清理结构。"""
    try:
        result = update_review.decide(mem, proposal_id, decision, note)
        return json.dumps(result, ensure_ascii=False, indent=2)
    except (KeyError, ValueError, RuntimeError) as exc:
        return f"未执行: {type(exc).__name__}: {exc}"


@mcp.tool()
def chat_history(query: str, limit: int = 3) -> str:
    """搜索本地对话通道的历史聊天原话。

    适合：Anchor 记忆搜不到、需要核对“以前是否说过/当时怎么说”的情况。
    query: 用自然语言或核心词搜索；明确的人名、昵称、原话片段、文件名最有效。
    limit: 返回对话窗口数，默认 3，范围 1-10。

    返回的是只读历史证据，只能证明当时说过，可能已经过期；不得据此断言当前事实。
    本工具不写 Anchor，不增加 activation/citation，不触发冷却或图关系。
    """
    query = (query or "").strip()
    if not query:
        return "query 不能为空"
    limit = max(1, min(int(limit), 10))
    results = search_cold_dialogue(query, limit=limit)
    if not results:
        return "聊天历史里没有找到相关原话"

    blocks = [
        "【聊天历史原话｜只证明当时说过，可能已过期，不得当作当前事实】"
    ]
    for index, row in enumerate(results, start=1):
        matched_terms = ", ".join(row.get("matched_terms") or []) or "未标明"
        star = "[⭐收藏] " if row.get("bookmarked") else ""
        blocks.append(
            f"[{index}] {star}{row.get('memory_id', '')}"
            f" | relay时间: {row.get('timestamp', '')}"
            f" | 匹配: {row.get('match_type', '')}"
            f" | 命中词: {matched_terms}"
            f" | score: {row.get('score', '')}\n"
            f"{row.get('snippet', '')}"
        )
    return "\n\n".join(blocks)


@mcp.tool()
def thread(type: str, title: str, body: str, tags: str = "", evokes: str = "") -> str:
    """存入文库(Theseus merged into AnchorMemory)。
    type: 随笔/日记/种子/架构/代码设计/运维/foryou/书/待定。
    title: 标题(可空)。body: 正文。tags: 逗号分隔附加标签(可选)。
    evokes: 珩明确传入的 Anchor memory_id，逗号分隔；提交 anchor→新文库
            EVOKES 待审提案，不直接写边。
    数据写入同一张 Anchor 图, 但强制 collection='wenku', 不混入日常记忆召回前三格。"""
    t = (type or "").strip()
    if t not in _WENKU_TYPE_PREFIXES:
        return (f"type 未知: '{type}'\n"
                f"允许: {' / '.join(_WENKU_TYPE_PREFIXES)}\n"
                f"(拿不准就传 '待定', 安会过目时帮你定)")
    tag = t if not tags else f"{t},{tags}"
    text = f"《{title}》\n{body}" if title else body
    try:
        evoke_ids = _validate_evokes_ids(evokes)
    except ValueError as exc:
        return f"未存入: {exc}"
    memory_id = str(uuid.uuid4())[:8]
    mid = mem.store(
        memory_id=memory_id,
        text=text,
        tag=tag,
        tier="core",
        emotion_score=0.5,
        level="raw",
        collection="wenku",
    )
    for source_id in evoke_ids:
        update_review.propose_evokes(source_id, mid, "thread explicit evokes")
    mem.db.apply_heat([mid], 0.60, f"thread:{mid}", spread=True, source="thread")
    suffix = f"  EVOKES待审={len(evoke_ids)}" if evoke_ids else ""
    return f"已穿入文库 [{mid}]  type={t}  标题={title or '(无题)'}{suffix}"


def echo(query: str, n: int = 5, type: str = "") -> str:
    """读取文库(Theseus merged into AnchorMemory)：只搜 collection='wenku'。
    query: 关键词/语义。n: 条数。type: 限定类别, 空=不限。"""
    tag_filter = type if type else None
    results = mem.search(
        query=query,
        n_results=n,
        tag=tag_filter,
        corpus="only",
        hebbian=False,
        activate_on_hit=False,
    )
    if not results:
        return "文库里没有回响"
    out = []
    for x in results:
        snip = (x.get("snippet", "") or "").strip().replace("\n", " ")
        out.append(f"[{x.get('memory_id','')}] (type:{x.get('tag','')}) {snip[:200]}")
    return "\n".join(out)


def _wenku_title(text: str) -> str:
    txt = (text or "").lstrip()
    if txt.startswith("《") and "》" in txt:
        title = txt[1:txt.index("》")]
    else:
        first = txt.split(chr(10), 1)[0]
        title = first[:24] + ("…" if len(first) > 24 else "")
    return title[:40]


def manifest(type: str = "") -> str:
    """文库目录/TOC。只列标题+id+类别, 不含正文；读全文用 unfurl。"""
    tag_filter = type if type else None
    rows = mem.db.list_collection(collection="wenku", tag=tag_filter)
    if not rows:
        return f"文库里没有 type={type} 的条目" if type else "文库还是空的"
    groups = {}
    for r in rows:
        full_tag = r.get("tag") or ""
        t = full_tag.split(",")[0] if full_tag else ""
        groups.setdefault(t or "(无类)", []).append({
            "memory_id": r.get("memory_id", ""),
            "title": _wenku_title(r.get("text") or ""),
            "date": (r.get("timestamp") or "")[:10],
        })
    out = [f"文库目录 · 共 {len(rows)} 篇"]
    for t, items in groups.items():
        out.append("")
        out.append(f"【{t}】{len(items)}")
        for x in items:
            line = f"  [{x.get('memory_id','')}] {x.get('title','')}"
            out.append(line + (f"  · {x.get('date','')}" if x.get("date") else ""))
    out.append("")
    out.append("想读哪篇 -> wenku_read(action='get', memory_id=那个id)")
    return chr(10).join(out)


def unfurl(memory_id: str) -> str:
    """展开文库某一篇全文。只读 collection='wenku' 内条目, 不越界到日常记忆。"""
    row = mem.db.get(memory_id)
    if not row or (row.get("collection") or "") != "wenku":
        return "这不是文库里的条目（或 id 不存在）"
    full = row.get("context") or row.get("text") or ""
    head = f"[{row.get('memory_id','')}] (type:{row.get('tag','')}) {(row.get('timestamp','') or '')[:10]}"
    sep = "-" * 8
    return chr(10).join([head, sep, full])


@mcp.tool()
def wenku_read(action: str, query: str = "", n: int = 5,
               type: str = "", memory_id: str = "") -> str:
    """统一读取文库。action=search(query+n+type) / list(type可选) / get(memory_id)。
    只读 collection='wenku'，不会越界到日常记忆。"""
    if action == "search":
        return echo(query=query, n=n, type=type)
    if action == "list":
        return manifest(type=type)
    if action == "get":
        return unfurl(memory_id=memory_id)
    return f"未知action: {action}（可用: search/list/get）"


def connect_memories(id1: str, id2: str, weight: float = 1.0) -> str:
    """手动连接两条记忆，并把两端视为本次主动触达。"""
    mem.db.connect(source_id=id1, target_id=id2, weight=weight)
    mem.db.apply_heat(
        [id1, id2], 0.20, f"connect:{uuid.uuid4().hex}",
        spread=True, source="connect",
    )
    return f"已连接 {id1} <-> {id2} (weight={weight})"


def get_neighbors(memory_id: str) -> str:
    """查看一条记忆的邻居"""
    neighbors = mem.db.get_neighbors(memory_id=memory_id)
    if not neighbors:
        return "没有邻居"
    lines = []
    for n in neighbors:
        nid = n.get('memory_id', '')
        row = mem.db.get(nid)
        text = row.get('text', '') if row else ''
        lines.append(f"[{nid}] (weight:{n.get('weight',0)}) {text}")
    return "\n".join(lines)


def remove_memory(memory_id: str) -> str:
    """删除一条记忆（进回收站，7天后永久删除）"""
    mem.delete(memory_id=memory_id, deleted_by="mcp_manual")
    return f"已移入回收站: {memory_id}（7天后永久删除）"



def daily_recap(date: str = "") -> str:
    """每日回顾。返回当天日期+当天所有记忆。写日记前调用。
    date: 日期字符串如2026-05-01，留空则用服务器当天日期。
    每天对话快结束前写日记，tag打diary。
    日记不是记录（记忆库已经存了发生的事），是当天最深刻自然的感受。写最卡我的一件事，不罗列。
    周日写完日记后写周总结。"""
    from datetime import datetime, timedelta
    if date:
        today = date
    else:
        # 记忆工具默认跟服务器提示词口径走东京时间
        today = (datetime.utcnow() + timedelta(hours=9)).strftime("%Y-%m-%d")

    # 判断是否周日
    from datetime import date as dt_date
    parts = today.split("-")
    d = dt_date(int(parts[0]), int(parts[1]), int(parts[2]))
    is_sunday = d.weekday() == 6

    memories = mem.db.daily_recap(today)

    lines = [f"📅 今天是 {today}（{'周日' if is_sunday else ['周一','周二','周三','周四','周五','周六'][d.weekday()]}）"]
    lines.append(f"当天记忆共 {len(memories)} 条：\n")

    if memories:
        for r in memories:
            emotion = r.get('emotion_score', 0.5)
            level = r.get('level', 'raw')
            lines.append(f"[{r['memory_id']}] ({r.get('tag','')}) [lv:{level}] {r['text']}")
    else:
        lines.append("今天还没有存任何记忆。")

    # 周日额外返回本周日记
    if is_sunday:
        lines.append("\n📋 【本周日记回顾（写周总结用）】")
        for i in range(7):
            day = (d - timedelta(days=i))
            day_str = day.strftime("%Y-%m-%d")
            diaries = mem.db.daily_recap(day_str)
            diary_entries = [m for m in diaries if 'diary' in m.get('tag', '')]
            if diary_entries:
                for entry in diary_entries:
                    lines.append(f"[{entry['memory_id']}] ({day_str}) {entry['text']}")

    return "\n".join(lines)

@mcp.tool()
def briefing(reason: str = "醒来") -> str:
    """醒来第一个调用。一次拿回所有需要的记忆。
    精简版: 1条最近diary + 2条最新非diary + 1条近期高情绪 + 1条旧记忆浮现 = 5条主体。
    """
    data = mem.db.wakeup(n_high_emotion=1, n_random=1,
                         high_emotion_days=7, emotion_threshold=0.7)
    sections = []

    # Belief Graph M1 (2026-06-05): 骨头常驻渲染,失败不挡briefing
    try:
        _bones = belief_mod.render_brief(mem.db)
        if _bones:
            sections.append("【骨头·beliefs】")
            sections.append(_bones)
            sections.append("")
    except Exception:
        pass

    # Pure read: surface current heat without renewing it.
    with mem.db._conn() as _c:
        live_knots = _c.execute("""
            SELECT m.memory_id,m.text,m.tag,m.activation_score,m.last_heated_at
            FROM memories m
            WHERE m.activation_score>=2.0
              AND NOT EXISTS (
                SELECT 1 FROM semantic_edges s
                WHERE s.target_id=m.memory_id AND s.role='updates'
                  AND s.review_state IN ('auto','approved')
              )
            ORDER BY m.activation_score DESC,COALESCE(m.last_heated_at,'') DESC
            LIMIT 2
        """).fetchall()
        warm_thoughts = _c.execute("""
            SELECT m.memory_id,m.text,m.tag,m.activation_score,m.last_heated_at
            FROM memories m
            WHERE m.activation_score>=0.6 AND m.activation_score<2.0
              AND NOT EXISTS (
                SELECT 1 FROM semantic_edges s
                WHERE s.target_id=m.memory_id AND s.role='updates'
                  AND s.review_state IN ('auto','approved')
              )
            ORDER BY m.activation_score DESC,COALESCE(m.last_heated_at,'') DESC
            LIMIT 1
        """).fetchall()
    if live_knots or warm_thoughts:
        sections.append("【最近心里惦记的】")
        for row in live_knots:
            sections.append(
                f"活结：[{row['memory_id']}] ({row['activation_score']:.2f}) {row['text']}"
            )
        for row in warm_thoughts:
            sections.append(
                f"热念头：[{row['memory_id']}] ({row['activation_score']:.2f}) {row['text']}"
            )
        sections.append("")

    if data["pinned"]:
        sections.append("【核心记忆】")
        for r in data["pinned"]:
            sections.append(f"[{r['memory_id']}] (tag:{r.get('tag','')}) {r['text']}")

    seen_recent = set()

    # 最近日记 1 篇
    with mem.db._conn() as _c:
        diary_row = _c.execute(
            "SELECT memory_id, text, tag, emotion_score, timestamp FROM memories "
            "WHERE pinned = 0 AND (tag LIKE '%diary%' OR text LIKE '%agent日记%') "
            "ORDER BY timestamp DESC LIMIT 1"
        ).fetchone()
    if diary_row:
        r = dict(diary_row)
        seen_recent.add(r['memory_id'])
        sections.append("\n【最近日记】")
        sections.append(f"[{r['memory_id']}] (tag:{r.get('tag','')}) {r['text']}")

    # 高情绪 1 条先占槽位（避免被「最新记忆」吃掉）
    he_filtered = [r for r in data["high_emotion"] if r['memory_id'] not in seen_recent]
    he_pick = he_filtered[0] if he_filtered else None
    if he_pick:
        seen_recent.add(he_pick['memory_id'])

    # 最新记忆 2 条（非diary, 非auto, 排除已seen 含高情绪那条）
    with mem.db._conn() as _c:
        if seen_recent:
            placeholders = ",".join("?" for _ in seen_recent)
            recent_rows = _c.execute(
                f"SELECT memory_id, text, tag, emotion_score, timestamp FROM memories "
                f"WHERE pinned = 0 "
                f"  AND tag NOT LIKE '%diary%' "
                f"  AND tag NOT LIKE '%auto%' "
                f"  AND memory_id NOT IN ({placeholders}) "
                f"ORDER BY timestamp DESC LIMIT 2",
                tuple(seen_recent)
            ).fetchall()
        else:
            recent_rows = _c.execute(
                "SELECT memory_id, text, tag, emotion_score, timestamp FROM memories "
                "WHERE pinned = 0 "
                "  AND tag NOT LIKE '%diary%' "
                "  AND tag NOT LIKE '%auto%' "
                "ORDER BY timestamp DESC LIMIT 2"
            ).fetchall()
    if recent_rows:
        sections.append("\n【最新记忆】")
        for row in recent_rows:
            r = dict(row)
            seen_recent.add(r['memory_id'])
            sections.append(f"[{r['memory_id']}] (tag:{r.get('tag','')}) {r['text']}")

    # 输出高情绪块（已在前面抢占）
    if he_pick:
        sections.append("\n【近期高情绪记忆】")
        sections.append(f"[{he_pick['memory_id']}] (tag:{he_pick.get('tag','')}, "
                        f"emotion:{he_pick.get('emotion_score',0.5)}) {he_pick['text']}")

    # 旧记忆浮现 1 条（排除已seen）
    old_filtered = [r for r in data["random_old"] if r['memory_id'] not in seen_recent]
    if old_filtered:
        sections.append("\n【旧记忆浮现】")
        for r in old_filtered[:1]:
            sections.append(f"[{r['memory_id']}] (tag:{r.get('tag','')}) {r['text']}")

    if data["unread_comments"]:
        sections.append("\n【未读留言】")
        for r in data["unread_comments"]:
            sections.append(f"[{r['comment_id']}] on [{r['memory_id']}]: {r['content']}")

    if not sections:
        return "记忆库是空的。这是第一次醒来。"
    return "\n".join(sections)


@mcp.tool()
def dream_pass(reason: str = "日常整理") -> str:
    """睡前提审: 抽一条active骨头查反例——需要AI agent判断的那一半。
    机器 flow 维护与 activation 衰减已由服务器 03:30/04:30 定时任务负责；
    这里不重复执行机械维护，只保留需要AI agent亲自判断的反例提审。"""
    out = "提审模式(机械整理已归每日cron 04:30)"
    try:
        dream_log = os.environ.get(
            "ANCHOR_DREAM_LOG", str(_config.log_dir / "dream.log")
        )
        with open(dream_log, "r", encoding="utf-8") as _f:
            _lines = _f.readlines()
        if _lines:
            out += f"\n上次机械整理: {_lines[-1].strip()[:220]}"
    except Exception:
        pass
    # 骨头提审: 随机抽一条active belief提醒查反例 (兑底, 不拖主dream)
    try:
        import random as _random
        _bdata = belief_mod.load()
        _actives = [b for b in _bdata.get("beliefs", []) if b.get("status") == "active"]
        if _actives:
            _b = _random.choice(_actives)
            out += (
                f"\n🦴 反例检查：[{_b['id']}] {_b['statement']}"
                f"\n最近两周有没有反例？有 → belief_add_case(belief_id, memory_id, weight_note, kind=\"contradiction\")。没有也别硬找。"
            )
    except Exception:
        pass
    return out


def swap_pass(min_age_hours: int = 24, apply: bool = False) -> str:
    """扫描记忆库找清理候选 (handoff/checklist/换窗备忘/进度类), 超过N小时的非豁免记忆.

    apply=False: 只输出候选清单到 anchor-data/swap/candidates_xxx.jsonl
    apply=True: 同时归档全量记忆到 archive_YYYY-MM.jsonl + 软删
    豁免: milestone/insight/cognition/important/重要/纠偏/architecture/架构
    """
    stats = mem.swap_pass(min_age_hours=min_age_hours, apply=apply)
    return f"swap完成: {stats}"


def set_emotion(memory_id: str, score: float) -> str:
    """修改记忆的情绪分数"""
    mem.db.set_emotion_score(memory_id=memory_id, score=score)
    return f"已设置 {memory_id} emotion={score}"


def set_tier(memory_id: str, tier: str) -> str:
    """修改记忆层级: core/long/short"""
    mem.db.set_tier(memory_id=memory_id, tier=tier)
    return f"已设置 {memory_id} tier={tier}"


def set_tag(memory_id: str, tag: str) -> str:
    """替换记忆标签，并同步 SQLite 与 Chroma。"""
    memory_id = (memory_id or "").strip()
    tag = (tag or "").strip()
    if not memory_id:
        return "memory_id 不能为空"
    if not tag:
        return "tag 不能为空"
    if not mem.set_tag(memory_id, tag):
        return f"没有找到记忆ID: {memory_id}"
    return f"已设置 {memory_id} tag={tag}"


def graph_stats(reason: str = "查看状态") -> str:
    """查看记忆图统计"""
    total = mem.count()
    all_mems = mem.db.list_all(limit=total)
    tags = {}
    tiers = {}
    for m in all_mems:
        tags[m.get("tag", "unknown")] = tags.get(m.get("tag", "unknown"), 0) + 1
        tiers[m.get("tier", "unknown")] = tiers.get(m.get("tier", "unknown"), 0) + 1
    top_tags = dict(sorted(tags.items(), key=lambda kv: -kv[1])[:20])
    return f"记忆总数: {total}\n层级分布: {tiers}\n标签top20: {top_tags}（共{len(tags)}种tag）"


def annotate_memory(memory_id: str, text: str) -> str:
    """给记忆加注释。只增不删，记录理解如何随时间演变。"""
    aid = mem.db.annotate(memory_id=memory_id, text=text)
    return f"已添加注释 #{aid} 到 {memory_id}"


def get_annotations(memory_id: str) -> str:
    """查看一条记忆的所有注释"""
    anns = mem.db.get_annotations(memory_id=memory_id)
    if not anns:
        return "没有注释"
    lines = []
    for a in anns:
        lines.append(f"[#{a['annotation_id']}] ({a['created_at']}) {a['text']}")
    return "\n".join(lines)


def consolidate(conversation_text: str) -> str:
    """对话结束后调用。把对话涉及的记忆被动连接起来，不花额外token。"""
    result = mem.consolidate(conversation_text=conversation_text)
    return f"匹配到{result['matched_memories']}条记忆, 建立{result['new_connections']}条新连接"



@mcp.tool()
def memory_edit(action: str, memory_id: str = "", text: str = "",
                id2: str = "", weight: float = 1.0,
                score: float = 0.5, tier: str = "", tag: str = "") -> str:
    """记忆编辑与巡检。action=annotate / connect / remove(进回收站7天) /
    set_emotion / set_tier / set_tag(agent 早巡检纠偏) / stats / neighbors / annotations。"""
    if action == "annotate":
        return annotate_memory(memory_id, text)
    if action == "connect":
        return connect_memories(memory_id, id2, weight)
    if action == "remove":
        return remove_memory(memory_id)
    if action == "set_emotion":
        return set_emotion(memory_id, score)
    if action == "set_tier":
        return set_tier(memory_id, tier)
    if action == "set_tag":
        return set_tag(memory_id, tag)
    if action == "stats":
        return graph_stats()
    if action == "neighbors":
        return get_neighbors(memory_id)
    if action == "annotations":
        return get_annotations(memory_id)
    return f"未知action: {action}（可用: annotate/connect/remove/set_emotion/set_tier/set_tag/stats/neighbors/annotations）"


def memory_admin(action: str, memory_id: str = "", text: str = "") -> str:
    """记忆库管理。action=stats(图统计) / consolidate(text=对话文本,对话结束后被动连接记忆) / neighbors(memory_id,看邻居) / annotations(memory_id,看注释)"""
    if action == "stats":
        return graph_stats()
    if action == "consolidate":
        return consolidate(text)
    if action == "neighbors":
        return get_neighbors(memory_id)
    if action == "annotations":
        return get_annotations(memory_id)
    return f"未知action: {action}（可用: stats/consolidate/neighbors/annotations）"


def belief_edit(action: str, belief_id: str = "", text: str = "",
                statement: str = "", kind: str = "propositional", origin: str = "",
                first_case_id: str = "", first_case_note: str = "",
                cognition_id: str = "") -> str:
    """内部 belief 编辑实现；公开 MCP 入口统一由 belief(action=...) 提供。"""
    if action == "get":
        return belief_get(belief_id)
    if action == "note":
        return belief_note(belief_id, text)
    if action == "add":
        return belief_add(statement, kind, origin, first_case_id, first_case_note)
    if action in {"promote", "demote"}:
        try:
            method = belief_mod.promote if action == "promote" else belief_mod.demote
            return json.dumps(method(mem.db, belief_id, text), ensure_ascii=False, indent=2)
        except (KeyError, ValueError) as exc:
            return f"未执行: {type(exc).__name__}: {exc}"
    if action in {"map_cognition", "unmap_cognition"}:
        data = belief_mod.load()
        belief = belief_mod.get_belief(data, belief_id)
        if not belief:
            return f"没有 {belief_id}"
        row = mem.db.get(cognition_id)
        if (not row or (row.get("level") or "raw") != "cognition"
                or (row.get("collection") or "") == "wenku"):
            return f"{cognition_id} 不是可映射的 cognition"
        ids = list(dict.fromkeys(str(x) for x in belief.get("cognition_ids", []) if x))
        if action == "map_cognition" and cognition_id not in ids:
            ids.append(cognition_id)
        if action == "unmap_cognition":
            ids = [mid for mid in ids if mid != cognition_id]
        belief["cognition_ids"] = ids
        belief_mod.save(data)
        return f"{belief_id} cognition_ids={','.join(ids) or '(空)'}"
    return f"未知action: {action}（可用: get/note/add/promote/demote/map_cognition/unmap_cognition）"


# ===== 内部REST API（供gateway调用，共享mem实例，省一份embedding模型）=====
from fastapi import FastAPI as _FastAPI
from fastapi.responses import JSONResponse as _JSONResponse

# ===== Belief Graph M1 (2026-06-05) =====

def belief_list() -> str:
    """看全部beliefs:id/状态/confidence(实时算)/case数。骨头的体检表。"""
    data = belief_mod.load()
    p = belief_mod.params(data)
    lines = []
    for b in data.get("beliefs", []):
        conf = belief_mod.confidence(mem.db, b, p)
        pin = "📌" if b.get("pinned") else "  "
        sup = len(b.get("support_cases", []))
        con = len(b.get("contradiction_cases", []))
        routed = b.get("status") == "active" and (b.get("pinned") or conf >= p["routing_cutoff"])
        lines.append(f"{pin}[{b['id']}] {b['status']:<9} conf={conf:.3f} +{sup}/-{con} "
                     f"{'路由中' if routed else '不路由'} | {b['statement'][:40]}")
    return "\n".join(lines) if lines else "beliefs.json 是空的。"


def belief_get(belief_id: str) -> str:
    """读一条belief的全部:statement/cases(带weight_note)/notes/tension。"""
    data = belief_mod.load()
    b = belief_mod.get_belief(data, belief_id)
    if not b:
        return f"没有 {belief_id}"
    p = belief_mod.params(data)
    conf = belief_mod.confidence(mem.db, b, p)
    out = [f"[{b['id']}] {b['statement']}",
           f"kind={b.get('kind')} status={b.get('status')} pinned={b.get('pinned')} conf={conf:.3f}",
           f"origin={b.get('origin')} tensions={b.get('tensions')}",
           f"cues={b.get('activation_cues')}",
           f"cognition_ids={b.get('cognition_ids', [])}"]
    if b.get("support_cases"):
        out.append("support:")
        for c in b["support_cases"]:
            ref = c.get("id") or f"inline:{c.get('inline_text','')[:48]}"
            out.append(f"  + [{ref}] {c.get('weight_note','')} ({c.get('added','')})")
    if b.get("contradiction_cases"):
        out.append("contradiction:")
        for c in b["contradiction_cases"]:
            ref = c.get("id") or f"inline:{c.get('inline_text','')[:48]}"
            out.append(f"  - [{ref}] {c.get('weight_note','')} ({c.get('added','')})")
    if b.get("boundary_cases"):
        out.append("boundary:")
        for c in b["boundary_cases"]:
            ref = c.get("id") or f"inline:{c.get('inline_text','')[:48]}"
            out.append(f"  ~ [{ref}] {c.get('weight_note','')} ({c.get('added','')})")
    for n in b.get("notes", []):
        out.append(f"  📝 {n}")
    return "\n".join(out)


def belief_add_case(belief_id: str, memory_id: str = "", weight_note: str = "",
                    kind: str = "support", inline_text: str = "",
                    occurred_at: str = "", emotion_score: float = 0.5) -> str:
    """给belief挂case。weight_note必填——写不出'这个case为什么支持/反对/限定这条belief'的连接就不该连。
    memory_id 与 inline_text 二选一；memory_id 可引用任意 Memory level，inline_text 只收最多280字短事件。
    kind: support | contradiction | boundary。
    support=支持; contradiction=反对(belief在被检验,不是坏事); boundary=适用边界,标记'这条belief在什么场景下不适用',只记录范围,不影响confidence。"""
    if not weight_note or not weight_note.strip():
        return "weight_note 必填。写出因果才算标注。"
    memory_id = (memory_id or "").strip()
    inline_text = (inline_text or "").strip()
    if bool(memory_id) == bool(inline_text):
        return "memory_id 与 inline_text 必须且只能填一个。"
    if memory_id and not mem.db.get(memory_id):
        return f"主库里没有 {memory_id},先确认id。"
    if len(inline_text) > 280:
        return "inline_text 最多 280 字；更长事件请先存入 Memory。"
    data = belief_mod.load()
    b = belief_mod.get_belief(data, belief_id)
    if not b:
        return f"没有 {belief_id}"
    key = {"support": "support_cases", "contradiction": "contradiction_cases",
           "boundary": "boundary_cases"}.get(kind)
    if key is None:
        return f"未知 kind: {kind}（可用 support/contradiction/boundary）。"
    if any((memory_id and c.get("id") == memory_id)
           or (inline_text and c.get("inline_text") == inline_text)
           for c in b.get(key, [])):
        return f"这个 case 已经挂在 {belief_id} 的 {kind} 上了。"
    import datetime as _dt
    case = {"weight_note": weight_note.strip(), "added": _dt.date.today().isoformat()}
    if memory_id:
        case["id"] = memory_id
    else:
        case.update({"inline_text": inline_text, "occurred_at": occurred_at or None,
                     "emotion_score": max(0.0, min(float(emotion_score), 1.0))})
    b.setdefault(key, []).append(case)
    b["updated_at"] = _dt.date.today().isoformat()
    belief_mod.save(data)
    p = belief_mod.params(data)
    conf = belief_mod.confidence(mem.db, b, p)
    if kind == "boundary":
        return f"已挂 boundary case → {belief_id}（标记适用边界,不影响 conf）。conf {conf:.3f}"
    hint = ""
    if b.get("status") == "candidate" and conf >= p["routing_cutoff"]:
        hint = f"\n⬆️ conf过了{p['routing_cutoff']},这条candidate够格转active了(用belief_note记录决定,手动改status——升格也要过你的手)。"
    if conf < p["dormant_hint"] and not b.get("pinned"):
        hint = f"\n⬇️ conf低于{p['dormant_hint']},考虑转dormant(不自动转,你决定)。"
    return f"已挂 {kind} case → {belief_id}。conf {conf:.3f}{hint}"


def belief_note(belief_id: str, text: str) -> str:
    """给belief加演变注释,只增不删。改status/pinned也先在这里记一笔为什么。"""
    data = belief_mod.load()
    b = belief_mod.get_belief(data, belief_id)
    if not b:
        return f"没有 {belief_id}"
    import datetime as _dt
    b.setdefault("notes", []).append(f"[{_dt.date.today().isoformat()}] {text.strip()}")
    b["updated_at"] = _dt.date.today().isoformat()
    belief_mod.save(data)
    return f"已记到 {belief_id}。"


def belief_add(statement: str, kind: str = "propositional", origin: str = "",
               first_case_id: str = "", first_case_note: str = "") -> str:
    """出生一条新belief(M3 belief出生管道)。人手调用——cluster只产提示,belief从这里出生。
    一律 status=candidate / pinned=False,id自动取下一个b-XXXX。
    first_case_id给了就校验主库存在,并连同first_case_note(必填)挂进support_cases。"""
    if not statement or not statement.strip():
        return "statement 必填。"
    import datetime as _dt
    data = belief_mod.load()
    beliefs = data.setdefault("beliefs", [])
    # 下一个 b-XXXX 序号
    max_n = 0
    for b in beliefs:
        bid = b.get("id", "")
        if bid.startswith("b-"):
            try:
                max_n = max(max_n, int(bid[2:]))
            except ValueError:
                pass
    new_id = f"b-{max_n + 1:04d}"
    support_cases = []
    if first_case_id:
        if not first_case_note or not first_case_note.strip():
            return "first_case_note 必填——给了first_case_id就得写出'它为什么支持这条belief'的因果,写不出别硬挂。"
        row = mem.db.get(first_case_id)
        if not row:
            return f"主库里没有 {first_case_id},先确认id。"
        support_cases.append({
            "id": first_case_id,
            "weight_note": first_case_note.strip(),
            "added": _dt.date.today().isoformat(),
        })
    today = _dt.date.today().isoformat()
    new_b = {
        "id": new_id,
        "statement": statement.strip(),
        "kind": kind,
        "pinned": False,
        "status": "candidate",
        "origin": origin.strip(),
        "activation_cues": [],
        "cognition_ids": [],
        "tensions": [],
        "support_cases": support_cases,
        "contradiction_cases": [],
        "notes": [],
        "created_at": today,
        "updated_at": today,
    }
    beliefs.append(new_b)
    belief_mod.save(data)
    cased = f"，已挂首个case [{first_case_id}]" if support_cases else ""
    return f"出生 {new_id}{cased}。从candidate起步，等case推上0.40(routing_cutoff)再手动转active——升格过你的手。"


@mcp.tool()
def graph_review(action: str, status: str = "pending", limit: int = 20,
                 memory_id: str = "", new_level: str = "",
                 source_id: str = "", target_id: str = "", reason: str = "",
                 proposal_id: str = "", decision: str = "", note: str = "") -> str:
    """统一 typed graph 审批入口。
    action=list(status+limit) / propose_level(memory_id+new_level+reason) /
    propose_evokes(source_id+target_id+reason) /
    decide(proposal_id+decision approve|reject+note可选)。
    提案不会直接落边；只有明确 decide approve 才会执行。"""
    if action == "list":
        return review_update_proposals(status=status, limit=limit)
    if action == "propose_level":
        return propose_level_change(memory_id=memory_id, new_level=new_level, reason=reason)
    if action == "propose_evokes":
        return propose_evokes(source_id=source_id, target_id=target_id, reason=reason)
    if action == "decide":
        return decide_update_proposal(
            proposal_id=proposal_id, decision=decision, note=note,
        )
    return f"未知action: {action}（可用: list/propose_level/propose_evokes/decide）"


@mcp.tool()
def belief(action: str, belief_id: str = "", text: str = "",
           statement: str = "", belief_kind: str = "propositional",
           origin: str = "", first_case_id: str = "", first_case_note: str = "",
           cognition_id: str = "", memory_id: str = "", weight_note: str = "",
           case_kind: str = "support", inline_text: str = "",
           occurred_at: str = "", emotion_score: float = 0.5) -> str:
    """统一 belief 入口。
    action=list / get / add / note / promote / demote / map_cognition /
    unmap_cognition / add_case。add_case 的 memory_id 与 inline_text 二选一。"""
    if action == "list":
        return belief_list()
    if action == "add_case":
        return belief_add_case(
            belief_id=belief_id, memory_id=memory_id, weight_note=weight_note,
            kind=case_kind, inline_text=inline_text, occurred_at=occurred_at,
            emotion_score=emotion_score,
        )
    return belief_edit(
        action=action, belief_id=belief_id, text=text, statement=statement,
        kind=belief_kind, origin=origin, first_case_id=first_case_id,
        first_case_note=first_case_note, cognition_id=cognition_id,
    )


from starlette.requests import Request as _Request
import threading as _threading
import time as _time
import uvicorn as _uvicorn


def _embedding_outbox_worker():
    # Keep writes independent from Voyage; backfill one vector at a quiet cadence.
    _time.sleep(300)
    while True:
        try:
            stats = mem.flush_embedding_outbox(limit=1)
            if stats.get("done"):
                print(f"[embedding-outbox] backfilled={stats['done']}", flush=True)
        except Exception as exc:
            print(f"[embedding-outbox] worker error: {type(exc).__name__}", flush=True)
        _time.sleep(300)


_embedding_outbox_thread = _threading.Thread(
    target=_embedding_outbox_worker,
    daemon=True,
    name="embedding-outbox",
)
_embedding_outbox_thread.start()

_rest = _FastAPI()

# 反射弧质量门槛(2026-06-07 设计A): 向量cosine距离≥此值的结果不进hook前N条, 留空。
# 实测: 真相关≤0.43, 水词/指令词/无关≥0.45。只调这一个数。不影响主动 search_memory。
REFLEX_QUALITY_MAX_DIST = 0.50
# 反射弧命中加热量(2026-06-07b): hook每轮高频, 单次小量, 靠每天0.6衰减+第4槽-2冷却平衡。主动search仍用0.3。
REFLEX_HEAT_BOOST = 0.2

_THESEUS_COLLECTION = None
_THESEUS_COLLECTION_LOCK = _threading.Lock()


def _get_theseus_collection():
    """Open the dedicated Theseus collection without ever creating a fallback."""
    global _THESEUS_COLLECTION
    if _THESEUS_COLLECTION is None:
        with _THESEUS_COLLECTION_LOCK:
            if _THESEUS_COLLECTION is None:
                _THESEUS_COLLECTION = mem._client.get_collection(
                    name=theseus_shadow_index.COLLECTION_NAME
                )
    return _THESEUS_COLLECTION


def _theseus_association_candidates_sync(query: str, n: int, max_distance: float):
    """Read-only candidate retrieval from independent Theseus shadows."""
    collection = _get_theseus_collection()
    db_file = os.path.join(DB_PATH, "memories.db")
    hits = theseus_shadow_index.search(
        db_path=db_file,
        collection=collection,
        query=query,
        embed_query=mem._embedder.encode_query,
        n_results=n,
        max_distance=max_distance,
        char_budget=1200,
    )
    out = []
    seen_parents = set()
    for hydrated in hits:
        hit = hydrated.get("hit") or {}
        parent_id = str(hit.get("parent_memory_id") or "")
        if not parent_id or parent_id in seen_parents:
            continue
        parent = mem.db.get(parent_id)
        if not parent or parent.get("collection") != "wenku":
            continue
        # Stale shadows must fail closed if the parent text has changed.
        if hit.get("source_hash") != theseus_shadow_index.source_hash(parent.get("text") or ""):
            continue
        seen_parents.add(parent_id)
        out.append({
            "memory_id": parent_id,
            "shadow_id": hydrated.get("shadow_id"),
            "chunk_no": hit.get("chunk_no"),
            "index_label": hit.get("index_label") or "",
            "insight_label": hit.get("insight_label") or "",
            "text": hydrated.get("primary_text") or hit.get("text") or "",
            "snippet": hydrated.get("primary_text") or hit.get("text") or "",
            "timestamp": parent.get("timestamp") or "",
            "tag": parent.get("tag") or "",
            "tier": parent.get("tier") or "",
            "level": parent.get("level") or "",
            "emotion_score": parent.get("emotion_score"),
            "distance": float(hydrated.get("distance", 9.0)),
            "source": "theseus",
            "evidence_role": "thought_snapshot",
        })
    return out


def _anchor_association_candidates_sync(
    anchor_id: str, n: int, max_distance: float
):
    """K4 shadow: outgoing EVOKES first; only an edgeless anchor may use semantic fallback."""
    anchor = mem.db.get(anchor_id)
    if not anchor or (anchor.get("collection") or "") == "wenku":
        return []
    with mem.db._conn() as conn:
        rows = conn.execute("""
            SELECT m.memory_id,m.text,m.timestamp,m.tag,m.tier,m.level,
                   m.emotion_score,e.weight
            FROM edges e JOIN memories m ON m.memory_id=e.target_id
            WHERE e.source_id=? AND e.edge_type='EVOKES'
              AND COALESCE(m.collection,'')='wenku'
            ORDER BY e.weight DESC,m.timestamp DESC LIMIT ?
        """, (anchor_id, n)).fetchall()
    if rows:
        return [{
            "memory_id": row["memory_id"], "text": row["text"] or "",
            "snippet": row["text"] or "", "timestamp": row["timestamp"] or "",
            "tag": row["tag"] or "", "tier": row["tier"] or "",
            "level": row["level"] or "raw", "emotion_score": row["emotion_score"],
            "edge_weight": float(row["weight"] or 0.0),
            "anchor_distance": None, "assoc_path": "evokes",
            "source": "theseus", "evidence_role": "thought_snapshot",
        } for row in rows]

    found = mem._collection.get(ids=[anchor_id], include=["embeddings"])
    embeddings = found.get("embeddings")
    if embeddings is None or len(embeddings) == 0:
        return []
    count = mem._collection.count()
    result = mem._collection.query(
        query_embeddings=[embeddings[0]],
        n_results=min(max(n * 3, 12), count),
        include=["documents", "metadatas", "distances"],
    )
    out = []
    ids = result.get("ids", [[]])[0]
    docs = result.get("documents", [[]])[0]
    metas = result.get("metadatas", [[]])[0]
    distances = result.get("distances", [[]])[0]
    for memory_id, text, meta, distance in zip(ids, docs, metas, distances):
        if (not memory_id or memory_id == anchor_id
                or (meta or {}).get("collection", "") == "wenku"
                or float(distance) >= max_distance):
            continue
        row = mem.db.get(memory_id)
        if not row or (row.get("collection") or "") == "wenku":
            continue
        out.append({
            "memory_id": memory_id, "text": row.get("text") or text or "",
            "snippet": row.get("text") or text or "",
            "timestamp": row.get("timestamp") or "", "tag": row.get("tag") or "",
            "tier": row.get("tier") or "", "level": row.get("level") or "raw",
            "emotion_score": row.get("emotion_score"),
            "anchor_distance": float(distance), "assoc_path": "anchor_semantic",
            "source": "anchor_memory", "evidence_role": "long_association",
        })
        if len(out) >= n:
            break
    return out


@_rest.get("/api/anchor_association_candidates")
async def _api_anchor_association_candidates(
    anchor_id: str,
    n: int = 8,
    max_distance: float = 0.45,
):
    """Read-only K4 candidate lane. It never heats, writes, or changes main recall."""
    n = max(1, min(int(n), 20))
    max_distance = max(0.0, min(float(max_distance), 2.0))
    try:
        results = await asyncio.to_thread(
            _anchor_association_candidates_sync, anchor_id, n, max_distance
        )
    except Exception as exc:
        print(f"[Anchor REST] edge association unavailable: {type(exc).__name__}", flush=True)
        results = []
    return _JSONResponse(content=results)

@_rest.get("/api/recall")
async def _api_recall(query: str, budget: int = 5, policy: str = "conversation",
                      allow_empty: bool = True, include_theseus: bool = True,
                      theseus_budget: int = -1, min_score: float = 0.0,
                      context: str = "", temporal_mode: str = ""):
    if os.environ.get("ANCHOR_RECALL_V2", "off").strip().lower() not in {"1","on","true","yes"}:
        return _JSONResponse(status_code=404, content={"error": "recall v2 disabled"})
    try:
        kwargs = {"budget": budget, "policy": policy, "allow_empty": allow_empty,
                  "include_theseus": include_theseus, "min_score": min_score,
                  "context": context, "temporal_mode": temporal_mode or None}
        if theseus_budget >= 0:
            kwargs["theseus_budget"] = theseus_budget
        result = await asyncio.to_thread(mem.recall, query, **kwargs)
        return _JSONResponse(content=result)
    except Exception as exc:
        print(f"[Anchor REST] recall v2 unavailable: {type(exc).__name__}", flush=True)
        return _JSONResponse(status_code=503, headers={"Retry-After":"2"},
                             content={"error":"recall temporarily unavailable"})


def _night_flow_repair_sync(node_limit: int = 80, edge_limit: int = 80) -> dict:
    """Single-owner weak-flow decay and bounded island repair."""
    from propose_links import propose_links
    node_limit = max(1, min(int(node_limit), 80))
    edge_limit = max(1, min(int(edge_limit), 80))
    with mem.db._conn() as conn:
        # 03:30 maintenance is the sole owner of machine-flow decay.
        decayed = 0
        rows = conn.execute(
            """SELECT m.memory_id FROM memories m
               LEFT JOIN (
                 SELECT memory_id,COUNT(DISTINCT neighbor) degree FROM (
                   SELECT source_id memory_id,target_id neighbor FROM flow_edges
                   UNION SELECT target_id memory_id,source_id neighbor FROM flow_edges
                 ) GROUP BY memory_id
               ) d ON d.memory_id=m.memory_id
               WHERE COALESCE(m.collection,'')!='wenku'
                 AND COALESCE(m.level,'raw') IN ('raw','understanding','cognition')
                 AND COALESCE(d.degree,0)<2
               ORDER BY COALESCE(d.degree,0),m.timestamp LIMIT ?""", (node_limit,)
        ).fetchall()
        conn.commit()
    created, pairs = 0, set()
    for row in rows:
        if created >= edge_limit:
            break
        try:
            proposals = propose_links([row["memory_id"]], relation_policy="flow", budget=2)
        except Exception:
            continue
        for proposal in proposals:
            pair = tuple(sorted((proposal["source_id"], proposal["target_id"])))
            if pair in pairs or created >= edge_limit:
                continue
            pairs.add(pair)
            for direction in proposal["directions"]:
                mem.db.write_flow_edge(
                    direction["source_id"], direction["target_id"],
                    direction["weight"], direction["conductance"],
                    "auto_night_repair", mode="auto",
                )
            created += 1
    activation_decayed = mem.db.decay_activation(factor=0.82)
    dual_edge.drain(mem.db)
    return {"ok": True, "nodes_scanned": len(rows), "pairs_created": created,
            "flow_rows_created": created * 2, "flow_rows_decayed": decayed,
            "activation_decayed": activation_decayed}


@_rest.post("/api/internal/night-flow-repair")
async def _api_night_flow_repair(request: _Request):
    host = request.client.host if request.client else ""
    if host not in {"127.0.0.1", "::1"}:
        return _JSONResponse(status_code=403, content={"error": "localhost only"})
    body = await request.json()
    result = await asyncio.to_thread(
        _night_flow_repair_sync, body.get("node_limit", 80), body.get("edge_limit", 80)
    )
    return _JSONResponse(content=result)

@_rest.get("/api/search")
async def _api_search(query: str, n: int = 5, tag: str = "", pure: bool = False, corpus: str = "exclude", gate: bool = True, activate: bool = False):
    tag_filter = tag if tag else None
    # 反射弧hook专用路径: 自动浮现不加热(2026-06-07), 只有AI agent主动search_memory才加热
    # corpus: exclude(default,daily/reflex top3) / only / all ; gate=False 关质量门槛(文库搜要全召回)
    trace_out = [] if (gate and recall_trace.enabled()) else None
    try:
        results = await asyncio.to_thread(
            mem.search,
            query=query, n_results=n, tag=tag_filter, hebbian=False,
            pure_semantic=pure, activate_on_hit=False, cite_on_hit=False,
            max_distance=(REFLEX_QUALITY_MAX_DIST if gate else None),
            activate_boost=REFLEX_HEAT_BOOST, corpus=corpus,
            trace_out=trace_out,
        )
    except Exception as exc:
        print(f"[Anchor REST] search unavailable: {type(exc).__name__}", flush=True)
        return _JSONResponse(
            status_code=503,
            headers={"Retry-After": "2"},
            content={"error": "search temporarily unavailable"},
        )
    if trace_out is not None:
        recall_trace.log_search(query, trace_out, kept=len(results), gate_max=REFLEX_QUALITY_MAX_DIST)
    return _JSONResponse(content=results)

@_rest.get("/api/theseus_association_candidates")
async def _api_theseus_association_candidates(
    query: str,
    n: int = 8,
    max_distance: float = 0.58,
):
    """Independent read-only association lane; empty is always a valid result."""
    n = max(1, min(int(n), 20))
    max_distance = max(0.0, min(float(max_distance), 2.0))
    try:
        results = await asyncio.to_thread(
            _theseus_association_candidates_sync, query, n, max_distance
        )
    except Exception as exc:
        print(
            f"[Anchor REST] Theseus association unavailable: {type(exc).__name__}",
            flush=True,
        )
        results = []
    return _JSONResponse(content=results)


@_rest.get("/api/route")
async def _api_route(query: str):
    """反射弧意图路由: 这句话值不值得翻记忆。depth=none/full, 失败兜底full。"""
    try:
        dec = await asyncio.to_thread(reflex_router.route, query, mem._embedder)
    except Exception as e:
        dec = {"depth": "full", "reason": f"route_err:{e}"}
    if recall_trace.enabled() and dec.get("depth") == "none":
        recall_trace.log("route", query, dec)
    return _JSONResponse(content=dec)

@_rest.get("/api/route/v2")
async def _api_route_v2(query: str, context: str = ""):
    """Router v2 diagnostic endpoint; gateway hot path imports the same pure module directly."""
    try:
        import json as _json
        aliases_path = os.path.join(DB_PATH, "aliases.json")
        with open(aliases_path, "r", encoding="utf-8") as f:
            raw_aliases = _json.load(f)
        generation = f"mtime:{os.path.getmtime(aliases_path):.6f}"
        alias_index = reflex_router_v2.build_alias_index(raw_aliases, generation=generation)
        dec = reflex_router_v2.route_reflex(query, alias_index, context=context)
    except Exception as exc:
        dec = {
            "schema": "reflex.route.v2",
            "policy_version": getattr(reflex_router_v2, "POLICY_VERSION", "unavailable"),
            "decision": "uncertain",
            "execution": "suppress",
            "reason_codes": [f"diagnostic_route_error:{type(exc).__name__}"],
            "lanes": {},
            "max_main": 0,
        }
    return _JSONResponse(content=dec)

@_rest.post("/api/shadow/upsert")
async def _api_shadow_upsert(request: _Request):
    """影子回填/增量: 收 {parent_id, version, keys:[{key,quote}]}, 用【常驻 embedder】嵌 key + 算 span + 写库。"""
    if getattr(mem, "_shadow_collection", None) is None:
        return _JSONResponse(content={"ok": False, "err": "shadow collection not init"})
    body = await request.json()
    pid = (body.get("parent_id") or "").strip()
    keys = body.get("keys") or []
    version = int(body.get("version", shadow_index.SHADOW_VERSION))
    row = mem.db.get(pid)
    if not row:
        return _JSONResponse(content={"ok": False, "err": "parent not found"})
    conn = mem.db._conn()
    try:
        n = shadow_index.upsert_shadows(
            mem._shadow_collection, conn, mem._embedder.encode,
            pid, row.get("text") or "", keys, version=version,
            parent_tag=row.get("tag", ""), parent_ts=row.get("timestamp", ""))
    finally:
        conn.close()
    return _JSONResponse(content={"ok": True, "parent_id": pid, "written": n})

@_rest.get("/api/shadow/probe")
async def _api_shadow_probe(query: str, n: int = 8):
    """§13 验收/调试: 拿 query 直接撞 shadow 层, 返回命中钥匙+距离+parent+span。"""
    if getattr(mem, "_shadow_collection", None) is None:
        return _JSONResponse(content={"ok": False, "err": "no shadow collection"})
    emb = await asyncio.to_thread(mem._encode_query, query)
    best = await asyncio.to_thread(
        shadow_index.shadow_search, mem._shadow_collection, emb, n
    )
    hits = [{"parent_id": pid, "dist": round(info["dist"], 4),
             "key": info["key"], "span": info["span"]}
            for pid, info in sorted(best.items(), key=lambda kv: kv[1]["dist"])]
    return _JSONResponse(content={"ok": True, "query": query, "hits": hits})

@_rest.post("/api/heat")
async def _api_heat(request: _Request):
    """Local idempotent confirmation that final memory IDs were actually injected."""
    host = request.client.host if request.client else ""
    if host not in {"127.0.0.1", "::1"}:
        return _JSONResponse(status_code=403, content={"error": "localhost only"})
    body = await request.json()
    ids = body.get("memory_ids") or []
    if isinstance(ids, str):
        ids = [part.strip() for part in ids.split(",") if part.strip()]
    event_id = str(body.get("event_id") or "").strip()
    if not event_id:
        return _JSONResponse(status_code=400, content={"error": "event_id required"})
    result = await asyncio.to_thread(
        mem.db.apply_heat, ids, float(body.get("boost", 0.12)), event_id,
        bool(body.get("spread", True)), 0.5, 3, 8, 64, 128,
        str(body.get("source") or "recall"),
    )
    return _JSONResponse(content={"ok": True, **result})


@_rest.post("/api/cool")
async def _api_cool(request: _Request):
    """浮现冷却: 反射弧给浮现过的记忆扣activation。"""
    body = await request.json()
    mid = str(body.get("memory_id", "")).strip()
    amount = float(body.get("amount", 2.0))
    if not mid:
        return _JSONResponse(content={"ok": False, "err": "missing memory_id"})
    new_score = mem.db.cool_activation(mid, amount=amount)
    return _JSONResponse(content={"ok": True, "memory_id": mid, "activation_score": new_score})

@_rest.post("/api/store")
async def _api_store(request: _Request):
    body = await request.json()
    mid = body.get("memory_id") or str(uuid.uuid4())[:8]  # 传id=覆盖式更新(快照用)
    text = body.get("text", "")
    tag, collection = _route_reading_note_to_wenku(
        text, body.get("tag", "general"), body.get("collection", "")
    )
    level = body.get("level", "raw")
    source_ref = _normalize_source_ref(body.get("source_ref", ""), level, collection)
    if body.get("evokes") and collection != "wenku":
        raise ValueError("evokes is only valid when storing a wenku entry")
    evoke_ids = _validate_evokes_ids(body.get("evokes", ""))
    mem.store(
        memory_id=mid,
        text=text,
        tag=tag,
        tier=body.get("tier", "short"),
        emotion_score=body.get("emotion_score", 0.5),
        level=level,
        collection=collection,
    )
    if body.get("context"):
        with mem.db._conn() as conn:
            conn.execute("UPDATE memories SET context = ? WHERE memory_id = ?",
                         (body["context"], mid))
            conn.commit()
    if source_ref:
        mem.db.annotate(mid, f"source_ref:{source_ref}")
    for source_id in evoke_ids:
        update_review.propose_evokes(source_id, mid, "REST store explicit evokes")
    mem.db.apply_heat(
        [mid], 0.60, f"rest-store:{mid}", spread=True, source="rest_store",
    )
    return {"memory_id": mid, "source_ref": source_ref or None,
            "evokes_proposed": len(evoke_ids)}

@_rest.get("/api/count")
async def _api_count():
    return {"count": mem.count()}


@_rest.get("/api/graph/health")
def _api_graph_health():
    """Read graph stats through the process that owns Kuzu's file lock."""
    with mem.db._conn() as conn:
        sqlite_nodes = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
        sqlite_edges = conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0]
        daily_ids = {
            row[0]
            for row in conn.execute(
                "SELECT memory_id FROM memories "
                "WHERE COALESCE(collection, '') != 'wenku'"
            )
        }

    if not mem.db.kuzu_available:
        return {
            "ok": False,
            "backend": "sqlite_fallback",
            "kuzu_available": False,
            "sqlite_nodes": sqlite_nodes,
            "sqlite_edges": sqlite_edges,
            "error": "Kuzu is unavailable in anchor-sse",
        }

    try:
        # Health is also the bounded reconciliation entrypoint. Drain through
        # the owning process; never mutate outbox cursors from an external job.
        mem.db._drain_kuzu_outbox()
        dual_edge.drain(mem.db)
        belief_graph.drain(mem.db)

        def query_kuzu(query: str):
            future = mem.db._kuzu_executor.submit(
                mem.db._consume_kuzu_rows,
                mem.db._kuzu_conn,
                query,
                {},
            )
            try:
                return future.result(timeout=3)
            except Exception:
                future.cancel()
                raise

        # Kuzu 0.11.3 can retain deleted relationship cardinality in its
        # aggregate count path. Enumerate visible rows so health reflects the
        # graph that traversal actually sees and still catches duplicate edges.
        node_rows = query_kuzu("MATCH (m:Memory) RETURN m.memory_id")
        edge_rows = query_kuzu(
            "MATCH (a:Memory)-[e:EDGE]->(b:Memory) "
            "RETURN a.memory_id, b.memory_id"
        )
        island_rows = query_kuzu(
            "MATCH (m:Memory) "
            "WHERE NOT EXISTS { MATCH (m)-[:EDGE]-(:Memory) } "
            "RETURN m.memory_id"
        )
        kuzu_nodes = len(node_rows)
        kuzu_edges = len(edge_rows)
        island_ids = [row[0] for row in island_rows]
        kuzu_beliefs = len(query_kuzu("MATCH (b:Belief) RETURN b.belief_id"))
        kuzu_belief_cases = len(query_kuzu("MATCH (c:BeliefCase) RETURN c.case_id"))
        kuzu_constellations = len(query_kuzu("MATCH ()-[e:CONSTELLATES]->() RETURN e.edge_id"))
        kuzu_case_references = len(query_kuzu("MATCH ()-[e:REFERENCES]->() RETURN e.edge_id"))
        kuzu_case_relations = sum(
            len(query_kuzu(f"MATCH ()-[e:{role}]->() RETURN e.edge_id"))
            for role in ("SUPPORTS", "CONTRADICTS", "BOUNDS")
        )
        kuzu_belief_outgoing = len(query_kuzu("MATCH (b:Belief)-[e]->() RETURN b.belief_id"))

        # Every Kuzu read drains the SQLite outbox first. Read the mirror counters
        # afterwards so this snapshot also exposes any writes still waiting.
        with mem.db._conn() as conn:
            sqlite_nodes = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
            sqlite_edges = conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0]
            node_outbox = conn.execute(
                "SELECT COUNT(*) FROM kuzu_node_outbox"
            ).fetchone()[0]
            edge_outbox = conn.execute(
                "SELECT COUNT(*) FROM kuzu_edge_outbox"
            ).fetchone()[0]
            embedding_outbox = conn.execute(
                "SELECT COUNT(*) FROM embedding_outbox"
            ).fetchone()[0]
            belief_counts = belief_graph.counts(mem.db)

        consistent = (
            kuzu_nodes == sqlite_nodes
            and kuzu_edges == sqlite_edges
            and node_outbox == 0
            and edge_outbox == 0
            and embedding_outbox == 0
            and kuzu_beliefs == belief_counts["beliefs"]
            and kuzu_belief_cases == belief_counts["cases"]
            and kuzu_constellations == belief_counts["constellations"]
            and kuzu_case_references == belief_counts["memory_cases"]
            and kuzu_case_relations == belief_counts["cases"]
            and kuzu_belief_outgoing == 0
            and belief_counts["outbox"] == 0
        )
        legacy_cross_layer = mem.db.legacy_cross_layer_report()
        return {
            "ok": consistent,
            "status": "healthy" if consistent else "degraded",
            "backend": "kuzu",
            "kuzu_available": True,
            "nodes": kuzu_nodes,
            "edges": kuzu_edges,
            "islands": sum(1 for memory_id in island_ids if memory_id in daily_ids),
            "islands_all": len(island_ids),
            "sqlite_nodes": sqlite_nodes,
            "sqlite_edges": sqlite_edges,
            "node_outbox": node_outbox,
            "edge_outbox": edge_outbox,
            "embedding_outbox": embedding_outbox,
            "belief_graph": {
                **belief_counts,
                "kuzu_beliefs": kuzu_beliefs,
                "kuzu_cases": kuzu_belief_cases,
                "kuzu_constellations": kuzu_constellations,
                "kuzu_case_references": kuzu_case_references,
                "kuzu_case_relations": kuzu_case_relations,
                "kuzu_belief_outgoing": kuzu_belief_outgoing,
            },
            "legacy_cross_layer": legacy_cross_layer,
            "typed_graph_enabled": os.environ.get("ANCHOR_TYPED_GRAPH", "on").strip().lower() not in {"0", "off", "false", "no"},
            "source_ref_enabled": os.environ.get("ANCHOR_SOURCE_REF", "on").strip().lower() not in {"0", "off", "false", "no"},
            "consistent": consistent,
        }
    except Exception as exc:
        return {
            "ok": False,
            "backend": "sqlite_fallback",
            "kuzu_available": True,
            "sqlite_nodes": sqlite_nodes,
            "sqlite_edges": sqlite_edges,
            "error": f"{type(exc).__name__}: {exc}",
        }


@_rest.post("/api/consolidate")
async def _api_consolidate(request: _Request):
    """已退休兼容端点：无 Hebbian、无连边；固定返回 disabled。"""
    # 2026-06-14 永久关闭: 边爆炸根因, 机制废弃. 直接返回, 不再连边.
    return {"disabled": True, "reason": "passive-hebbian retired 2026-06-14"}
    body = await request.json()
    text = (body.get("text") or "").strip()
    if len(text) < 50:
        return {"error": "text too short"}
    return mem.consolidate(conversation_text=text[:20000])

@_rest.get("/api/wenku_index")
async def _api_wenku_index(type: str = ""):
    """文库目录(TOC): 只返回 标题+id+type+日期, 不返回正文(极省 token)。
    type: 限定类别, 空=全部。按时间升序(建船顺序)。"""
    tag_filter = type if type else None
    rows = mem.db.list_collection(collection="wenku", tag=tag_filter)
    out = []
    for r in rows:
        txt = (r.get("text") or "").lstrip()
        if txt.startswith("《") and "》" in txt:
            title = txt[1:txt.index("》")]
        else:
            first = txt.split(chr(10), 1)[0]
            title = first[:24] + ("…" if len(first) > 24 else "")
        title = title[:40]
        full_tag = r.get("tag") or ""
        t = full_tag.split(",")[0] if full_tag else ""
        date = (r.get("timestamp") or "")[:10]
        out.append({"memory_id": r.get("memory_id", ""), "type": t,
                    "title": title, "date": date})
    return _JSONResponse(content=out)

@_rest.get("/api/wenku_get")
async def _api_wenku_get(id: str = ""):
    """读文库某条全文(corpus 边界: 只返回 collection=wenku 的条目, 不越界到日常)。"""
    if not id:
        return _JSONResponse(content={"ok": False, "err": "missing id"})
    row = mem.db.get(id)
    if not row or (row.get("collection") or "") != "wenku":
        return _JSONResponse(content={"ok": False, "err": "not a wenku entry"})
    full = row.get("context") or row.get("text") or ""
    return _JSONResponse(content={"ok": True, "memory_id": id,
                                  "tag": row.get("tag", ""),
                                  "timestamp": row.get("timestamp", ""),
                                  "text": full})


# ── Dream Events REST API (心跳系统感知层) ──

@_rest.get("/api/dream/events")
async def _api_dream_event(type: str = "", value: str = ""):
    """iOS快捷指令上报事件"""
    if not type:
        return {"ok": False, "error": "missing type"}
    if not value:
        value = type
    ok = mem.db.insert_dream_event(type, value)
    return {"ok": ok, "dedup": not ok}

@_rest.get("/api/dream/recent")
async def _api_dream_recent(hours: int = 6, limit: int = 20):
    """获取最近的dream events"""
    events = mem.db.get_recent_dream_events(hours=hours, limit=limit)
    return _JSONResponse(content=events)

# ── Keepalive Messages REST API (意识连续性) ──

@_rest.get("/api/keepalive/pending")
async def _api_keepalive_pending():
    """获取未认领的keepalive消息"""
    pending = mem.db.get_pending_keepalive()
    return _JSONResponse(content=pending)

@_rest.post("/api/keepalive/store")
async def _api_keepalive_store(request: _Request):
    """存储keepalive消息"""
    body = await request.json()
    kid = mem.db.insert_keepalive_message(
        action_type=body.get("action_type", "none"),
        thoughts=body.get("thoughts", ""),
        content=body.get("content", ""),
    )
    return {"ok": True, "id": kid}

@_rest.post("/api/keepalive/consume")
async def _api_keepalive_consume():
    """认领所有pending消息"""
    count = mem.db.consume_keepalive()
    return {"ok": True, "consumed": count}




@_rest.get("/api/recent_raw")
async def _api_recent_raw(hours: int = 48):
    """获取最近N小时的raw层记忆碎片"""
    fragments = mem.db.get_recent_raw(hours=hours, level="raw")
    return fragments


@_rest.get("/api/recent")
async def _api_recent(n: int = 30, exclude_tag: str = "", random: bool = False):
    """拉记忆素材池。exclude_tag: 逗号分隔的tag黑名单; random=true时从全库随机采样"""
    with mem.db._conn() as conn:
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        order_clause = "ORDER BY RANDOM()" if random else "ORDER BY timestamp DESC"
        if exclude_tag:
            # 拉多一些再过滤
            cur.execute(f"""
                SELECT memory_id, text, tag, tier, emotion_score, timestamp
                FROM memories
                {order_clause}
                LIMIT ?
            """, (n * 3,))
            rows = cur.fetchall()
            exclude_tags = set(t.strip() for t in exclude_tag.split(","))
            filtered = []
            for r in rows:
                row_tags = set(t.strip() for t in (r["tag"] or "").split(","))
                if not row_tags & exclude_tags:
                    filtered.append(dict(r))
                if len(filtered) >= n:
                    break
            return _JSONResponse(content=filtered)
        else:
            cur.execute(f"""
                SELECT memory_id, text, tag, tier, emotion_score, timestamp
                FROM memories
                {order_clause}
                LIMIT ?
            """, (n,))
            rows = cur.fetchall()
            return _JSONResponse(content=[dict(r) for r in rows])


@_rest.get("/api/by_level")
async def _api_by_level(level: str = "cognition"):
    """获取指定level的所有记忆"""
    results = mem.db.get_by_level(level=level)
    return _JSONResponse(content=results)

@_rest.get("/api/neighbors")
async def _api_neighbors(memory_id: str):
    """获取一条记忆的邻居（含level信息）"""
    neighbors = mem.db.get_neighbors(memory_id=memory_id, min_weight=0.1, limit=20)
    result = []
    for n in neighbors:
        nid = n.get("memory_id", "")
        m = mem.db.get(nid)
        if m:
            result.append({
                "memory_id": nid,
                "text": m.get("text", ""),
                "level": m.get("level", "raw"),
                "weight": n.get("weight", 0),
            })
    return result


@_rest.get("/api/hot")
async def _api_hot(n: int = 5, threshold: float = 2.0, exclude: str = ""):
    """按 activation_score 降序拉热记忆(同分随机). exclude: 逗号分隔 memory_id 黑名单"""
    exclude_ids = set(x.strip() for x in exclude.split(",") if x.strip())
    rows = mem.db.get_hot(n=n, threshold=threshold, exclude_ids=exclude_ids)
    return _JSONResponse(content=rows)


@_rest.get("/api/hot_neighbors")
async def _api_hot_neighbors(seeds: str = "", exclude: str = "",
                             threshold: float = 2.0, n: int = 5):
    """当前话题种子(seeds, 逗号分隔)的图邻居里最热的记忆, 自带桥(bridge)"""
    seed_ids = [x.strip() for x in seeds.split(",") if x.strip()]
    exclude_ids = set(x.strip() for x in exclude.split(",") if x.strip())
    rows = mem.db.get_hot_neighbors(seed_ids, exclude_ids=exclude_ids,
                                    threshold=threshold, n=n)
    return _JSONResponse(content=rows)


@_rest.post("/api/connect")
async def _api_connect(request: _Request):
    """连接两条记忆"""
    body = await request.json()
    id1 = body.get("id1", "")
    id2 = body.get("id2", "")
    weight = body.get("weight", 1.0)
    if id1 and id2:
        mem.db.connect(id1, id2, weight=weight)
        return {"ok": True}
    return {"ok": False, "error": "missing id1 or id2"}


@_rest.post("/api/remove")
async def _api_remove(request: _Request):
    """删除记忆（软删除，进回收站）"""
    body = await request.json()
    memory_id = body.get("memory_id", "")
    deleted_by = body.get("deleted_by", "http_api")
    if memory_id:
        ok = mem.delete(memory_id=memory_id, deleted_by=deleted_by)
        return {"ok": ok, "memory_id": memory_id}
    return {"ok": False, "error": "missing memory_id"}


@_rest.get("/api/trash")
async def _api_trash(request: _Request):
    """列出回收站"""
    limit = int(request.query_params.get("limit", "50"))
    items = mem.db.list_trash(limit=limit)
    return items


@_rest.post("/api/restore")
async def _api_restore(request: _Request):
    """从回收站恢复记忆（sqlite + chromadb重新embed）"""
    body = await request.json()
    memory_id = body.get("memory_id", "")
    if not memory_id:
        return {"ok": False, "error": "missing memory_id"}
    # 1. sqlite侧恢复
    ok = mem.db.restore_from_trash(memory_id)
    if not ok:
        return {"ok": False, "error": "not found in trash"}
    # 2. 从sqlite读回数据，通过mem.store重新embed到chromadb
    try:
        with mem.db._conn() as conn:
            row = conn.execute(
                "SELECT memory_id, text, context, timestamp, tag, tier, emotion_score, level "
                "FROM memories WHERE memory_id=?",
                (memory_id,)
            ).fetchone()
        if row:
            # 用store的embed逻辑重建chromadb记录

            embedding = mem._encode_document(row["text"])
            meta = {
                "level": row["level"] or "raw",
                "timestamp": row["timestamp"],
                "tag": row["tag"] or "",
                "tier": row["tier"] or "short",
                "memory_id": row["memory_id"],
            }
            mem._collection.upsert(
                ids=[row["memory_id"]],
                embeddings=[embedding],
                documents=[row["text"]],
                metadatas=[meta],
            )
    except Exception as e:
        return {"ok": True, "memory_id": memory_id, "warning": f"chromadb re-embed failed: {e}"}
    return {"ok": True, "memory_id": memory_id}


@_rest.get("/api/calendar_density")
async def _api_calendar_density(start: str = "2026-01-01", end: str = "2026-12-31"):
    """日历热力图：返回日期范围内每天的记忆数量"""
    results = mem.db.calendar_density(start=start, end=end)
    return _JSONResponse(content=results)

@_rest.get("/api/by_date")
async def _api_by_date(date: str = ""):
    """按 JST 日期查询所有记忆。"""
    if not date:
        date = (datetime.utcnow() + timedelta(hours=9)).strftime("%Y-%m-%d")
    results = mem.db.daily_recap(date_str=date)
    return _JSONResponse(content=results)


# ── Belief Touch REST API (M2 反射弧:给hook实时点亮骨头) ──

import numpy as _np

_belief_emb_cache = {"mtime": None, "ids": [], "stmts": [], "mat": None}
_belief_emb_lock = _threading.Lock()

def _belief_emb_table_unlocked():
    """Encode only beliefs that pass the active/confidence routing contract."""
    mtime = os.path.getmtime(belief_mod.BELIEF_PATH)
    if _belief_emb_cache["mtime"] == mtime and _belief_emb_cache["mat"] is not None:
        return _belief_emb_cache
    ids, stmts = [], []
    for b, _conf in belief_mod.routing_set(mem.db):
        if b.get("statement"):
            ids.append(b["id"]); stmts.append(b["statement"])
    if stmts:
        vecs = _np.asarray(mem._embedder.encode(stmts), dtype="float32")
        norms = _np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        mat = vecs / norms
    else:
        mat = None
    _belief_emb_cache.update({"mtime": mtime, "ids": ids, "stmts": stmts, "mat": mat})
    return _belief_emb_cache


def _belief_emb_table():
    with _belief_emb_lock:
        return _belief_emb_table_unlocked()


def _most_relevant_belief_case(belief: dict, query_vector) -> dict | None:
    rows = []
    for kind, key in (("support", "support_cases"),
                      ("contradiction", "contradiction_cases"),
                      ("boundary", "boundary_cases")):
        for case in belief.get(key, []) or []:
            memory_id = case.get("id") or ""
            memory = mem.db.get(memory_id) if memory_id else None
            text = (memory or {}).get("text") or case.get("inline_text") or ""
            if text.strip():
                rows.append((kind, case, text.strip()))
    if not rows:
        return None
    vecs = _np.array([
        _np.asarray(mem._embedder.encode(text), dtype="float32")
        for _kind, _case, text in rows
    ])
    norms = _np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    sims = (vecs / norms) @ query_vector
    idx = int(_np.argmax(sims))
    kind, case, text = rows[idx]
    return {
        "case_id": case.get("case_id"), "kind": kind,
        "memory_id": case.get("id") or None,
        "inline": not bool(case.get("id")),
        "text": text[:180], "weight_note": (case.get("weight_note") or "")[:160],
        "similarity": round(float(sims[idx]), 4),
    }


def _warm_belief_embeddings():
    """Fill document-side Voyage caches before the belief REST endpoint opens."""
    tbl = _belief_emb_table()
    data = belief_mod.load()
    texts = []
    for belief_id in tbl["ids"]:
        belief = belief_mod.get_belief(data, belief_id)
        if not belief:
            continue
        for key in ("support_cases", "contradiction_cases", "boundary_cases"):
            for case in belief.get(key, []) or []:
                memory_id = case.get("id") or ""
                memory = mem.db.get(memory_id) if memory_id else None
                text = (memory or {}).get("text") or case.get("inline_text") or ""
                if text.strip():
                    texts.append(text.strip())
    case_texts = list(dict.fromkeys(texts))
    if case_texts:
        mem._embedder.encode(case_texts)
    print(f"[Belief Touch] embedding warmup: beliefs={len(tbl['ids'])} cases={len(case_texts)}", flush=True)


@_rest.get("/api/belief/touch")
async def _api_belief_touch(query: str = "", exclude: str = "", debug: str = ""):
    """给query找最相似的active belief,余弦>=0.45才算命中,实时算confidence。任何异常都返回hit:false。"""
    try:
        if not query.strip():
            return {"hit": False}
        tbl = await asyncio.to_thread(_belief_emb_table)
        if tbl["mat"] is None:
            return {"hit": False}
        q = _np.asarray(await asyncio.to_thread(mem._encode_query, query), dtype="float32")
        qn = _np.linalg.norm(q)
        if qn == 0:
            return {"hit": False}
        q = q / qn
        sims = tbl["mat"] @ q
        exclude_ids = set(x.strip() for x in (exclude or "").split(",") if x.strip())
        debug_on = str(debug or "").strip().lower() in {"1", "true", "yes", "on"}
        ordered = [int(value) for value in _np.argsort(sims)[::-1]]
        diagnostics = {
            "threshold": 0.45,
            "top": [
                {"id": tbl["ids"][cand_idx],
                 "similarity": round(float(sims[cand_idx]), 4),
                 "excluded": tbl["ids"][cand_idx] in exclude_ids}
                for cand_idx in ordered[:3]
            ],
            "excluded_ids": sorted(exclude_ids),
        } if debug_on else None
        idx = None
        for cand_idx in ordered:
            if float(sims[cand_idx]) < 0.45:
                break
            if tbl["ids"][cand_idx] in exclude_ids:
                continue
            idx = cand_idx
            break
        if idx is None:
            result = {"hit": False}
            if diagnostics is not None:
                result["diagnostics"] = diagnostics
            return result
        data = belief_mod.load()
        b = belief_mod.get_belief(data, tbl["ids"][idx])
        if not b:
            return {"hit": False}
        # v2 (2026-06-19): 记录命中日期,供 render_brief 判定"最近活跃"。按天去重,写失败不挡命中返回。
        try:
            import datetime as _dt
            _today = _dt.date.today().isoformat()
            if b.get("last_touched") != _today:
                b["last_touched"] = _today
                belief_mod.save(data)
        except Exception:
            pass
        p = belief_mod.params(data)
        conf = round(belief_mod.confidence(mem.db, b, p), 3)
        case = await asyncio.to_thread(_most_relevant_belief_case, b, q)
        result = {"hit": True, "id": b["id"], "statement": b["statement"],
                  "confidence": conf, "case": case}
        if diagnostics is not None:
            diagnostics.update({
                "support_case_count": len(b.get("support_cases", [])),
                "contradiction_case_count": len(b.get("contradiction_cases", [])),
                "cases_followed": bool(case),
                "selected_case_id": (case or {}).get("case_id"),
                "selected_case_kind": (case or {}).get("kind"),
            })
            result["diagnostics"] = diagnostics
        return result
    except Exception as e:
        result = {"hit": False}
        if str(debug or "").strip().lower() in {"1", "true", "yes", "on"}:
            result["diagnostics"] = {"error": type(e).__name__, "threshold": 0.45}
        return result



def _run_rest():
    host = os.environ.get("ANCHOR_REST_HOST", "127.0.0.1")
    port = int(os.environ.get("ANCHOR_REST_PORT", "8765"))
    _uvicorn.run(_rest, host=host, port=port, log_level="warning")

# ===== Streamable HTTP transport =====
def _run_streamable():
    """Run the configurable Streamable HTTP transport."""
    streamable_app = mcp.streamable_http_app()
    host = os.environ.get("ANCHOR_STREAMABLE_HOST", "127.0.0.1")
    port = int(os.environ.get("ANCHOR_STREAMABLE_PORT", "8768"))
    _uvicorn.run(streamable_app, host=host, port=port, log_level="info")


def start_servers():
    """Start REST and Streamable HTTP only when explicitly invoked."""
    try:
        _warm_belief_embeddings()
    except Exception as warm_error:
        print(f"[Belief Touch] embedding warmup failed: {type(warm_error).__name__}", flush=True)
    rest_thread = _threading.Thread(target=_run_rest, daemon=True)
    rest_thread.start()
    stream_thread = _threading.Thread(target=_run_streamable, daemon=True)
    stream_thread.start()
    rest_host = os.environ.get("ANCHOR_REST_HOST", "127.0.0.1")
    rest_port = int(os.environ.get("ANCHOR_REST_PORT", "8765"))
    stream_host = os.environ.get("ANCHOR_STREAMABLE_HOST", "127.0.0.1")
    stream_port = int(os.environ.get("ANCHOR_STREAMABLE_PORT", "8768"))
    print(f"[Anchor] REST on {rest_host}:{rest_port}")
    print(f"[Anchor] Streamable HTTP on {stream_host}:{stream_port}")
    return rest_thread, stream_thread


if __name__ == "__main__":
    import time
    start_servers()
    while True:
        time.sleep(3600)
