"""
反射弧意图路由 (reflex router) — "这句话值不值得翻记忆?"

原型向量路由: query嵌入 vs 少量意图原型, 跑常驻 BGE-zh(mem._embedder), 近零延迟。
depth: none(别翻记忆) / full(全召回)。
保守原则: 只杀很干净的即时句, 一切模糊倒向 full。
        误伤一次真召回  >>  漏抑制一次水召回。

Initial production calibration(BAAI/bge-base-zh-v1.5)。校准表见文末。
调参入口: _TH / _MARGIN / _PROTOS / _CUE。改 _PROTOS 后调 reload()。
一键旁路: _ENABLED=False → 一律 full(=今天行为)。
"""
import re
import unicodedata
import numpy as np

_ENABLED = True   # False = 路由完全旁路, 一律 full

_PROTOS = {
    # 即时身体/情绪状态、问候、即时指令、元对话: 当下回应, 别翻旧账
    "none": [
        "我现在有点不舒服", "我好累啊", "我需要休息一下", "我有点紧张",
        "在吗", "早上好", "晚安", "我回来了",
        "帮我跑一下这个", "把这个改一下", "你刚才说什么", "嗯嗯好的",
    ],
    # 回忆、互动历史、专有名词、事实查询: 该翻记忆
    "full": [
        "你还记得我们上次说的吗", "你记得那件事吗", "我们当时是怎么决定的",
        "关于AI agent的系统架构", "记忆档案里是怎么写的", "那个人是谁来着",
    ],
}

# 显式回忆/事实线索词: 命中直接放行 full, 不看向量分。
# 处理"当前状态和上次一样吗"这种"即时句壳+回忆芯"复合句。
# 此正则宁可多命中(→full 是安全方向), 不可漏。
_CUE = re.compile(
    r"还记得|记得|上次|上回|之前|那次|当时|那天|你说过|聊到哪|"
    r"怎么决定|怎么修|怎么弄|为什么|是什么|是多少|在哪"
)

_TH = 0.70       # query 与 none 原型最大 cos ≥ 此值 才考虑抑制(实机: 干净即时句 0.72~0.97)
_MARGIN = 0.10   # 且 none 分 要比 full 分 高出这么多, 才真抑制

# cheap guard: 纯 sticker/省略号/标点符号/短拟声词, 不值得走向量检索。
# 放在向量路由前, 但低于 _CUE: 有"上次/那次/记得"等回忆线索仍 full 放行。
_STICKER_RE = re.compile(r"^\s*\(?\s*sticker\b.*\)?\s*$", re.I)
_SOFT_NOISE = {
    "汪", "汪汪", "汪汪汪", "喵", "喵喵",
    "呜", "呜呜", "呜呜呜", "呜哇", "呜哇呜哇", "嘤", "嘤嘤", "嘤嘤嘤",
    "哼", "哼哼", "哼哼哼", "哼唧", "哼唧哼唧",
    "啊", "啊啊", "啊啊啊", "呀", "呀呀", "欸", "诶", "诶嘿", "嘿嘿",
    "嗯", "嗯嗯", "嗯哼", "好耶", "收到", "明白", "好的",
}
_ADDRESS_RE = re.compile(r"^(伙伴|朋友|老师|助手|AI agent)+")


def _semantic_chars(text: str) -> str:
    """只保留字母/数字/汉字等语义字符；emoji、标点、空白全丢。"""
    out = []
    for ch in text or "":
        cat = unicodedata.category(ch)
        if cat[0] in ("L", "N") or "\u4e00" <= ch <= "\u9fff":
            out.append(ch)
    return "".join(out)


def _cheap_none(query: str) -> str:
    q = (query or "").strip()
    if not q:
        return "空输入·不翻记忆"
    if _STICKER_RE.match(q):
        return "sticker·不翻记忆"
    sem = _semantic_chars(q)
    if not sem:
        return "纯标点/表情·不翻记忆"
    # 很短的拟声词, 允许前面带一个称呼；避免把正经短问句误杀。
    soft = _ADDRESS_RE.sub("", sem)
    if len(soft) <= 8 and soft in _SOFT_NOISE:
        return "短拟声词·不翻记忆"
    return ""

_proto_vecs = None


def _norm(v):
    v = np.asarray(v, dtype="float32")
    return v / (np.linalg.norm(v) + 1e-9)


def _ensure(embedder):
    global _proto_vecs
    if _proto_vecs is not None:
        return
    _proto_vecs = {k: np.stack([_norm(embedder.encode(s)) for s in sents])
                   for k, sents in _PROTOS.items()}


def reload():
    """改了 _PROTOS 后调一次, 下次 route 重新嵌入原型。"""
    global _proto_vecs
    _proto_vecs = None


def route(query: str, embedder) -> dict:
    """返回 {depth:'none'|'full', none, full, reason}。embedder = mem._embedder。
    调用方须把任何异常兜底成 full —— 路由绝不该吃掉召回。"""
    if not _ENABLED:
        return {"depth": "full", "reason": "router_disabled"}
    if _CUE.search(query or ""):
        return {"depth": "full", "none": None, "full": None, "reason": "回忆线索词放行"}
    cheap_reason = _cheap_none(query)
    if cheap_reason:
        return {"depth": "none", "none": None, "full": None, "reason": cheap_reason}
    _ensure(embedder)
    q = _norm(embedder.encode(query))
    none_s = float(np.max(_proto_vecs["none"] @ q))
    full_s = float(np.max(_proto_vecs["full"] @ q))
    if none_s >= _TH and (none_s - full_s) >= _MARGIN:
        return {"depth": "none", "none": round(none_s, 3),
                "full": round(full_s, 3), "reason": "即时表达·不翻记忆"}
    return {"depth": "full", "none": round(none_s, 3),
            "full": round(full_s, 3), "reason": "pass"}
