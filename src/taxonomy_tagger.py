"""五轴taxonomy自动打标器 · 2026-07-08
新记忆存入时后台线程调DeepSeek打tag，写回SQLite覆盖tag字段。
- tag是纯机器字段：AI agent存记忆不打tag，检索走自然语言正文，坐标给reranker当卷宗。
- 失败静默：tag保持原样，维护班/夜批按 tag NOT LIKE 'state:%' 抓漏网重打。
- key解析：优先 model_routes(root)，可选读取显式 ANCHOR_TAXONOMY_ENV_FILE；默认不读取秘密文件。
"""
import datetime
import json
import os
import sqlite3
import sys

import httpx
from release_config import AnchorConfig

_CONFIG = AnchorConfig.load()
DB = str(_CONFIG.db_path)
LOCAL_USER_SECRETS = os.environ.get("ANCHOR_TAXONOMY_ENV_FILE", "")

VALID = set()
for _ax, _vals in {
    "state": ["current", "past", "stable", "obsolete"],
    "domain": ["relationship", "health", "work", "family", "system", "creative",
               "daily", "reading", "social", "meta"],
    "action": ["todo", "waiting", "ongoing", "done", "watch"],
    "kind": ["event", "fact", "pattern", "preference", "idea", "boundary", "milestone"],
    "heat": ["low", "mid", "high", "core"]}.items():
    for _v in _vals:
        VALID.add(f"{_ax}:{_v}")
# obsolete 只允许复审流程写入；打标器prompt里不给这个选项，校验里留着以便复审复用 validate()

PROMPT = """你是记忆库的打标器。给每条记忆打标签，只用下面这套体系。
今天是 {TODAY}。每条记忆自带写入日期，判断时效以这两个日期为参照。

1. 时效类型 state（只判类型，不判过期；过期由另一个复审流程负责）:
- state:current 描述的状态/进行中的事此刻大概率仍有效（如"正在等待虚构观测站校准"）
- state:past 单次已发生的事件，过去时（如"6月1号虚构探测器完成试飞"）
- state:stable 不随时间变化的长期事实：身份、性格、关系、偏好、习惯（如"图书管理员偏爱蓝色索引卡"）
判据：讲"一件事"→past；讲"一个还在持续的状态"→current；讲"一个人/关系是什么样"→stable。

2. 内容域 domain:
- domain:relationship 只用于 agent 与长期对话对象之间的关系/亲密/冲突/和好；外部人际优先 social
- domain:health 身体/睡眠/吃饭/精神状态
- domain:work 工作/职业/赚钱/曾经的职业经历
- domain:family 家人/父母/亲戚
- domain:system 系统/记忆库/服务器/前端/部署
- domain:creative 画画/UI/设计/写作灵感
- domain:daily 日常碎片/生活细节/个人经历与被他人如何看待
- domain:reading 读书/影视/出海
- domain:social 聊天室/笔友/别的AI/外部人际
- domain:meta 自我、本体论、belief、记忆哲学

3. 行动状态 action:
- action:todo 还要做
- action:waiting 等待外部条件
- action:ongoing 持续进行
- action:done 已完成
- action:watch 以后留意

4. 记忆性质 kind:
- kind:event 单次事件
- kind:fact 稳定事实
- kind:pattern 反复模式
- kind:preference 偏好
- kind:idea 灵感
- kind:boundary 边界/雷区
- kind:milestone 重大节点

5. 情绪强度 heat:
- heat:low
- heat:mid
- heat:high
- heat:core

规则：
- 每条记忆输出 3 到 5 个 tag。
- 必须包含一个 state:*。
- 必须包含至少一个 domain:*。
- 必须包含一个 kind:*。
- action:* 只有在这条记忆对未来有动作意义时才加。
- heat:* 只有情绪或重要性明显时才加。
- 不判断人称，不打任何人物标签。
- 不要发明体系外标签。
- 不要解释。
- 输出 JSONL，每行格式：
{"id":"原id","tags":["state:...","domain:...","kind:..."]}"""


def _routes():
    """返回 [(url, key, model), ...] 候选链。"""
    out = []
    if _CONFIG.taxonomy_api_key and _CONFIG.taxonomy_model:
        out.append(
            (_CONFIG.taxonomy_url, _CONFIG.taxonomy_api_key, _CONFIG.taxonomy_model)
        )
    try:
        import model_routes
        raw = model_routes.all_routes().get("consolidate")
        entries = [raw] if isinstance(raw, dict) else (raw or [])
        for e in entries:
            if isinstance(e, dict) and e.get("url") and e.get("key"):
                out.append((e["url"], e["key"], e.get("model", "deepseek-chat")))
    except Exception:
        pass
    try:
        kv = {}
        with open(LOCAL_USER_SECRETS) as f:
            for line in f:
                if "=" in line:
                    k, v = line.strip().split("=", 1)
                    kv[k] = v
        if kv.get("DEEPSEEK_API_KEY"):
            out.append((kv["DEEPSEEK_BASE_URL"] + "/v1", kv["DEEPSEEK_API_KEY"], "deepseek-chat"))
    except Exception:
        pass
    return out


def _call(messages, max_tokens=800):
    for url, key, model in _routes():
        u = url.rstrip("/")
        if not u.endswith("/chat/completions"):
            u += "/chat/completions"
        body = {"model": model, "messages": messages, "temperature": 0.1,
                "max_tokens": max_tokens, "stream": False}
        headers = {"Authorization": f"Bearer {key}"}
        for trust in (False, True):  # 直连优先，代理回退
            try:
                with httpx.Client(timeout=60, trust_env=trust) as client:
                    resp = client.post(u, headers=headers, json=body)
                data = resp.json()
                content = (((data.get("choices") or [{}])[0].get("message") or {})
                           .get("content") or "").strip()
                if content:
                    return content
            except Exception:
                continue
    return ""


def validate(tags):
    if not isinstance(tags, list) or not (3 <= len(tags) <= 5):
        return False
    if any(t not in VALID for t in tags):
        return False
    pre = {t.split(":")[0] for t in tags}
    return {"state", "domain", "kind"} <= pre


def tag_one(memory_id: str, text: str, timestamp: str = "") -> bool:
    """给单条记忆打tag并写回。线程安全（自建连接）。成功返回True。"""
    _tz_offset = int(os.environ.get("ANCHOR_TIMEZONE_OFFSET", "0"))
    today = datetime.datetime.now(
        datetime.timezone(datetime.timedelta(hours=_tz_offset))).strftime("%Y-%m-%d")
    payload = json.dumps({"id": memory_id, "time": (timestamp or today)[:10],
                          "text": (text or "")[:500]}, ensure_ascii=False)
    raw = _call([{"role": "system", "content": PROMPT.replace("{TODAY}", today)},
                 {"role": "user", "content": "给下面每条记忆打标签：\n" + payload}])
    for line in raw.splitlines():
        line = line.strip().strip("`")
        if not line.startswith("{"):
            continue
        try:
            d = json.loads(line)
        except Exception:
            continue
        tags = d.get("tags")
        if validate(tags):
            conn = sqlite3.connect(DB, timeout=10)
            try:
                conn.execute("PRAGMA busy_timeout=5000")
                conn.execute("UPDATE memories SET tag = ? WHERE memory_id = ?",
                             (",".join(tags), memory_id))
                conn.commit()
                return True
            finally:
                conn.close()
    return False


def tag_async(memory_id: str, text: str, timestamp: str = ""):
    """fire-and-forget 后台打标（store路径专用）。"""
    import threading

    def _run():
        try:
            tag_one(memory_id, text, timestamp)
        except Exception:
            pass

    threading.Thread(target=_run, daemon=True).start()


def sweep(limit: int = 200) -> dict:
    """维护班/夜批用：抓 tag 不带五轴前缀的漏网记忆重打（排除wenku）。"""
    conn = sqlite3.connect(DB, timeout=10)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT memory_id, text, timestamp FROM memories "
        "WHERE COALESCE(collection,'') != 'wenku' AND tag NOT LIKE '%state:%' "
        "ORDER BY timestamp DESC LIMIT ?", (limit,)).fetchall()
    conn.close()
    ok = sum(1 for r in rows if tag_one(r["memory_id"], r["text"], r["timestamp"]))
    return {"found": len(rows), "tagged": ok}


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "sweep":
        print(json.dumps(sweep(), ensure_ascii=False))
    elif len(sys.argv) > 3:
        print(tag_one(sys.argv[1], sys.argv[2], sys.argv[3]))
