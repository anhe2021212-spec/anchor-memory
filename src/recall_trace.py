"""
检索留痕 (recall trace) — 旁路诊断, 绝不进AI agent上文。

记录两类事件：
1. Anchor search 的擦边/留空。
2. Gateway reflex 每轮召回审计：门禁、候选、选中、拒绝、fallback、slot4。

物理独立文件只保留近24小时，另有行数/体积安全阀。
运维按需查: python -m recall_trace [关键词] [条数]；AI agent 从不读它。
纯 additive: _ENABLED=False 即完全静默、零行为变化。
"""
import json
import os
import time
from release_config import AnchorConfig

_CONFIG = AnchorConfig.load()
_FILE = os.environ.get("ANCHOR_RECALL_TRACE", str(_CONFIG.data_dir / "recall_trace.jsonl"))
_DIR = os.path.dirname(os.path.abspath(_FILE))
_MAX_AGE_SECONDS = 24 * 3600
_MAX_LINES = 5000          # 安全阀；正常主要靠24小时裁剪
_MAX_BYTES = 8 * 1024 * 1024
_ENABLED = True            # 总开关; False = 完全静默

# 向量"擦边"窗口: 被gate丢、且距离落这区间的, 才是"真差一点", 值得记
_NEAR_LO = 0.45
_NEAR_HI = 0.68


def enabled() -> bool:
    return _ENABLED


def _now_ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _parse_ts(value: str) -> float:
    try:
        return time.mktime(time.strptime(str(value or ""), "%Y-%m-%d %H:%M:%S"))
    except Exception:
        return 0.0


def _keep_recent(lines: list) -> list:
    cutoff = time.time() - _MAX_AGE_SECONDS
    kept = []
    for ln in lines:
        try:
            rec = json.loads(ln)
        except Exception:
            continue
        ts = _parse_ts(rec.get("t"))
        if ts and ts >= cutoff:
            kept.append(json.dumps(rec, ensure_ascii=False) + "\n")
    if len(kept) > _MAX_LINES:
        kept = kept[-_MAX_LINES:]
    return kept


def _trim_if_needed(force: bool = False):
    try:
        if not os.path.exists(_FILE):
            return
        size = os.path.getsize(_FILE)
        if not force and size <= _MAX_BYTES:
            # 轻量路径也要偶尔按24小时裁剪，避免低流量旧日志长期躺着。
            if int(time.time()) % 60 != 0:
                return
        with open(_FILE, "r", encoding="utf-8") as f:
            lines = f.readlines()
        kept = _keep_recent(lines)
        if force or len(kept) != len(lines) or size > _MAX_BYTES:
            with open(_FILE, "w", encoding="utf-8") as f:
                f.writelines(kept)
    except Exception:
        pass


def _clip(value, n: int = 240):
    s = " ".join(str(value or "").split())
    return s if len(s) <= n else s[:n] + "…"


def log(stage: str, query: str, payload: dict):
    """写一条留痕。stage: route|search|reflex。绝不抛异常拖垮检索。"""
    if not _ENABLED:
        return
    try:
        os.makedirs(_DIR, exist_ok=True)
        _trim_if_needed()
        rec = {"t": _now_ts(), "stage": stage, "q": _clip(query, 180)}
        rec.update(payload or {})
        with open(_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception:
        pass


def log_search(query: str, trace_candidates: list, kept: int, gate_max: float):
    """从 search() 回传的 trace_out 里挑擦边事件记。
    trace_candidates: [{id,dist,score,tag,ts,snip,verdict,reason}]。
    只在 (留空) 或 (有擦边被丢) 时落盘——成功的高分命中不记, 控量。"""
    if not _ENABLED:
        return
    near = []
    for c in trace_candidates:
        if c.get("verdict") != "drop":
            continue
        d = c.get("dist")
        if d is None:
            near.append({**c, "why": "纯BM25被丢"})
        elif _NEAR_LO <= d <= _NEAR_HI:
            near.append({**c, "why": "向量擦边被丢"})
    if kept == 0 or near:
        log("search", query, {
            "gate_max": gate_max,
            "kept": kept,
            "empty": kept == 0,
            "n_cand": len(trace_candidates),
            "near_miss": near[:8],
        })


def log_reflex(query: str, payload: dict):
    """Gateway 反射弧每轮审计。payload 由 gateway 控制字段和裁剪。"""
    if not _ENABLED:
        return
    log("reflex", query, payload or {})


def tail(substr: str = "", n: int = 30) -> list:
    _trim_if_needed(force=True)
    if not os.path.exists(_FILE):
        return []
    with open(_FILE, "r", encoding="utf-8") as f:
        lines = f.readlines()
    out = []
    for ln in reversed(lines):
        if substr and substr not in ln:
            continue
        try:
            out.append(json.loads(ln))
        except Exception:
            continue
        if len(out) >= n:
            break
    return out


if __name__ == "__main__":
    import sys
    sub = sys.argv[1] if len(sys.argv) > 1 else ""
    nn = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    rows = tail(sub, nn)
    if not rows:
        print("(无留痕)" + (f"  关键词={sub}" if sub else ""))
    for r in rows:
        print(json.dumps(r, ensure_ascii=False))
