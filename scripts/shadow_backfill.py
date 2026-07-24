#!/usr/bin/env python3
"""
影子索引回填 / 增量 (shadow backfill) — 给现有长记忆补"话题钥匙"。

工作流(护 RAM: 本脚本绝不加载 bge 模型):
  1. 扫 memories 表, 过 should_index 门控、且无当前 version shadow 的记忆。
  2. 每条: 调 DeepSeek 产 keys(纯网络) → POST /api/shadow/upsert(常驻 anchor-sse 进程
     用它那个常驻 embedder 嵌 key + 算 span + 写 shadows 表 & memory_shadows collection)。
  3. 幂等: 按 version 跳过已处理; 可中断续跑; 失败写 jsonl 可重试。

默认 dry-run(不落库, 对齐 big_consolidate/swap_pass 惯例)。--apply 才真写。

用法:
  python3 shadow_backfill.py                      # dry-run: 只数 + 预览门控统计
  python3 shadow_backfill.py --sample 5           # dry-run: 真打 5 条估 token/看钥匙质量(不写)
  python3 shadow_backfill.py --apply              # 全量回填(走服务嵌入)
  python3 shadow_backfill.py --apply --limit 20   # 增量(cron 每 N 分钟跑一批)
  python3 shadow_backfill.py --retry-failed       # 重试失败档里的记忆
"""
import os
import sys
import json
import time
import argparse
import sqlite3

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import shadow_index as si
import httpx
from release_config import AnchorConfig

_CONFIG = AnchorConfig.load()
DB_PATH = str(_CONFIG.db_path)
FAIL_LOG = os.environ.get(
    "ANCHOR_SHADOW_FAILURE_LOG", str(_CONFIG.log_dir / "shadow_backfill_failures.jsonl")
)
DEFAULT_API = os.environ.get("ANCHOR_INTERNAL_API", "http://127.0.0.1:8765")


def _candidates(conn, version, only_ids=None):
    """过门控、且无当前 version shadow 的记忆。only_ids: 限定一组 id(重试用)。"""
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT memory_id, text, tag, collection, level, timestamp FROM memories "
        "ORDER BY timestamp DESC").fetchall()
    todo, gated_out, already = [], 0, 0
    for r in rows:
        mid = r["memory_id"]
        if only_ids is not None and mid not in only_ids:
            continue
        if not si.should_index(r["text"] or "", r["tag"] or "", r["collection"] or "", r["level"] or "raw"):
            gated_out += 1
            continue
        if si.has_current_shadow(conn, mid, version):
            already += 1
            continue
        todo.append(dict(r))
    return todo, gated_out, already, len(rows)


def _upsert_via_service(api, parent_id, keys, version):
    """POST 到常驻 anchor-sse, 由它用常驻 embedder 嵌入+写库。返回 written 条数, 失败抛异常。"""
    resp = httpx.post(f"{api}/api/shadow/upsert",
                      json={"parent_id": parent_id, "version": version, "keys": keys},
                      timeout=30)
    if resp.status_code != 200:
        raise RuntimeError(f"upsert http {resp.status_code}: {resp.text[:160]}")
    data = resp.json()
    if not data.get("ok"):
        raise RuntimeError(f"upsert rejected: {data}")
    return data.get("written", 0)


def _log_failure(parent_id, err):
    os.makedirs(os.path.dirname(FAIL_LOG), exist_ok=True)
    with open(FAIL_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps({"t": si._utcnow(), "parent_id": parent_id, "err": str(err)[:300]},
                           ensure_ascii=False) + "\n")


def _load_failed_ids():
    if not os.path.exists(FAIL_LOG):
        return set()
    ids = set()
    with open(FAIL_LOG, "r", encoding="utf-8") as f:
        for ln in f:
            try:
                ids.add(json.loads(ln)["parent_id"])
            except Exception:
                pass
    return ids


def main():
    ap = argparse.ArgumentParser(description="影子索引回填/增量")
    ap.add_argument("--apply", action="store_true", help="真写库(默认 dry-run)")
    ap.add_argument("--limit", type=int, default=None, help="最多处理 N 条(cron 增量用)")
    ap.add_argument("--sample", type=int, default=0, help="dry-run 下真打 N 条估 token/看质量")
    ap.add_argument("--version", type=int, default=si.SHADOW_VERSION, help="shadow version")
    ap.add_argument("--api", default=DEFAULT_API, help="anchor-sse 内部 REST")
    ap.add_argument("--sleep", type=float, default=0.5, help="每条间隔秒")
    ap.add_argument("--retry-failed", action="store_true", help="只重试失败档里的记忆")
    ap.add_argument("--only", default=None, help="只处理这一个 memory_id(pilot/调试, 跳过门控与去重)")
    ap.add_argument("--db", default=DB_PATH)
    args = ap.parse_args()

    si.ensure_shadow_table(args.db)
    routes = si.load_routes()
    if not routes or not any(route.get("key") for route in routes):
        print("[FATAL] model_routes.json 缺 consolidate.key"); sys.exit(1)
    try:
        si.load_prompt()
    except Exception as e:
        print(f"[FATAL] 读不到 shadow_prompt.txt: {e}"); sys.exit(1)

    if args.only:                              # pilot: 强制处理这一条, 跳过门控/去重
        conn = sqlite3.connect(args.db, timeout=10.0); conn.row_factory = sqlite3.Row
        r = conn.execute("SELECT memory_id, text, tag, collection, timestamp "
                         "FROM memories WHERE memory_id=?", (args.only,)).fetchone()
        conn.close()
        if not r:
            print(f"找不到记忆 {args.only}"); return
        todo, gated_out, already, total = [dict(r)], 0, 0, 1
    else:
        conn = sqlite3.connect(args.db, timeout=10.0)
        only = _load_failed_ids() if args.retry_failed else None
        if args.retry_failed and not only:
            print("失败档为空, 无可重试。"); return
        todo, gated_out, already, total = _candidates(conn, args.version, only_ids=only)
        conn.close()

    model_chain = " -> ".join(route["model"] for route in routes)
    print(f"== 影子回填 v{args.version} | models={model_chain} ==")
    print(f"库内记忆 {total} 条 | 过门控待办 {len(todo)} | 已有v{args.version} {already} | 门控跳过 {gated_out}")
    if args.limit:
        todo = todo[:args.limit]
        print(f"本批限 {args.limit} → 实处理 {len(todo)} 条")

    # ── dry-run ──
    if not args.apply:
        if args.sample > 0:
            n = min(args.sample, len(todo))
            print(f"\n-- 抽样 {n} 条真打 DeepSeek(只看不写) --")
            tot_tok = 0
            for r in todo[:n]:
                try:
                    keys, usage = si.call_deepseek_any(
                        r["text"], routes, return_usage=True)
                except Exception as e:
                    print(f"  [{r['memory_id']}] 生成失败: {e}"); continue
                tot_tok += usage.get("total_tokens", 0)
                print(f"  [{r['memory_id']}] tok={usage.get('total_tokens')} "
                      f"reason={usage.get('completion_tokens_details',{}).get('reasoning_tokens')} "
                      f"keys={len(keys)}")
                for k in keys:
                    span = si.locate_span(r["text"], k["quote"])
                    flag = "OK" if span else "XX-span未命中(降级指向整条)"
                    print(f"      [{flag}] {k['key']!r} ⇐ {k['quote'][:32]!r}")
                time.sleep(args.sleep)
            if n:
                avg = tot_tok / n
                print(f"\n  实测均 {avg:.0f} tok/条 → 全量 {len(todo)} 条估 ~{avg*len(todo)/1000:.0f}k tokens")
        else:
            print("\n(dry-run。--sample 5 抽样看钥匙质量+估token; --apply 真写)")
        return

    # ── apply ──
    print(f"\n-- APPLY: 走 {args.api}/api/shadow/upsert(常驻模型嵌入) --")
    ok, fail, written_total, tok_total = 0, 0, 0, 0
    for i, r in enumerate(todo, 1):
        mid = r["memory_id"]
        try:
            keys, usage = si.call_deepseek_any(
                r["text"], routes, return_usage=True)
            tok_total += usage.get("total_tokens", 0)
            if not keys:
                print(f"  [{i}/{len(todo)}] {mid} 0 keys, 跳过"); continue
            written = _upsert_via_service(args.api, mid, keys, args.version)
            written_total += written
            ok += 1
            print(f"  [{i}/{len(todo)}] {mid} ✓ {written} keys")
        except Exception as e:
            fail += 1
            _log_failure(mid, e)
            print(f"  [{i}/{len(todo)}] {mid} ✗ {str(e)[:80]}")
        time.sleep(args.sleep)
    print(f"\n完成: 成功 {ok} | 失败 {fail}(见 {FAIL_LOG}) | 写入 {written_total} keys | 耗 ~{tok_total/1000:.0f}k tok")


if __name__ == "__main__":
    main()
