#!/usr/bin/env python3
"""
Anchor Memory · 维护脚本（真正的 swap）

每日 cron 调用。覆盖：
  trash    过期 trash 永删
  events   孤儿 events 清理（events.memory_id 指向已删 memory）
  edges    只衰减/回收权威 flow_edges 的机器弱边；semantic/legacy 不动
  tier     tier 脏值修复
  fts      fts_map 孤儿清理 + 有界补齐缺失索引
  chroma   ChromaDB 向量数 vs memories 行数一致性（只报告）
  vacuum   VACUUM 回收空间

默认 dry-run。--apply 才真改。

用法：
  python3 maintenance.py                              # 全任务 dry-run
  python3 maintenance.py --apply                      # 全任务真改
  python3 maintenance.py --only trash,edges --apply  # 只做指定任务
  python3 maintenance.py --json                       # 机器可读

调度器示例：python -m maintenance --apply --json
日志位置由调用方或 ANCHOR_LOG_DIR 决定。
"""
import argparse
import json
import os
import sqlite3
import sys
from datetime import datetime, timedelta
from release_config import AnchorConfig

_CONFIG = AnchorConfig.load()
WEIGHT_CAP = float(os.environ.get("ANCHOR_EDGE_WEIGHT_CAP", "2.0"))

DB_PATH = str(_CONFIG.db_path)
CHROMA_PATH = str(_CONFIG.chroma_dir or (_CONFIG.data_dir / "chroma"))

DEFAULTS = {
    "trash_days": 7,
    # edge_threshold = 0.3 比原版 0.5 保守：服务器当前 75% edges weight<0.3，
    # 0.5 阈值会一次删 9000+ 条；先用 0.3 慢慢逼近，确认 dream_pass 接管之后
    # 可以再调高。改阈值之前务必 dry-run 一次看 weak_deleted 数字。
    "edge_threshold": 0.08,
    "edge_cap": WEIGHT_CAP,
    # 2026-06-13 edge-explosion 防复发: 弱边(<edge_stale_weight)且超过
    # edge_stale_days 没被强化/激活(last_fired) 的, 当作废弃 spurious 边回收。
    # 拦住"被强化到 >=0.3 逃过绝对阈值、但其实从此没人用"的爆炸残留。
    "edge_stale_weight": 0.5,
    "edge_stale_days": 14,
    # 每次最多回收 100 个无向 pair；同时保留每个端点至少一个邻居，避免首轮清理
    # 因历史积压批量删边或制造孤岛。衰减仍覆盖全部机器边。
    "edge_delete_pairs_per_run": 100,
    "canonical_tiers": {"core", "long", "short"},
}

ALL_TASKS = ["trash", "events", "edges", "tier", "fts", "chroma", "vacuum"]


def _conn(db_path):
    conn = sqlite3.connect(db_path, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def task_trash(conn, args, dry_run):
    # deleted_at 是 Python isoformat (含 'T')，SQLite datetime() 返回空格分隔；
    # 字符串比较时 'T'(84) > ' '(32)，会让同日早于 cutoff 时刻的记录漏清。
    # 用 replace 把 'T' 换成空格再比，跟 datetime() 输出对齐。
    q = "SELECT COUNT(*) FROM trash WHERE replace(deleted_at,'T',' ') < datetime('now', ?)"
    delta = f"-{args.trash_days} days"
    n = conn.execute(q, (delta,)).fetchone()[0]
    if not dry_run and n:
        conn.execute(
            "DELETE FROM trash WHERE replace(deleted_at,'T',' ') < datetime('now', ?)",
            (delta,),
        )
        conn.commit()
    return {"task": "trash", "candidates": n, "applied": not dry_run}


def task_events(conn, args, dry_run):
    q = """
        SELECT COUNT(*) FROM events
        WHERE memory_id IS NOT NULL
          AND memory_id NOT IN (SELECT memory_id FROM memories)
    """
    n = conn.execute(q).fetchone()[0]
    if not dry_run and n:
        conn.execute("""
            DELETE FROM events
            WHERE memory_id IS NOT NULL
              AND memory_id NOT IN (SELECT memory_id FROM memories)
        """)
        conn.commit()
    return {"task": "events", "orphans": n, "applied": not dry_run}


def task_edges(conn, args, dry_run):
    exists = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='flow_edges'"
    ).fetchone()
    if not exists:
        return {"task": "edges", "weak_deleted": 0, "stale_deleted": 0,
                "machine_decayed": 0, "applied": not dry_run, "note": "flow_edges unavailable"}
    machine = "(provenance LIKE 'auto_%' OR provenance IN ('cluster','knn_legacy'))"
    pair_limit = getattr(args, "edge_delete_pairs_per_run",
                         DEFAULTS["edge_delete_pairs_per_run"])
    stale_cutoff = datetime.utcnow() - timedelta(days=args.edge_stale_days)
    stale_iso = stale_cutoff.isoformat()
    machine_n = conn.execute(
        f"SELECT COUNT(*) FROM flow_edges WHERE {machine}"
    ).fetchone()[0]

    # 以无向 pair 为回收单位，只处理两向都属于机器 provenance 的 pair。
    # 先按当前全图计算 distinct-neighbor degree，再逐个模拟删除；任一端点只剩
    # 一个邻居时就跳过，避免维护任务制造孤岛。
    degree = {}
    for row in conn.execute("""
        SELECT node, COUNT(DISTINCT neighbor) AS degree FROM (
          SELECT source_id AS node, target_id AS neighbor FROM flow_edges
          UNION
          SELECT target_id AS node, source_id AS neighbor FROM flow_edges
        ) GROUP BY node
    """):
        degree[row["node"]] = int(row["degree"])
    candidates = conn.execute(f"""
        WITH machine_pairs AS (
          SELECT CASE WHEN source_id < target_id THEN source_id ELSE target_id END AS a,
                 CASE WHEN source_id < target_id THEN target_id ELSE source_id END AS b,
                 MAX(weight) AS max_weight,
                 MAX(COALESCE(last_fired, '')) AS newest_fire,
                 COUNT(*) AS row_count
          FROM flow_edges
          WHERE {machine}
          GROUP BY a, b
        )
        SELECT p.*,
               CASE WHEN p.max_weight < ? THEN 'weak' ELSE 'stale' END AS reason
        FROM machine_pairs p
        WHERE (p.max_weight < ? OR (p.max_weight < ? AND p.newest_fire < ?))
          AND NOT EXISTS (
            SELECT 1 FROM flow_edges f
            WHERE ((f.source_id=p.a AND f.target_id=p.b)
                OR (f.source_id=p.b AND f.target_id=p.a))
              AND NOT (f.provenance LIKE 'auto_%'
                       OR f.provenance IN ('cluster','knn_legacy'))
          )
        ORDER BY p.max_weight ASC, p.newest_fire ASC
    """, (args.edge_threshold, args.edge_threshold,
          args.edge_stale_weight, stale_iso)).fetchall()
    selected = []
    for row in candidates:
        a, b = row["a"], row["b"]
        if degree.get(a, 0) <= 1 or degree.get(b, 0) <= 1:
            continue
        selected.append(row)
        degree[a] -= 1
        degree[b] -= 1
        if len(selected) >= pair_limit:
            break
    weak_n = sum(int(r["row_count"]) for r in selected if r["reason"] == "weak")
    stale_n = sum(int(r["row_count"]) for r in selected if r["reason"] == "stale")
    if not dry_run:
        conn.execute(f"UPDATE flow_edges SET weight=weight*0.96 WHERE {machine}")
        for row in selected:
            conn.execute(
                f"DELETE FROM flow_edges WHERE {machine} AND "
                "((source_id=? AND target_id=?) OR (source_id=? AND target_id=?))",
                (row["a"], row["b"], row["b"], row["a"]),
            )
        conn.commit()
    return {
        "task": "edges",
        "weak_deleted": weak_n,
        "stale_deleted": stale_n,
        "machine_decayed": machine_n,
        "candidate_pairs": len(candidates),
        "selected_pairs": len(selected),
        "pair_limit": pair_limit,
        "island_guard": True,
        "applied": not dry_run,
    }


def task_tier(conn, args, dry_run):
    placeholders = ",".join("?" * len(args.canonical_tiers))
    canon = list(args.canonical_tiers)
    n = conn.execute(
        f"SELECT COUNT(*) FROM memories WHERE tier NOT IN ({placeholders})", canon
    ).fetchone()[0]
    if not dry_run and n:
        conn.execute(
            f"UPDATE memories SET tier='long' WHERE tier NOT IN ({placeholders})", canon
        )
        conn.commit()
    return {"task": "tier", "dirty": n, "applied": not dry_run}


def task_fts(conn, args, dry_run):
    """孤儿 fts_map 行（指向已删 memory）→ 删；缺失 fts 行只报告。
    反向孤儿（memories_fts 有 rowid 但 fts_map 没人指）→ 删（FTS 表只能靠 rowid 找回，
    没有 fts_map 反向映射就是死索引，搜不到、占空间、还可能让搜索返回幽灵命中）。
    """
    orphan_q = """
        SELECT memory_id, fts_rowid FROM fts_map
        WHERE memory_id NOT IN (SELECT memory_id FROM memories)
    """
    orphans = conn.execute(orphan_q).fetchall()
    missing_q = """
        SELECT memory_id,text FROM memories
        WHERE memory_id NOT IN (SELECT memory_id FROM fts_map)
        ORDER BY timestamp LIMIT 200
    """
    missing_rows = conn.execute(missing_q).fetchall()
    missing = len(missing_rows)
    reverse_orphan_q = """
        SELECT COUNT(*) FROM memories_fts
        WHERE rowid NOT IN (SELECT fts_rowid FROM fts_map)
    """
    reverse_orphan = conn.execute(reverse_orphan_q).fetchone()[0]
    if not dry_run and orphans:
        for r in orphans:
            conn.execute("DELETE FROM memories_fts WHERE rowid = ?", (r["fts_rowid"],))
        conn.execute("""
            DELETE FROM fts_map
            WHERE memory_id NOT IN (SELECT memory_id FROM memories)
        """)
        conn.commit()
    if not dry_run and reverse_orphan:
        conn.execute("""
            DELETE FROM memories_fts
            WHERE rowid NOT IN (SELECT fts_rowid FROM fts_map)
        """)
        conn.commit()
    repaired = 0
    if not dry_run and missing_rows:
        from anchor_db import AnchorDB
        db = AnchorDB.__new__(AnchorDB)
        db.db_path = args.db
        for row in missing_rows:
            db.fts_upsert(row["memory_id"], row["text"] or "")
            repaired += 1
    return {
        "task": "fts",
        "orphan_index": len(orphans),
        "missing_index": missing,
        "missing_repaired": repaired,
        "reverse_orphan": reverse_orphan,
        "applied": not dry_run,
        "note": "bounded repair limit=200" if missing else "ok",
    }


def task_chroma(conn, args, dry_run):
    """只报告：chroma 向量数 vs memories 行数。"""
    try:
        import chromadb
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        col = client.get_or_create_collection("memories")
        chroma_n = col.count()
    except Exception as e:
        return {"task": "chroma", "error": str(e), "applied": False}
    mem_n = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
    diff = chroma_n - mem_n
    return {
        "task": "chroma",
        "chroma_count": chroma_n,
        "memories_count": mem_n,
        "diff": diff,
        "applied": False,
        "note": "diff != 0 表示向量库与 sqlite 不一致" if diff else "ok",
    }


def task_vacuum(conn, args, dry_run):
    if dry_run:
        return {"task": "vacuum", "skipped": "dry-run"}
    size_before = conn.execute(
        "SELECT page_count * page_size FROM pragma_page_count(), pragma_page_size()"
    ).fetchone()[0]
    conn.execute("VACUUM")
    size_after = conn.execute(
        "SELECT page_count * page_size FROM pragma_page_count(), pragma_page_size()"
    ).fetchone()[0]
    return {
        "task": "vacuum",
        "applied": True,
        "size_before": size_before,
        "size_after": size_after,
        "reclaimed": size_before - size_after,
    }


TASK_FUNCS = {
    "trash": task_trash,
    "events": task_events,
    "edges": task_edges,
    "tier": task_tier,
    "fts": task_fts,
    "chroma": task_chroma,
    "vacuum": task_vacuum,
}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--apply", action="store_true", help="真改（默认 dry-run）")
    p.add_argument("--only", default=None, help="逗号分隔任务列表（默认全部）")
    p.add_argument("--json", action="store_true", help="JSON 输出")
    p.add_argument("--trash-days", type=int, default=DEFAULTS["trash_days"])
    p.add_argument("--edge-threshold", type=float, default=DEFAULTS["edge_threshold"])
    p.add_argument("--edge-cap", type=float, default=DEFAULTS["edge_cap"])
    p.add_argument("--edge-stale-weight", type=float, default=DEFAULTS["edge_stale_weight"])
    p.add_argument("--edge-stale-days", type=int, default=DEFAULTS["edge_stale_days"])
    p.add_argument("--edge-delete-pairs-per-run", type=int,
                   default=DEFAULTS["edge_delete_pairs_per_run"])
    p.add_argument("--db", default=DB_PATH)
    args = p.parse_args()
    args.canonical_tiers = DEFAULTS["canonical_tiers"]

    tasks = args.only.split(",") if args.only else ALL_TASKS
    unknown = [t for t in tasks if t not in TASK_FUNCS]
    if unknown:
        print(f"unknown task(s): {unknown}", file=sys.stderr)
        sys.exit(2)

    dry_run = not args.apply
    started = datetime.utcnow().isoformat()
    results = []

    conn = _conn(args.db)
    try:
        for t in tasks:
            try:
                results.append(TASK_FUNCS[t](conn, args, dry_run))
            except Exception as e:
                results.append({"task": t, "error": str(e)})
        # 每次成功的 apply 都写 maintenance run event。不能用“删除数>0”代表
        # “运行成功”：纯衰减、stale 回收或零动作健康轮也必须可被巡检看见。
        if not dry_run:
            try:
                failures = [r for r in results if r.get("error")]
                if not failures:
                    detail = json.dumps({
                        "status": "success",
                        "tasks": [r["task"] for r in results],
                        "summary": results,
                    }, ensure_ascii=False)
                    conn.execute(
                        "INSERT INTO events (memory_id, event_type, detail, created_at) "
                        "VALUES (NULL, 'maintenance', ?, datetime('now'))",
                        (detail[:2000],),  # 截断防止 detail 撑爆表
                    )
                    conn.commit()
            except Exception as e:
                # event 写不进不影响主流程，但要 print 出来
                print(f"[warn] event log failed: {e}", file=sys.stderr)
    finally:
        conn.close()

    report = {
        "started_at": started,
        "finished_at": datetime.utcnow().isoformat(),
        "dry_run": dry_run,
        "results": results,
    }

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        mode = "DRY-RUN" if dry_run else "APPLIED"
        print(f"=== Anchor Maintenance · {mode} · {started} ===")
        for r in results:
            print(f"  - {r}")
        print(f"=== done {report['finished_at']} ===")

    sys.exit(1 if any("error" in r for r in results) else 0)


if __name__ == "__main__":
    main()
