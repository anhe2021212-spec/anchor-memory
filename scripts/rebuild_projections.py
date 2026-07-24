#!/usr/bin/env python3
"""Build a fresh, self-verifying Chroma/Kuzu bundle from SQLite authority.

The source database is opened read-only and copied with SQLite backup. The
output directory must not exist, so a failed run can never overwrite a live
projection. The copied SQLite file is evidence and a reproducible rebuild
input; it is not automatically promoted to become the operator authority.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def readonly_backup(source: Path, destination: Path) -> None:
    uri = f"file:{source.resolve().as_posix()}?mode=ro"
    with sqlite3.connect(uri, uri=True) as src, sqlite3.connect(destination) as dst:
        src.backup(dst)


def memory_rows(db_path: Path) -> list[dict]:
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        return [
            dict(row)
            for row in conn.execute(
                "SELECT memory_id,text,timestamp,tag,tier,level,collection "
                "FROM memories ORDER BY memory_id"
            )
        ]


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--source-db", type=Path, required=True)
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument(
        "--embedding-provider",
        choices=("local", "bge", "voyage"),
        default="local",
    )
    result.add_argument("--embedding-model", default="BAAI/bge-base-zh-v1.5")
    result.add_argument("--batch-size", type=int, default=64)
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    source = args.source_db.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    if not source.is_file():
        print(f"source database not found: {source}", file=sys.stderr)
        return 2
    if output.exists():
        print(f"output directory must not exist: {output}", file=sys.stderr)
        return 2
    if args.batch_size < 1:
        print("batch size must be positive", file=sys.stderr)
        return 2

    output.mkdir(parents=True)
    snapshot = output / "memories.db"
    try:
        readonly_backup(source, snapshot)
        os.environ["ANCHOR_EMBED_PROVIDER"] = args.embedding_provider
        os.environ["ANCHOR_KUZU_PATH"] = str(output / "kuzu_db")
        os.environ.pop("ANCHOR_DISABLE_CHROMA", None)

        from anchor_memory import AnchorMemory

        memory = AnchorMemory(str(output), embedding_model=args.embedding_model)
        if memory._client is None:
            raise RuntimeError("Chroma is unavailable; install the chroma extra")
        if not memory.db.kuzu_available:
            raise RuntimeError("Kuzu is unavailable; install the kuzu extra")

        rows = memory_rows(snapshot)
        for start in range(0, len(rows), args.batch_size):
            batch = rows[start : start + args.batch_size]
            memory._collection.upsert(
                ids=[row["memory_id"] for row in batch],
                documents=[row["text"] for row in batch],
                embeddings=[memory._encode_document(row["text"]) for row in batch],
                metadatas=[
                    {
                        "memory_id": row["memory_id"],
                        "timestamp": row.get("timestamp") or "",
                        "tag": row.get("tag") or "general",
                        "tier": row.get("tier") or "short",
                        "level": row.get("level") or "raw",
                        "collection": row.get("collection") or "",
                    }
                    for row in batch
                ],
            )

        sqlite_count = len(rows)
        chroma_count = memory._collection.count()
        kuzu_rows = memory.db._kuzu_rows("MATCH (m:Memory) RETURN count(m)")
        kuzu_count = int(kuzu_rows[0][0]) if kuzu_rows else -1
        dimension = len(memory._encode_document("dimension probe"))
        evidence = {
            "status": "ok" if sqlite_count == chroma_count == kuzu_count else "failed",
            "built_at": datetime.now(timezone.utc).isoformat(),
            "source_sha256": sha256(source),
            "snapshot_sha256": sha256(snapshot),
            "embedding_provider": args.embedding_provider,
            "embedding_model": args.embedding_model,
            "embedding_dimension": dimension,
            "sqlite_count": sqlite_count,
            "chroma_count": chroma_count,
            "kuzu_count": kuzu_count,
        }
        (output / "projection-metadata.json").write_text(
            json.dumps(evidence, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(evidence, ensure_ascii=False, indent=2))
        return 0 if evidence["status"] == "ok" else 1
    except Exception as exc:
        print(f"projection rebuild failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        shutil.rmtree(output, ignore_errors=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
