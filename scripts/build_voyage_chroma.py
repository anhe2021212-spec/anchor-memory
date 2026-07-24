import json
import os
import sqlite3
import struct
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import chromadb
from release_config import AnchorConfig


_CONFIG = AnchorConfig.load()
MEMORY_DB = _CONFIG.db_path
VEC_DB = Path(os.environ.get("VOYAGE_OUT_DB", _CONFIG.data_dir / "voyage_embeddings.sqlite3"))
CHROMA_PATH = _CONFIG.chroma_dir or (_CONFIG.data_dir / "chroma")
_KEY_FILE_VALUE = os.environ.get("VOYAGE_KEY_FILE", "").strip()
KEY_FILE = Path(_KEY_FILE_VALUE).expanduser() if _KEY_FILE_VALUE else None

MODEL = os.environ.get("VOYAGE_MODEL", "voyage-4-large")
OUTPUT_DIM = int(os.environ.get("VOYAGE_OUTPUT_DIM", "1024"))
MEM_COLLECTION = os.environ.get("ANCHOR_CHROMA_COLLECTION", "memories_voyage4_1024")
SHADOW_COLLECTION = os.environ.get("ANCHOR_SHADOW_COLLECTION", "memory_shadows_voyage4_1024")
BATCH_SIZE = int(os.environ.get("VOYAGE_CHROMA_BATCH_SIZE", "64"))


def load_key() -> str:
    env_key = os.environ.get("VOYAGE_API_KEY", "").strip()
    if env_key:
        return env_key
    if KEY_FILE is None or not KEY_FILE.is_file():
        raise RuntimeError(f"missing key file: {KEY_FILE}")
    for raw in KEY_FILE.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("VOYAGE_API_KEY="):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
        if line.startswith("pa-"):
            return line
    raise RuntimeError(f"VOYAGE_API_KEY not found in {KEY_FILE}")


def unpack_vector(blob: bytes):
    return list(struct.unpack(f"<{OUTPUT_DIM}f", blob))


def voyage_embed(api_key: str, texts: list[str]) -> list[list[float]]:
    payload = {
        "model": MODEL,
        "input": [(t or " ").strip() or " " for t in texts],
        "input_type": "document",
        "output_dimension": OUTPUT_DIM,
        "truncation": True,
    }
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    last_error = None
    for attempt in range(12):
        req = urllib.request.Request(
            "https://api.voyageai.com/v1/embeddings",
            data=body,
            method="POST",
            headers=headers,
        )
        try:
            with urllib.request.urlopen(req, timeout=90) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            return [item["embedding"] for item in data["data"]]
        except urllib.error.HTTPError as exc:
            last_error = exc
            if exc.code != 429:
                detail = exc.read().decode("utf-8", errors="replace")[:300]
                raise RuntimeError(f"Voyage HTTP {exc.code}: {detail}") from exc
            wait = min(30 + attempt * 30, 180)
            print(f"rate limited; sleeping {wait}s", flush=True)
            time.sleep(wait)
    raise RuntimeError(f"Voyage still rate-limited: {last_error}") from last_error


def memory_rows():
    with sqlite3.connect(MEMORY_DB) as mem, sqlite3.connect(VEC_DB) as vec:
        mem.row_factory = sqlite3.Row
        rows = {
            r["memory_id"]: dict(r)
            for r in mem.execute(
                """
                SELECT memory_id, text, tag, tier, level, timestamp, collection
                FROM memories
                WHERE coalesce(collection, '') != 'wenku'
                  AND coalesce(text, '') != ''
                """
            )
        }
        for mid, blob in vec.execute(
            """
            SELECT memory_id, embedding
            FROM voyage_embeddings
            WHERE model=? AND output_dim=?
            """,
            (MODEL, OUTPUT_DIM),
        ):
            row = rows.get(mid)
            if not row:
                continue
            yield row, unpack_vector(blob)


def shadow_rows(api_key: str):
    with sqlite3.connect(MEMORY_DB) as con:
        con.row_factory = sqlite3.Row
        rows = con.execute(
            """
            SELECT s.shadow_id, s.parent_id, s.shadow_key, s.span_start, s.span_end,
                   s.kind, s.version, m.tag AS parent_tag, m.timestamp AS parent_ts
            FROM shadows s
            JOIN memories m ON m.memory_id = s.parent_id
            WHERE coalesce(m.collection, '') != 'wenku'
              AND coalesce(s.shadow_key, '') != ''
            ORDER BY s.shadow_id
            """
        ).fetchall()
    for start in range(0, len(rows), BATCH_SIZE):
        batch = [dict(r) for r in rows[start:start + BATCH_SIZE]]
        vectors = voyage_embed(api_key, [r["shadow_key"] for r in batch])
        for row, vec in zip(batch, vectors):
            yield row, vec


def upsert_batches(collection, items, kind: str):
    ids, docs, metas, embs = [], [], [], []
    total = 0
    for row, vec in items:
        if kind == "memory":
            ids.append(row["memory_id"])
            docs.append(row["text"])
            metas.append({
                "memory_id": row["memory_id"],
                "timestamp": row.get("timestamp") or "",
                "tag": row.get("tag") or "",
                "level": row.get("level") or "raw",
                "tier": row.get("tier") or "long",
                "collection": row.get("collection") or "",
            })
        else:
            ids.append(row["shadow_id"])
            docs.append(row["shadow_key"])
            metas.append({
                "parent_id": row["parent_id"],
                "span_start": row["span_start"] if row["span_start"] is not None else -1,
                "span_end": row["span_end"] if row["span_end"] is not None else -1,
                "kind": row.get("kind") or "topic",
                "shadow_version": int(row.get("version") or 1),
                "parent_tag": row.get("parent_tag") or "",
                "parent_ts": row.get("parent_ts") or "",
            })
        embs.append(vec)
        if len(ids) >= BATCH_SIZE:
            collection.upsert(ids=ids, embeddings=embs, documents=docs, metadatas=metas)
            total += len(ids)
            print(f"{kind}: upserted {total}", flush=True)
            ids, docs, metas, embs = [], [], [], []
    if ids:
        collection.upsert(ids=ids, embeddings=embs, documents=docs, metadatas=metas)
        total += len(ids)
        print(f"{kind}: upserted {total}", flush=True)
    return total


def main() -> int:
    api_key = load_key()
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    mem_col = client.get_or_create_collection(name=MEM_COLLECTION, metadata={"hnsw:space": "cosine"})
    shadow_col = client.get_or_create_collection(name=SHADOW_COLLECTION, metadata={"hnsw:space": "cosine"})

    mem_total = upsert_batches(mem_col, memory_rows(), "memory")
    shadow_total = upsert_batches(shadow_col, shadow_rows(api_key), "shadow")
    print(json.dumps({
        "memory_collection": MEM_COLLECTION,
        "memory_count": mem_col.count(),
        "memory_upserted": mem_total,
        "shadow_collection": SHADOW_COLLECTION,
        "shadow_count": shadow_col.count(),
        "shadow_upserted": shadow_total,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
