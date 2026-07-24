import hashlib
import json
import os
import signal
import sqlite3
import struct
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from release_config import AnchorConfig


_CONFIG = AnchorConfig.load()
MEMORY_DB = Path(os.environ.get("ANCHOR_MEMORY_DB", _CONFIG.db_path))
OUT_DB = Path(os.environ.get("VOYAGE_OUT_DB", _CONFIG.data_dir / "voyage_embeddings.sqlite3"))
LOG_PATH = Path(os.environ.get("VOYAGE_LOG", _CONFIG.log_dir / "voyage_backfill.log"))
_KEY_FILE_VALUE = os.environ.get("VOYAGE_KEY_FILE", "").strip()
KEY_FILE = Path(_KEY_FILE_VALUE).expanduser() if _KEY_FILE_VALUE else None

MODEL = os.environ.get("VOYAGE_MODEL", "voyage-4-large")
OUTPUT_DIM = int(os.environ.get("VOYAGE_OUTPUT_DIM", "1024"))
BATCH_SIZE = int(os.environ.get("VOYAGE_BATCH_SIZE", "8"))
MAX_CHARS = int(os.environ.get("VOYAGE_MAX_CHARS", "1200"))
SUCCESS_SLEEP = int(os.environ.get("VOYAGE_SUCCESS_SLEEP", "75"))
MAX_RUNTIME_SECONDS = int(os.environ.get("VOYAGE_MAX_RUNTIME_SECONDS", str(22 * 3600)))
DELETE_KEY_ON_COMPLETE = os.environ.get("VOYAGE_DELETE_KEY_ON_COMPLETE", "0") == "1"

STOP = False


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def log(msg: str) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    line = f"{utc_now()} {msg}"
    print(line, flush=True)
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def handle_stop(signum, frame) -> None:
    global STOP
    STOP = True
    log(f"received signal {signum}; stopping after current batch")


signal.signal(signal.SIGTERM, handle_stop)
signal.signal(signal.SIGINT, handle_stop)


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


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def pack_vector(vec: list[float]) -> bytes:
    return struct.pack(f"<{len(vec)}f", *vec)


def init_out_db() -> None:
    OUT_DB.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(OUT_DB) as con:
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS voyage_embeddings (
                memory_id TEXT PRIMARY KEY,
                model TEXT NOT NULL,
                output_dim INTEGER NOT NULL,
                text_hash TEXT NOT NULL,
                embedding BLOB NOT NULL,
                source_timestamp TEXT,
                source_tag TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS voyage_backfill_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                started_at TEXT NOT NULL,
                finished_at TEXT,
                status TEXT NOT NULL,
                model TEXT NOT NULL,
                output_dim INTEGER NOT NULL,
                batch_size INTEGER NOT NULL,
                max_chars INTEGER NOT NULL,
                embedded_count INTEGER DEFAULT 0,
                note TEXT DEFAULT ''
            )
            """
        )
        con.execute("CREATE INDEX IF NOT EXISTS idx_voyage_model_hash ON voyage_embeddings(model, output_dim, text_hash)")


def load_pending(limit: int) -> list[dict]:
    sql = """
    SELECT memory_id, text, tag, tier, level, timestamp, collection
    FROM memories
    WHERE coalesce(collection, '') != 'wenku'
      AND coalesce(text, '') != ''
    ORDER BY timestamp DESC
    """
    with sqlite3.connect(MEMORY_DB) as src, sqlite3.connect(OUT_DB) as out:
        src.row_factory = sqlite3.Row
        existing = {
            row[0]: row[1]
            for row in out.execute(
                "SELECT memory_id, text_hash FROM voyage_embeddings WHERE model=? AND output_dim=?",
                (MODEL, OUTPUT_DIM),
            )
        }
        pending = []
        for row in src.execute(sql):
            item = dict(row)
            h = text_hash(item["text"])
            if existing.get(item["memory_id"]) == h:
                continue
            item["text_hash"] = h
            pending.append(item)
            if len(pending) >= limit:
                break
        return pending


def counts() -> tuple[int, int]:
    with sqlite3.connect(MEMORY_DB) as src, sqlite3.connect(OUT_DB) as out:
        total = src.execute(
            "SELECT count(*) FROM memories WHERE coalesce(collection, '') != 'wenku' AND coalesce(text, '') != ''"
        ).fetchone()[0]
        done = out.execute(
            "SELECT count(*) FROM voyage_embeddings WHERE model=? AND output_dim=?",
            (MODEL, OUTPUT_DIM),
        ).fetchone()[0]
        return total, done


def voyage_embed(api_key: str, texts: list[str]) -> tuple[list[list[float]], dict]:
    payload = {
        "model": MODEL,
        "input": [(t or " ").strip()[:MAX_CHARS] or " " for t in texts],
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
    for attempt in range(24):
        req = urllib.request.Request("https://api.voyageai.com/v1/embeddings", data=body, method="POST", headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            return [item["embedding"] for item in data["data"]], data.get("usage") or {}
        except urllib.error.HTTPError as exc:
            last_error = exc
            detail = exc.read().decode("utf-8", errors="replace")[:240]
            if exc.code != 429:
                raise RuntimeError(f"voyage http {exc.code}: {detail}")
            retry_after = exc.headers.get("Retry-After")
            try:
                wait = int(retry_after) if retry_after else 0
            except ValueError:
                wait = 0
            if wait <= 0:
                wait = min(60 + attempt * 30, 300)
            log(f"rate limited; sleeping {wait}s attempt={attempt + 1}/24")
            time.sleep(wait)
            if STOP:
                raise RuntimeError("stopped during rate-limit sleep")
    raise RuntimeError(f"voyage still rate-limited after retries: {last_error}")


def save_embeddings(rows: list[dict], vectors: list[list[float]]) -> None:
    now = utc_now()
    with sqlite3.connect(OUT_DB) as con:
        for row, vec in zip(rows, vectors):
            con.execute(
                """
                INSERT INTO voyage_embeddings (
                    memory_id, model, output_dim, text_hash, embedding,
                    source_timestamp, source_tag, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(memory_id) DO UPDATE SET
                    model=excluded.model,
                    output_dim=excluded.output_dim,
                    text_hash=excluded.text_hash,
                    embedding=excluded.embedding,
                    source_timestamp=excluded.source_timestamp,
                    source_tag=excluded.source_tag,
                    updated_at=excluded.updated_at
                """,
                (
                    row["memory_id"],
                    MODEL,
                    OUTPUT_DIM,
                    row["text_hash"],
                    pack_vector(vec),
                    row.get("timestamp"),
                    row.get("tag"),
                    now,
                    now,
                ),
            )


def mark_run(run_id: int, status: str, embedded_count: int, note: str = "") -> None:
    with sqlite3.connect(OUT_DB) as con:
        con.execute(
            """
            UPDATE voyage_backfill_runs
            SET finished_at=?, status=?, embedded_count=?, note=?
            WHERE id=?
            """,
            (utc_now(), status, embedded_count, note[:1000], run_id),
        )


def delete_key_file_if_complete() -> None:
    if DELETE_KEY_ON_COMPLETE and KEY_FILE is not None and KEY_FILE.exists():
        try:
            KEY_FILE.unlink()
            log(f"deleted key file {KEY_FILE}")
        except Exception as exc:
            log(f"warning: failed to delete key file {KEY_FILE}: {exc}")


def main() -> int:
    init_out_db()
    api_key = load_key()
    with sqlite3.connect(OUT_DB) as con:
        cur = con.execute(
            """
            INSERT INTO voyage_backfill_runs
                (started_at, status, model, output_dim, batch_size, max_chars)
            VALUES (?, 'running', ?, ?, ?, ?)
            """,
            (utc_now(), MODEL, OUTPUT_DIM, BATCH_SIZE, MAX_CHARS),
        )
        run_id = cur.lastrowid

    start = time.time()
    embedded = 0
    try:
        total, done = counts()
        log(f"start run={run_id} model={MODEL} dim={OUTPUT_DIM} batch={BATCH_SIZE} max_chars={MAX_CHARS} done={done}/{total}")
        while not STOP:
            if time.time() - start > MAX_RUNTIME_SECONDS:
                log("max runtime reached; exiting partial run")
                mark_run(run_id, "partial", embedded, "max runtime reached")
                return 0

            rows = load_pending(BATCH_SIZE)
            if not rows:
                total, done = counts()
                log(f"complete done={done}/{total}")
                mark_run(run_id, "complete", embedded, "all current memories embedded")
                delete_key_file_if_complete()
                return 0

            vectors, usage = voyage_embed(api_key, [r["text"] for r in rows])
            if len(vectors) != len(rows):
                raise RuntimeError(f"embedding count mismatch: {len(vectors)} for {len(rows)} rows")
            save_embeddings(rows, vectors)
            embedded += len(rows)
            total, done = counts()
            usage_s = json.dumps(usage, ensure_ascii=False, separators=(",", ":"))
            log(f"batch saved={len(rows)} run_embedded={embedded} done={done}/{total} usage={usage_s}")
            if done >= total:
                log(f"complete done={done}/{total}")
                mark_run(run_id, "complete", embedded, "all current memories embedded")
                delete_key_file_if_complete()
                return 0
            time.sleep(SUCCESS_SLEEP)

        mark_run(run_id, "stopped", embedded, "signal received")
        return 0
    except Exception as exc:
        log(f"error: {exc}")
        mark_run(run_id, "error", embedded, str(exc))
        return 1


if __name__ == "__main__":
    sys.exit(main())
