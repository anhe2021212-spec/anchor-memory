"""Backfill ``collection='wenku'`` into the independent Theseus shadow index."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sqlite3
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests
from release_config import AnchorConfig

from theseus_shadow_index import (
    COLLECTION_NAME,
    MODEL,
    SPEC_VERSION,
    ShadowValidationError,
    audit_counts,
    is_parent_current,
    parse_model_output,
    normalize_model_chunks,
    record_failure,
    replace_parent_chunks,
    source_hash,
    validate_chunks,
)


ROOT = Path(__file__).resolve().parent
_CONFIG = AnchorConfig.load()
DEFAULT_DB = Path(os.environ.get("THESEUS_SHADOW_DB_PATH", _CONFIG.db_path))
DEFAULT_CHROMA = Path(os.environ.get("THESEUS_SHADOW_CHROMA_PATH", _CONFIG.chroma_dir or (_CONFIG.data_dir / "chroma")))
DEFAULT_PROMPT = Path(os.environ.get("THESEUS_SHADOW_PROMPT", ROOT / "theseus_shadow_prompt_v3.txt"))
API_URL = os.environ.get("THESEUS_SHADOW_API_URL", "https://api.example.invalid/v1")
API_MODEL = os.environ.get("THESEUS_SHADOW_API_MODEL", MODEL)


LOG = logging.getLogger("theseus-shadow-backfill")


class ModelCallError(RuntimeError):
    pass


@dataclass
class Stats:
    scanned: int = 0
    skipped: int = 0
    succeeded: int = 0
    failed: int = 0
    chunks: int = 0
    indexed: int = 0
    format_warnings: int = 0
    deterministic_repairs: int = 0


class ShadowModelClient:
    def __init__(
        self,
        spec: str,
        *,
        url: str = API_URL,
        model: str = API_MODEL,
        api_key: str | None = None,
        timeout: float = 240,
        retries: int = 2,
    ):
        self.spec = spec
        self.url = url
        self.model = model
        self.api_key = api_key or os.environ.get("THESEUS_SHADOW_API_KEY", "")
        if not self.api_key:
            raise RuntimeError("THESEUS_SHADOW_API_KEY is required")
        self.timeout = timeout
        self.retries = retries
        self.session = requests.Session()

    def _call_once(self, source_text: str) -> str:
        try:
            response = self.session.post(
                self.url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model,
                    "temperature": 0,
                    "max_tokens": 12000,
                    "messages": [
                        {"role": "system", "content": self.spec},
                        {"role": "user", "content": source_text},
                    ],
                },
                timeout=self.timeout,
            )
            response.raise_for_status()
            payload = response.json()
            content = payload["choices"][0]["message"]["content"]
            if not isinstance(content, str):
                raise ModelCallError("response content is not a string")
            return content
        except (requests.RequestException, ValueError, KeyError, TypeError, IndexError) as exc:
            raise ModelCallError("model HTTP or response-shape failure") from exc

    def generate_validated(
        self, source_text: str
    ) -> tuple[list[dict[str, Any]], int, bool, dict[str, int]]:
        last_error: Exception | None = None
        for attempt in range(1, self.retries + 2):
            try:
                raw = self._call_once(source_text)
                chunks, warned = parse_model_output(raw)
                chunks, repairs = normalize_model_chunks(chunks)
                return validate_chunks(source_text, chunks), attempt, warned, repairs
            except (ModelCallError, ShadowValidationError) as exc:
                last_error = exc
                if attempt <= self.retries:
                    time.sleep(min(2 ** (attempt - 1), 4))
        assert last_error is not None
        raise last_error


def load_memories(db_path: Path, memory_ids: list[str], limit: int | None) -> list[sqlite3.Row]:
    conn = sqlite3.connect(str(db_path), timeout=30)
    conn.row_factory = sqlite3.Row
    try:
        sql = "SELECT memory_id, text, tag, timestamp FROM memories WHERE collection='wenku'"
        params: list[Any] = []
        if memory_ids:
            sql += " AND memory_id IN (" + ",".join("?" for _ in memory_ids) + ")"
            params.extend(memory_ids)
        sql += " ORDER BY timestamp, memory_id"
        if limit is not None:
            sql += " LIMIT ?"
            params.append(limit)
        return conn.execute(sql, params).fetchall()
    finally:
        conn.close()


def make_collection(chroma_path: Path, collection_name: str):
    import chromadb

    client = chromadb.PersistentClient(path=str(chroma_path))
    return client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )


def make_document_embedder():
    # Reuse the deployed Voyage client and its exact document input_type contract.
    from anchor_memory import VoyageEmbedder

    return VoyageEmbedder().encode


def run(args: argparse.Namespace) -> int:
    prompt = args.prompt.read_text(encoding="utf-8").strip()
    client = ShadowModelClient(
        prompt,
        url=args.api_url,
        model=args.model,
        timeout=args.timeout,
        retries=args.retries,
    )
    # ``--limit`` is a processing budget, not a SQL scan limit.  Applying it
    # before the current-version check would make cron inspect the same oldest
    # 50 rows forever and never reach newly appended wenku entries.
    rows = load_memories(args.db, args.memory_id, None)
    stats = Stats()
    collection = None
    embed_documents = None
    if not args.dry_run:
        collection = make_collection(args.chroma, args.collection)
        embed_documents = make_document_embedder()

    attempted = 0
    for row in rows:
        stats.scanned += 1
        memory_id = str(row["memory_id"])
        text = row["text"] or ""
        digest = source_hash(text)
        if (
            not args.dry_run
            and not args.force
            and is_parent_current(args.db, memory_id, digest, args.spec_version, args.model)
        ):
            stats.skipped += 1
            LOG.debug("skip memory_id=%s hash=%s reason=current", memory_id, digest[:12])
            continue
        if args.limit is not None and attempted >= args.limit:
            break
        attempted += 1
        started = time.monotonic()
        attempts = args.retries + 1
        try:
            chunks, attempts, warned, repairs = client.generate_validated(text)
            if warned:
                stats.format_warnings += 1
                LOG.warning("format_warning memory_id=%s hash=%s type=markdown_fence", memory_id, digest[:12])
            if repairs:
                stats.deterministic_repairs += sum(repairs.values())
                repair_summary = ",".join(f"{key}:{repairs[key]}" for key in sorted(repairs))
                LOG.warning(
                    "normalization_warning memory_id=%s hash=%s repairs=%s",
                    memory_id,
                    digest[:12],
                    repair_summary,
                )
            if args.dry_run:
                result = {
                    "chunks": len(chunks),
                    "indexed": sum(c["index_policy"] == "index" for c in chunks),
                }
            else:
                result = replace_parent_chunks(
                    args.db,
                    collection,
                    embed_documents,
                    memory_id,
                    text,
                    chunks,
                    spec_version=args.spec_version,
                    model=args.model,
                )
            stats.succeeded += 1
            stats.chunks += result["chunks"]
            stats.indexed += result["indexed"]
            LOG.info(
                "ok memory_id=%s hash=%s attempts=%d chunks=%d indexed=%d elapsed_ms=%d",
                memory_id,
                digest[:12],
                attempts,
                result["chunks"],
                result["indexed"],
                int((time.monotonic() - started) * 1000),
            )
        except (ModelCallError, ShadowValidationError, RuntimeError) as exc:
            stats.failed += 1
            error_type = type(exc).__name__
            if not args.dry_run:
                record_failure(
                    args.db,
                    memory_id,
                    digest,
                    error_type,
                    attempts,
                    args.spec_version,
                    args.model,
                )
            LOG.error(
                "failed memory_id=%s hash=%s error_type=%s attempts=%d elapsed_ms=%d",
                memory_id,
                digest[:12],
                error_type,
                attempts,
                int((time.monotonic() - started) * 1000),
            )

    summary: dict[str, Any] = vars(stats)
    if collection is not None:
        summary["audit"] = audit_counts(args.db, collection)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    if stats.failed or (collection is not None and not summary["audit"]["consistent"]):
        return 1
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--chroma", type=Path, default=DEFAULT_CHROMA)
    parser.add_argument("--collection", default=os.environ.get("THESEUS_SHADOW_COLLECTION", COLLECTION_NAME))
    parser.add_argument("--prompt", type=Path, default=DEFAULT_PROMPT)
    parser.add_argument("--api-url", default=API_URL)
    parser.add_argument("--model", default=API_MODEL)
    parser.add_argument("--spec-version", default=os.environ.get("THESEUS_SHADOW_SPEC_VERSION", SPEC_VERSION))
    parser.add_argument("--timeout", type=float, default=240)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--memory-id", action="append", default=[])
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="call and validate the model, but do not embed or write")
    parser.add_argument("--log-level", default="INFO")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    if args.retries < 0 or args.retries > 2:
        raise SystemExit("--retries must be between 0 and 2")
    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit must be positive")
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
