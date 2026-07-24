"""Read-only recall over raw chat dialogue.

The relay database is always opened with SQLite ``mode=ro``.  Search state and
FTS data live in a separate database, so this module cannot alter relay.db.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sqlite3
import time
from contextlib import closing
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

from release_config import AnchorConfig

try:
    import jieba
except ImportError:  # pragma: no cover - deployment has jieba; fallback keeps failure isolated
    jieba = None


_CONFIG = AnchorConfig.load()
RELAY_DB = Path(os.environ.get("ANCHOR_CHAT_HISTORY_PATH", _CONFIG.chat_history_path or (_CONFIG.data_dir / "chat-history.sqlite3")))
INDEX_DB = Path(os.environ.get("ANCHOR_CHAT_INDEX_PATH", _CONFIG.data_dir / "chat-index.sqlite3"))
ALIASES_FILE = Path(os.environ.get("ANCHOR_ALIASES_FILE", _CONFIG.data_dir / "aliases.json"))
SYNC_BATCH = max(1, int(os.environ.get("COLD_STORE_SYNC_BATCH", "200")))
ALLOWED_KINDS = ("user", "voice", "reply")
MAX_NEIGHBOR_MESSAGES = 6
MAX_SNIPPET_CHARS = 1000
NEIGHBOR_GAP_SECONDS = 20 * 60

log = logging.getLogger("cold_store")

_QUOTED_RE = re.compile(r"[\"“”「『](.+?)[\"“”」』]")
_CODE_TOKEN_RE = re.compile(
    r"(?<![\w.-])(?:[\w@-]+(?:[./\\:_-][\w@.-]+)+|[\w@-]+\.[A-Za-z0-9]{1,12})(?![\w.-])",
    re.UNICODE,
)
_QUERY_STOPWORDS = {
    "老公", "老婆", "宝贝", "宝宝", "亲爱的",
    "你", "我", "他", "她", "它", "我们", "你们", "他们", "她们",
    "记得", "想起", "想起来", "说过", "提过", "聊过", "讲过",
    "以前", "之前", "上次", "当时", "那次", "刚才",
    "是不是", "有没有", "什么", "怎么", "为什么", "哪个", "哪里",
    "帮我", "请问", "一下", "这个", "那个", "这些", "那些",
    "的", "了", "吗", "嘛", "么", "呢", "啊", "呀", "吧", "哦", "喔",
}


def _relay_connection() -> sqlite3.Connection:
    uri = f"file:{RELAY_DB.resolve().as_posix()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True, timeout=3)
    conn.row_factory = sqlite3.Row
    return conn


def _index_connection() -> sqlite3.Connection:
    INDEX_DB.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(INDEX_DB), timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


def _create_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS cold_state (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS cold_docs (
            message_id INTEGER PRIMARY KEY,
            literal_text TEXT NOT NULL
        );
        CREATE VIRTUAL TABLE IF NOT EXISTS cold_fts USING fts5(
            message_id UNINDEXED,
            tokenized,
            tokenize='unicode61 remove_diacritics 2'
        );
        """
    )


def _tokenize(text: str) -> str:
    text = (text or "").strip()
    if not text:
        return ""
    if jieba is None:
        # Import failure must never break Anchor.  This fallback is useful only
        # for ASCII/code tokens; deployment validation requires jieba.
        return " ".join(re.findall(r"[\w.-]+", text.casefold(), re.UNICODE))
    return " ".join(token.strip().casefold() for token in jieba.cut_for_search(text) if token.strip())


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip()).casefold()


def sync_index(rebuild: bool = False) -> dict:
    """Backfill or incrementally advance the independent cold-store index.

    Incremental calls inspect at most ``SYNC_BATCH`` relay rows.  State advances
    over all kinds, while only user/voice/reply rows are indexed, preventing an
    internal-message tail from being rescanned forever.
    """
    started = time.monotonic()
    indexed = scanned = 0
    try:
        if rebuild and INDEX_DB.exists():
            INDEX_DB.unlink()
            for suffix in ("-wal", "-shm"):
                sidecar = Path(str(INDEX_DB) + suffix)
                if sidecar.exists():
                    sidecar.unlink()
        with closing(_relay_connection()) as relay, closing(_index_connection()) as idx:
            _create_schema(idx)
            row = idx.execute("SELECT value FROM cold_state WHERE key='last_seen_id'").fetchone()
            last_seen = 0 if rebuild or row is None else int(row[0])
            batch_size = 5000 if rebuild else SYNC_BATCH
            while True:
                rows = relay.execute(
                    "SELECT id, kind, text FROM messages WHERE id > ? ORDER BY id LIMIT ?",
                    (last_seen, batch_size),
                ).fetchall()
                if not rows:
                    break
                scanned += len(rows)
                for message in rows:
                    if message["kind"] not in ALLOWED_KINDS or not (message["text"] or "").strip():
                        continue
                    mid = int(message["id"])
                    text = message["text"]
                    idx.execute(
                        "INSERT INTO cold_docs(message_id,literal_text) VALUES(?,?) "
                        "ON CONFLICT(message_id) DO UPDATE SET literal_text=excluded.literal_text",
                        (mid, _normalise(text)),
                    )
                    idx.execute("DELETE FROM cold_fts WHERE message_id=?", (mid,))
                    idx.execute(
                        "INSERT INTO cold_fts(message_id,tokenized) VALUES(?,?)",
                        (mid, _tokenize(text)),
                    )
                    indexed += 1
                last_seen = int(rows[-1]["id"])
                idx.execute(
                    "INSERT INTO cold_state(key,value) VALUES('last_seen_id',?) "
                    "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                    (str(last_seen),),
                )
                idx.commit()
                if not rebuild:
                    break
            count = idx.execute("SELECT count(*) FROM cold_docs").fetchone()[0]
        return {
            "ok": True,
            "rebuild": rebuild,
            "scanned": scanned,
            "indexed": indexed,
            "last_seen_id": last_seen,
            "index_count": count,
            "elapsed_ms": round((time.monotonic() - started) * 1000),
        }
    except Exception as exc:
        log.warning("cold index sync failed: %s", type(exc).__name__)
        return {
            "ok": False,
            "rebuild": rebuild,
            "scanned": scanned,
            "indexed": indexed,
            "elapsed_ms": round((time.monotonic() - started) * 1000),
            "error": type(exc).__name__,
        }


def _salient_terms(query: str) -> list[str]:
    """Extract content-bearing query terms; vocatives/recall scaffolding are noise."""
    terms: list[str] = []
    terms.extend(match.group(1).strip() for match in _QUOTED_RE.finditer(query or ""))
    terms.extend(match.group(0).strip() for match in _CODE_TOKEN_RE.finditer(query or ""))
    for token in _tokenize(query).split():
        token = token.strip("，。！？!?；;：:、（）()【】[]‘’'\"“”").casefold()
        if token in _QUERY_STOPWORDS or len(token) < 2 or token.isdigit():
            continue
        terms.append(token)
    out = []
    seen = set()
    for term in terms:
        folded = _normalise(term)
        if len(folded) < 2 or folded in seen or folded in _QUERY_STOPWORDS:
            continue
        seen.add(folded)
        out.append(folded)
    # jieba cut_for_search can emit both a long term and its substring.  Keep
    # the more specific long form so generic fragments cannot crowd it out.
    return [term for term in out if not any(term != other and term in other for other in out)][:12]


def _literal_terms(query: str) -> list[str]:
    terms: list[str] = []
    qfold = (query or "").casefold()
    try:
        data = json.loads(ALIASES_FILE.read_text(encoding="utf-8"))
        for person in (data.get("people") or {}).values():
            for term in [person.get("canonical"), *(person.get("aliases") or [])]:
                if term and str(term).casefold() in qfold:
                    terms.append(str(term))
        for term in (data.get("slang") or {}):
            if str(term).casefold() in qfold:
                terms.append(str(term))
    except Exception as exc:
        log.warning("cold aliases load failed: %s", type(exc).__name__)
    terms.extend(match.group(1).strip() for match in _QUOTED_RE.finditer(query or ""))
    terms.extend(match.group(0).strip() for match in _CODE_TOKEN_RE.finditer(query or ""))
    # Plain content words are also literal evidence.  Previously only aliases,
    # slang, quotes and code tokens entered this lane, so an explicit rare word
    # such as "胡渣" was diluted by generic FTS hits from "老公/记得".
    terms.extend(_salient_terms(query))
    out = []
    seen = set()
    for term in terms:
        folded = _normalise(term)
        if len(folded) < 2 or folded in seen:
            continue
        seen.add(folded)
        out.append(folded)
    return out[:12]


def _fts_expression(query: str) -> str:
    tokens = []
    seen = set()
    for token in _salient_terms(query):
        token = token.strip().replace('"', '""')
        if len(token) < 2 or token in seen:
            continue
        seen.add(token)
        tokens.append(f'"{token}"')
    return " OR ".join(tokens[:24])


def _parse_ts(value: str) -> datetime | None:
    try:
        parsed = datetime.fromisoformat((value or "").replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    except (TypeError, ValueError):
        return None


def _fetch_messages(relay: sqlite3.Connection, ids: Iterable[int]) -> dict[int, sqlite3.Row]:
    ids = list(dict.fromkeys(int(i) for i in ids))
    if not ids:
        return {}
    placeholders = ",".join("?" for _ in ids)
    rows = relay.execute(
        f"SELECT id,ts,direction,kind,text,meta FROM messages WHERE id IN ({placeholders})",
        ids,
    ).fetchall()
    return {int(row["id"]): row for row in rows}


def _row_meta(row: sqlite3.Row) -> dict:
    try:
        value = json.loads(row["meta"] or "{}")
        return value if isinstance(value, dict) else {}
    except (TypeError, ValueError):
        return {}


def _bounded_rows(rows: list[sqlite3.Row], hit_id: int) -> list[sqlite3.Row]:
    rows = sorted({int(row["id"]): row for row in rows}.values(), key=lambda row: int(row["id"]))
    rows.sort(key=lambda row: (abs(int(row["id"]) - hit_id), int(row["id"])))
    rows = rows[:MAX_NEIGHBOR_MESSAGES]
    return sorted(rows, key=lambda row: int(row["id"]))


def _dialogue_neighborhood(
    relay: sqlite3.Connection, hit: sqlite3.Row, cutoff: datetime, exact_query: str
) -> tuple[str, set[int]]:
    hit_id = int(hit["id"])
    hit_meta = _row_meta(hit)
    nearby = relay.execute(
        "SELECT id,ts,direction,kind,text,meta FROM messages "
        "WHERE id BETWEEN ? AND ? AND kind IN ('user','voice','reply') ORDER BY id",
        (max(1, hit_id - 40), hit_id + 40),
    ).fetchall()
    nearby = [
        row for row in nearby
        if (_parse_ts(row["ts"]) and _parse_ts(row["ts"]) <= cutoff
            and _normalise(row["text"]) != exact_query)
    ]

    chosen: list[sqlite3.Row] = []
    api_session = hit_meta.get("api_session")
    if api_session:
        chosen = [row for row in nearby if _row_meta(row).get("api_session") == api_session]
    if not chosen:
        reply_ids = {hit_id}
        reply_to = hit_meta.get("reply_to")
        if str(reply_to or "").isdigit():
            reply_ids.add(int(reply_to))
        for row in nearby:
            candidate = _row_meta(row).get("reply_to")
            if str(candidate or "").isdigit() and int(candidate) == hit_id:
                reply_ids.add(int(row["id"]))
        if len(reply_ids) > 1:
            chosen = list(_fetch_messages(relay, reply_ids).values())
            chosen = [
                row for row in chosen
                if (_parse_ts(row["ts"]) and _parse_ts(row["ts"]) <= cutoff
                    and _normalise(row["text"]) != exact_query)
            ]
    if not chosen:
        # Grow around the hit only while consecutive dialogue remains within 20 minutes.
        ordered = list(nearby)
        position = next((i for i, row in enumerate(ordered) if int(row["id"]) == hit_id), -1)
        if position >= 0:
            left = right = position
            chosen = [ordered[position]]
            while len(chosen) < MAX_NEIGHBOR_MESSAGES and (left > 0 or right + 1 < len(ordered)):
                options = []
                if left > 0:
                    gap = abs((_parse_ts(ordered[left]["ts"]) - _parse_ts(ordered[left - 1]["ts"])).total_seconds()) if _parse_ts(ordered[left]["ts"]) and _parse_ts(ordered[left - 1]["ts"]) else 10**9
                    if gap <= NEIGHBOR_GAP_SECONDS:
                        options.append((abs(int(ordered[left - 1]["id"]) - hit_id), "left"))
                if right + 1 < len(ordered):
                    gap = abs((_parse_ts(ordered[right + 1]["ts"]) - _parse_ts(ordered[right]["ts"])).total_seconds()) if _parse_ts(ordered[right + 1]["ts"]) and _parse_ts(ordered[right]["ts"]) else 10**9
                    if gap <= NEIGHBOR_GAP_SECONDS:
                        options.append((abs(int(ordered[right + 1]["id"]) - hit_id), "right"))
                if not options:
                    break
                _, side = min(options)
                if side == "left":
                    left -= 1
                    chosen.insert(0, ordered[left])
                else:
                    right += 1
                    chosen.append(ordered[right])
    chosen = _bounded_rows(chosen or [hit], hit_id)
    lines = []
    used_ids = set()
    total = 0
    for row in chosen:
        role = "User" if row["direction"] == "in" else "Agent"
        stamp = (row["ts"] or "")[:19].replace("T", " ")
        text = re.sub(r"\s+", " ", row["text"] or "").strip()
        line = f"[{stamp}] {role}: {text}"
        remaining = MAX_SNIPPET_CHARS - total
        if remaining <= 0:
            break
        if len(line) > remaining:
            line = line[: max(0, remaining - 1)] + "…"
        lines.append(line)
        total += len(line) + 1
        used_ids.add(int(row["id"]))
    return "\n".join(lines), used_ids


def _bookmark_matches(relay: sqlite3.Connection, query: str, limit: int) -> list[dict]:
    """Return locally starred messages before ordinary cold-store candidates."""
    try:
        rows = relay.execute(
            "SELECT message_key,text,sender,ts,bookmarked_at "
            "FROM bookmarks ORDER BY bookmarked_at DESC"
        ).fetchall()
    except sqlite3.OperationalError:
        return []
    query_norm = _normalise(query)
    terms = list(dict.fromkeys(_literal_terms(query) + _salient_terms(query)))
    found = []
    for row in rows:
        text = row["text"] or ""
        norm = _normalise(text)
        matched = [term for term in terms if _normalise(term) in norm]
        if query_norm not in norm and not matched:
            continue
        role = "User" if row["sender"] == "user" else "Agent"
        stamp = (row["ts"] or "")[:19].replace("T", " ")
        compact = re.sub(r"\s+", " ", text).strip()
        snippet = f"[{stamp}] {role}: {compact}"
        if len(snippet) > MAX_SNIPPET_CHARS:
            snippet = snippet[: MAX_SNIPPET_CHARS - 1] + "…"
        key = str(row["message_key"])
        found.append({
            "memory_id": f"relay:{key}",
            "source": "bookmark",
            "evidence_role": "raw_dialogue",
            "timestamp": row["ts"],
            "tag": "⭐收藏·历史聊天原话",
            "snippet": snippet,
            "score": round(2.0 + min(0.5, 0.05 * len(matched)), 4),
            "matched_terms": sorted(matched, key=lambda term: (-len(term), term))[:8],
            "match_type": "bookmark",
            "bookmarked": True,
        })
        if len(found) >= limit:
            break
    return found


def cold_search(query: str, limit: int = 3, min_age_minutes: int = 30) -> list[dict]:
    """Return ranked raw-dialogue evidence, with starred messages first."""
    query = (query or "").strip()
    limit = max(0, min(int(limit), 10))
    if not query or not limit:
        return []
    try:
        with closing(_relay_connection()) as relay:
            bookmark_results = _bookmark_matches(relay, query, limit)
        if len(bookmark_results) >= limit:
            return bookmark_results
        sync = sync_index(rebuild=False)
        if not sync.get("ok"):
            return bookmark_results
        scored: dict[int, dict] = {}

        def _offer(mid: int, score: float, match_type: str, matched_terms: list[str]) -> None:
            item = scored.setdefault(mid, {"score": 0.0, "types": set(), "terms": set()})
            item["score"] = max(float(item["score"]), float(score))
            item["types"].add(match_type)
            item["terms"].update(term for term in matched_terms if term)

        with closing(_index_connection()) as idx:
            _create_schema(idx)
            salient = _salient_terms(query)
            expression = _fts_expression(query)
            if expression:
                fts_rows = idx.execute(
                    "SELECT CAST(cold_fts.message_id AS INTEGER) AS message_id, "
                    "bm25(cold_fts) AS rank, cold_docs.literal_text AS literal_text "
                    "FROM cold_fts JOIN cold_docs "
                    "ON cold_docs.message_id=CAST(cold_fts.message_id AS INTEGER) "
                    "WHERE cold_fts MATCH ? ORDER BY rank LIMIT 80",
                    (expression,),
                ).fetchall()
                denom = max(1, len(fts_rows) - 1)
                for position, row in enumerate(fts_rows):
                    matched = [term for term in salient if term in row["literal_text"]]
                    coverage = len(matched) / max(1, len(salient))
                    # Preserve FTS order instead of flattening nearly every hit
                    # to 0.70; exact literal evidence below remains stronger.
                    score = 0.62 + 0.16 * (1 - position / denom) + 0.04 * coverage
                    _offer(int(row["message_id"]), min(0.84, score), "fts", matched)
            terms = _literal_terms(query)
            if terms:
                clauses = " OR ".join("literal_text LIKE ? ESCAPE '\\'" for _ in terms)
                params = ["%" + term.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%" for term in terms]
                for row in idx.execute(
                    f"SELECT message_id,literal_text FROM cold_docs WHERE {clauses} LIMIT 120",
                    params,
                ):
                    mid = int(row["message_id"])
                    matched = [term for term in terms if term in row["literal_text"]]
                    _offer(mid, min(0.98, 0.92 + 0.02 * len(matched)), "literal", matched)
        if not scored:
            return bookmark_results

        with closing(_relay_connection()) as relay:
            messages = _fetch_messages(relay, scored.keys())
            cutoff = datetime.now(timezone.utc) - timedelta(minutes=max(0, min_age_minutes))
            exact = _normalise(query)
            eligible = []
            for mid, evidence in scored.items():
                row = messages.get(mid)
                stamp = _parse_ts(row["ts"]) if row else None
                if not row or not stamp or stamp > cutoff or _normalise(row["text"]) == exact:
                    continue
                eligible.append((mid, float(evidence["score"]), stamp, evidence))
            eligible.sort(key=lambda item: (-item[1], -item[2].timestamp(), -item[0]))

            results = list(bookmark_results)
            covered_ids: set[int] = set()
            for item in bookmark_results:
                key = str(item.get("memory_id") or "").removeprefix("relay:")
                if key.isdigit():
                    covered_ids.add(int(key))
            for mid, score, _, evidence in eligible:
                if mid in covered_ids:
                    continue
                snippet, neighborhood_ids = _dialogue_neighborhood(relay, messages[mid], cutoff, exact)
                if not snippet:
                    continue
                results.append({
                    "memory_id": f"relay:{mid}",
                    "source": "cold_store",
                    "evidence_role": "raw_dialogue",
                    "timestamp": messages[mid]["ts"],
                    "tag": "冷库·历史聊天原话",
                    "snippet": snippet,
                    "score": round(float(score), 4),
                    "matched_terms": sorted(evidence["terms"], key=lambda term: (-len(term), term))[:8],
                    "match_type": "+".join(sorted(evidence["types"])),
                })
                covered_ids.update(neighborhood_ids)
                # Adjacent hits are duplicates even if a constrained structural
                # neighborhood happened not to contain both ids.
                covered_ids.update(range(max(1, mid - 2), mid + 3))
                if len(results) >= limit:
                    break
            return results
    except Exception as exc:
        # Never include query or dialogue text in logs.
        log.warning("cold search failed: %s", type(exc).__name__)
        return locals().get("bookmark_results", [])


def _main() -> int:
    parser = argparse.ArgumentParser(description="Build the raw-dialogue cold-store index")
    parser.add_argument("--rebuild", action="store_true", help="discard and fully rebuild the independent index")
    args = parser.parse_args()
    result = sync_index(rebuild=args.rebuild)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(_main())
