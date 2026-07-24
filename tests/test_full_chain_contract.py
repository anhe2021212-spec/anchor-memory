from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import cold_store
import taxonomy_tagger
import update_review
from anchor_memory import AnchorMemory


def configured(tmp_path, monkeypatch):
    monkeypatch.setenv("ANCHOR_DISABLE_CHROMA", "1")
    monkeypatch.setenv("ANCHOR_EMBED_PROVIDER", "local")
    monkeypatch.setenv("ANCHOR_DUAL_EDGE", "on")
    return AnchorMemory(str(tmp_path))


def test_taxonomy_provider_route_and_validated_writeback(tmp_path, monkeypatch):
    memory = configured(tmp_path, monkeypatch)
    memory.integrate(
        "A synthetic observatory event needs automatic coordinates.",
        "raw", memory_id="tag-me", auto_link=False,
    )
    monkeypatch.setattr(
        taxonomy_tagger,
        "_CONFIG",
        SimpleNamespace(
            taxonomy_url="https://taxonomy.example.invalid/v1",
            taxonomy_api_key="configured-by-test",
            taxonomy_model="synthetic-model",
        ),
    )
    assert taxonomy_tagger._routes()[0] == (
        "https://taxonomy.example.invalid/v1",
        "configured-by-test",
        "synthetic-model",
    )
    monkeypatch.setattr(taxonomy_tagger, "DB", str(memory.db.db_path))
    monkeypatch.setattr(
        taxonomy_tagger,
        "_call",
        lambda messages, max_tokens=800: json.dumps(
            {
                "id": "tag-me",
                "tags": ["state:past", "domain:system", "kind:event"],
            }
        ),
    )
    assert taxonomy_tagger.tag_one("tag-me", "synthetic text") is True
    assert memory.db.get("tag-me")["tag"] == "state:past,domain:system,kind:event"


def test_agent_review_gate_approves_level_change(tmp_path, monkeypatch):
    memory = configured(tmp_path, monkeypatch)
    memory.integrate("Synthetic observation.", "raw", memory_id="review-me", auto_link=False)
    queue = tmp_path / "review" / "queue.json"
    monkeypatch.setattr(update_review, "QUEUE_PATH", queue)
    monkeypatch.setattr(update_review, "LOCK_PATH", Path(str(queue) + ".lock"))
    proposal = update_review.propose_level_change(
        "review-me", "understanding", "agent audit", db_path=memory.db.db_path,
    )
    assert memory.db.get("review-me")["level"] == "raw"
    result = update_review.decide(memory, proposal["proposal_id"], "approve", "agent approved")
    assert result["status"] == "approved"
    assert memory.db.get("review-me")["level"] == "understanding"
    assert result["action_result"]["projection_pending"] is True
    with memory.db._conn() as connection:
        pending = connection.execute(
            "SELECT COUNT(*) FROM embedding_outbox WHERE memory_id='review-me'"
        ).fetchone()[0]
    assert pending == 1


def test_chat_history_cjk_fts_and_bounded_dialogue_window(tmp_path, monkeypatch):
    relay = tmp_path / "relay.sqlite3"
    index = tmp_path / "chat-index.sqlite3"
    with sqlite3.connect(relay) as connection:
        connection.execute(
            "CREATE TABLE messages (id INTEGER PRIMARY KEY, ts TEXT, direction TEXT, "
            "kind TEXT, text TEXT, meta TEXT)"
        )
        connection.executemany(
            "INSERT INTO messages(id,ts,direction,kind,text,meta) VALUES(?,?,?,?,?,?)",
            [
                (1, "2020-01-01T00:00:00+00:00", "in", "user", "蓝色观测站正在校准。", "{}"),
                (2, "2020-01-01T00:01:00+00:00", "out", "reply", "收到，保持校准窗口。", "{}"),
                (3, "2020-01-01T03:00:00+00:00", "in", "user", "不相关的远处消息。", "{}"),
            ],
        )
    monkeypatch.setattr(cold_store, "RELAY_DB", relay)
    monkeypatch.setattr(cold_store, "INDEX_DB", index)
    monkeypatch.setattr(cold_store, "ALIASES_FILE", tmp_path / "missing-aliases.json")
    sync = cold_store.sync_index(rebuild=True)
    assert sync["ok"] is True
    results = cold_store.cold_search("蓝色观测站", limit=1, min_age_minutes=0)
    assert len(results) == 1
    assert "蓝色观测站" in results[0]["snippet"]
    assert "User:" in results[0]["snippet"]
    assert "Agent:" in results[0]["snippet"]
    assert "远处消息" not in results[0]["snippet"]


def test_maintenance_dry_run_and_failure_exit_codes(tmp_path, monkeypatch):
    healthy_dir = tmp_path / "healthy"
    healthy_dir.mkdir()
    memory = configured(healthy_dir, monkeypatch)
    memory.integrate("Synthetic maintenance record.", "raw", memory_id="maintain", auto_link=False)
    script = Path(__file__).parents[1] / "src" / "maintenance.py"
    healthy = subprocess.run(
        [
            sys.executable,
            str(script),
            "--db",
            str(memory.db.db_path),
            "--only",
            "trash,events,edges,tier,fts",
            "--json",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert healthy.returncode == 0, healthy.stderr
    report = json.loads(healthy.stdout)
    assert report["dry_run"] is True
    assert not [row for row in report["results"] if row.get("error")]

    broken = subprocess.run(
        [sys.executable, str(script), "--db", str(tmp_path / "empty.db"), "--only", "trash", "--json"],
        text=True,
        capture_output=True,
        check=False,
    )
    assert broken.returncode == 1
    assert json.loads(broken.stdout)["results"][0]["error"]
