from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import dual_edge
import recall_v2
from anchor_memory import AnchorMemory


def test_real_chroma_is_used_by_live_seed_path(tmp_path, monkeypatch):
    monkeypatch.delenv("ANCHOR_DISABLE_CHROMA", raising=False)
    monkeypatch.setenv("ANCHOR_EMBED_PROVIDER", "local")
    monkeypatch.setenv("ANCHOR_DUAL_EDGE", "on")
    memory = AnchorMemory(str(tmp_path))
    query = "Synthetic exact vector phrase."
    memory.integrate(query, "raw", memory_id="vector-a", auto_link=False)
    rows, status = recall_v2._seed_candidates(
        memory, query, memory._encode_query(query), 5, None,
    )
    assert memory._collection.count() == 1
    assert rows[0]["memory_id"] == "vector-a"
    assert rows[0]["distance"] is not None
    assert status == "unavailable"


def test_real_chroma_drives_auto_link_proposals(tmp_path, monkeypatch):
    monkeypatch.delenv("ANCHOR_DISABLE_CHROMA", raising=False)
    monkeypatch.setenv("ANCHOR_EMBED_PROVIDER", "local")
    monkeypatch.setenv("ANCHOR_DUAL_EDGE", "on")
    memory = AnchorMemory(str(tmp_path))
    memory.integrate("Synthetic same-topic record.", "raw", memory_id="first", auto_link=False)
    result = memory.integrate("Synthetic same-topic record.", "raw", memory_id="second")
    with memory.db._conn() as connection:
        flow_count = connection.execute("SELECT count(*) FROM flow_edges").fetchone()[0]
    assert result["flow_edges_created"] == 2
    assert flow_count == 2


def test_real_kuzu_rebuilds_from_sqlite_without_memory_text(tmp_path, monkeypatch):
    monkeypatch.setenv("ANCHOR_DISABLE_CHROMA", "1")
    monkeypatch.setenv("ANCHOR_EMBED_PROVIDER", "local")
    monkeypatch.setenv("ANCHOR_DUAL_EDGE", "on")
    memory = AnchorMemory(str(tmp_path))
    memory.integrate("Synthetic source.", "raw", memory_id="source", auto_link=False)
    memory.integrate(
        "Synthetic target.", "raw", memory_id="target",
        connect_to=["source"], auto_link=False,
    )
    assert memory.db.kuzu_available
    dual_edge.bootstrap_kuzu(memory.db)
    nodes = memory.db._kuzu_rows("MATCH (m:Memory) RETURN m.memory_id")
    flow = memory.db._kuzu_rows(
        "MATCH (a:Memory)-[e:FlowEdge]->(b:Memory) RETURN a.memory_id,b.memory_id"
    )
    assert sorted(row[0] for row in nodes) == ["source", "target"]
    assert len(flow) == 2


def test_fresh_projection_rebuild_command(tmp_path, monkeypatch):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    monkeypatch.setenv("ANCHOR_DISABLE_CHROMA", "1")
    monkeypatch.setenv("ANCHOR_EMBED_PROVIDER", "local")
    monkeypatch.setenv("ANCHOR_KUZU_PATH", str(tmp_path / "source-kuzu"))
    memory = AnchorMemory(str(source_dir))
    memory.integrate("Synthetic first record.", "raw", memory_id="a", auto_link=False)
    memory.integrate("Synthetic second record.", "raw", memory_id="b", auto_link=False)

    output = tmp_path / "rebuilt"
    script = Path(__file__).parents[1] / "scripts" / "rebuild_projections.py"
    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--source-db",
            str(source_dir / "memories.db"),
            "--output-dir",
            str(output),
        ],
        env=os.environ.copy(),
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    evidence = json.loads((output / "projection-metadata.json").read_text())
    assert evidence["status"] == "ok"
    assert evidence["sqlite_count"] == 2
    assert evidence["sqlite_count"] == evidence["chroma_count"] == evidence["kuzu_count"]
