from __future__ import annotations

import numpy as np
import dual_edge
import propose_links
import recall_v2
from anchor_memory import AnchorMemory


def configured(tmp_path, monkeypatch):
    monkeypatch.setenv("ANCHOR_DISABLE_CHROMA", "1")
    monkeypatch.setenv("ANCHOR_EMBED_PROVIDER", "local")
    monkeypatch.setenv("ANCHOR_DUAL_EDGE", "on")
    return AnchorMemory(str(tmp_path))


def test_sqlite_authority_and_candidate_recall_zero_write(tmp_path, monkeypatch):
    memory = configured(tmp_path, monkeypatch)
    memory.integrate(
        "The fictional observatory calibrated its blue sensor array.",
        "raw", memory_id="demo-a", auto_link=False,
    )
    memory.integrate(
        "The calibration reduced synthetic navigation noise.",
        "raw", memory_id="demo-b", connect_to=["demo-a"], auto_link=False,
    )
    before = memory.db.get("demo-a")["activation_score"]
    recalled = memory.recall("blue sensor calibration", budget=3)
    after = memory.db.get("demo-a")["activation_score"]
    assert memory.count() == 2
    assert recalled["results"]
    assert after == before


def test_auto_link_degrades_without_chroma_projection(tmp_path, monkeypatch):
    memory = configured(tmp_path, monkeypatch)
    memory.integrate("Synthetic same topic.", "raw", memory_id="first", auto_link=False)
    result = memory.integrate("Synthetic same topic.", "raw", memory_id="second")
    assert memory.count() == 2
    assert result["memory_id"] == "second"


def test_exported_bounded_cluster_kernel_is_callable():
    items = [
        {
            "id": "first",
            "timestamp": "2026-01-01T00:00:00+00:00",
            "embedding": np.asarray([1.0, 0.0], dtype=np.float32),
        },
        {
            "id": "second",
            "timestamp": "2026-01-02T00:00:00+00:00",
            "embedding": np.asarray([1.0, 0.0], dtype=np.float32),
        },
    ]
    clusters = propose_links.bounded_clusters(items)
    assert [[item["id"] for item in group] for group in clusters] == [["first", "second"]]


def test_final_injection_heat_is_idempotent_and_semantic_isolated(tmp_path, monkeypatch):
    memory = configured(tmp_path, monkeypatch)
    for mid, text, level in (
        ("a", "Synthetic blue sensor event.", "raw"),
        ("b", "Synthetic navigation consequence.", "raw"),
        ("u", "Synthetic understanding.", "understanding"),
    ):
        memory.integrate(text, level, memory_id=mid, auto_link=False)
    memory.db.write_flow_edge("a", "b", 0.8, 0.5, "test", mode="manual")
    memory.db.write_semantic_edge(
        "u", "a", "SUPPORTED_BY", strength=1.0, conductance=0.0,
        confidence=1.0, provenance="test", review_state="auto",
        created_by="agent",
    )
    before = {mid: memory.db.get(mid)["activation_score"] for mid in ("a", "b", "u")}
    memory.db.apply_heat(["a"], 0.4, "synthetic-event", spread=True, source="test")
    first = {mid: memory.db.get(mid)["activation_score"] for mid in ("a", "b", "u")}
    duplicate = memory.db.apply_heat(["a"], 0.4, "synthetic-event", spread=True, source="test")
    second = {mid: memory.db.get(mid)["activation_score"] for mid in ("a", "b", "u")}
    assert first["a"] > before["a"]
    assert first["b"] > before["b"]
    assert first["u"] == before["u"]
    assert second == first
    assert duplicate["duplicate"] is True


def test_chroma_vector_candidates_are_in_live_recall_path(tmp_path, monkeypatch):
    memory = configured(tmp_path, monkeypatch)
    memory.integrate("Synthetic vector candidate.", "raw", memory_id="vector-a", auto_link=False)

    class FakeCollection:
        def count(self):
            return 1

        def query(self, **kwargs):
            return {
                "ids": [["vector-a"]],
                "documents": [["Synthetic vector candidate."]],
                "metadatas": [[{"memory_id": "vector-a", "level": "raw", "collection": ""}]],
                "distances": [[0.1]],
            }

    memory._collection = FakeCollection()
    rows, status = recall_v2._seed_candidates(
        memory, "vector candidate", memory._encode_query("vector candidate"), 5, None,
    )
    assert rows[0]["memory_id"] == "vector-a"
    assert rows[0]["distance"] == 0.1
    assert status == "unavailable"

def test_release_default_enables_dual_edge(monkeypatch):
    monkeypatch.delenv("ANCHOR_DUAL_EDGE", raising=False)
    assert dual_edge.enabled() is True


def test_active_search_falls_back_to_fts_without_chroma(tmp_path, monkeypatch):
    memory = configured(tmp_path, monkeypatch)
    memory.store(
        "fts-only", "The fictional observatory recalibrated its blue sensor.",
        tier="long",
    )
    assert memory._collection.count() == 0
    assert memory.db.bm25_search("blue sensor calibration", limit=5)
    results = memory.search(
        "blue sensor calibration", n_results=5, hebbian=False,
        associate=False, activate_on_hit=False, cite_on_hit=False,
    )
    assert [row["memory_id"] for row in results] == ["fts-only"]


def test_readme_quick_start_returns_a_result(tmp_path, monkeypatch):
    memory = configured(tmp_path, monkeypatch)
    memory.integrate(
        "The fictional observatory recalibrated its blue sensor.",
        "raw", memory_id="demo-001", auto_link=False,
    )
    result = memory.recall(
        "blue sensor calibration", budget=3,
        policy="search", include_theseus=False,
    )
    assert [row["memory_id"] for row in result["results"]] == ["demo-001"]
