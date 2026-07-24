from __future__ import annotations

import importlib

import taxonomy_tagger


def test_runtime_registers_full_tool_surface_without_starting_servers(tmp_path, monkeypatch):
    data = tmp_path / "data"
    monkeypatch.setenv("ANCHOR_DATA_DIR", str(data))
    monkeypatch.setenv("ANCHOR_DB_PATH", str(data / "memories.db"))
    monkeypatch.setenv("ANCHOR_DISABLE_CHROMA", "1")
    monkeypatch.setenv("ANCHOR_KUZU_PATH", str(tmp_path / "kuzu"))
    monkeypatch.delenv("ANCHOR_GATEWAY_API_KEY", raising=False)
    monkeypatch.delenv("ANCHOR_HOOK_API_KEY", raising=False)
    module = importlib.import_module("anchor_sse")
    names = sorted(module.mcp._tool_manager._tools)
    assert names == sorted(
        [
            "store_memory",
            "search_memory",
            "chat_history",
            "thread",
            "wenku_read",
            "briefing",
            "dream_pass",
            "memory_edit",
            "graph_review",
            "belief",
        ]
    )
    assert "/api/graph/health" in {route.path for route in module._rest.routes}

    tagged = {}
    monkeypatch.setattr(
        taxonomy_tagger,
        "tag_async",
        lambda memory_id, text, timestamp="": tagged.update(
            {"memory_id": memory_id, "text": text}
        ),
    )
    stored = module.store_memory(
        "Synthetic runtime memory for full-chain verification.",
        level="raw",
    )
    assert stored.startswith("已存储:")
    assert tagged["text"] == "Synthetic runtime memory for full-chain verification."
    assert module.mem.db.get(tagged["memory_id"])

    brief = module.briefing("contract test")
    assert "Synthetic runtime memory for full-chain verification." in brief
    dream = module.dream_pass("contract test")
    assert "提审模式" in dream
