"""The demo app builds each model's harness with the prompt profile it was trained on."""
from fastapi.testclient import TestClient

import app.server as srv


def test_models_have_profiles_and_unique_ids():
    ids = [m["id"] for m in srv.MODELS]
    assert len(ids) == len(set(ids))
    assert {m["profile"] for m in srv.MODELS} <= {"full", "lean", "noschema"}
    assert sum(m["core"] for m in srv.MODELS) == 4


def test_harness_gets_profile_and_tool_schemas(monkeypatch, tmp_path):
    built = {}

    def fake_make_model(model_id, base_url, num_ctx=None, tool_schemas=True):
        return {"id": model_id, "tool_schemas": tool_schemas}

    class FakeHarness:
        def __init__(self, llm, cube, max_steps=8, profile="full"):
            built[llm["id"]] = (llm["tool_schemas"], profile)

        def run(self, db, question, hint):
            raise RuntimeError("no model in tests")

    monkeypatch.setattr(srv, "make_model", fake_make_model)
    monkeypatch.setattr(srv, "Harness", FakeHarness)
    monkeypatch.setattr(srv, "build_surface", lambda *a, **k: [])
    monkeypatch.setattr(srv.CubeClient, "meta", lambda self: {"cubes": []})
    client = TestClient(srv.create_app("http://cube.invalid", "http://ollama.invalid", 16384, 8, tmp_path))
    for m in srv.MODELS:
        r = client.post("/api/ask", json={"question": "q", "db": "formula_1", "model": m["id"]})
        assert r.status_code == 200
    assert built["ollama_chat/qwen14b-cube-lora"] == (True, "full")
    assert built["ollama_chat/qwen7b-cube-lean"] == (False, "lean")
    assert built["ollama_chat/qwen14b-cube-noschema"] == (False, "lean")
    assert built["ollama_chat/qwen7b-cube-sv3-nt"] == (True, "full")
    assert built["ollama_chat/qwen14b-cube-lean-sv3"] == (False, "lean")
