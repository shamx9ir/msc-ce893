"""End-to-end check of harness + evaluator with a scripted stand-in for the LLM.

Needs Cube running on localhost:4000 with the generated model. Run: python -m pytest tests -q
"""
import json
from pathlib import Path

from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_response import LlmResponse
from google.genai import types

from harness.agent import Harness, extract_json
from harness.cube_client import CubeClient
from harness.evaluate import cube_rows, results_match, run_gold

BIRD = Path("data/bird")
DB = "student_club"


def call(name, **args):
    return types.Part(function_call=types.FunctionCall(name=name, args=args))


class ScriptedLLM(BaseLlm):
    """An ADK model that replays fixed replies, like a model that follows the tool protocol."""

    model: str = "scripted"
    replies: list = []
    seen: list = []

    async def generate_content_async(self, llm_request, stream=False):
        self.seen.append(llm_request)
        part = self.replies.pop(0)
        part = types.Part(text=part) if isinstance(part, str) else part
        yield LlmResponse(content=types.Content(role="model", parts=[part]),
                          usage_metadata=types.GenerateContentResponseUsageMetadata(
                              prompt_token_count=100, candidates_token_count=20))


def gold(qid):
    q = next(q for q in json.load(open(BIRD / "dev.json", encoding="utf-8")) if q["question_id"] == qid)
    return run_gold(BIRD / "dev_databases" / DB / f"{DB}.sqlite", q["SQL"])


def test_tool_loop_with_retry_matches_gold():
    bad = {"measures": ["members.count"]}  # typo: harness should reject it with a hint
    good = {"measures": ["member.count"],
            "filters": [{"member": "major.college", "operator": "equals", "values": ["College of Engineering"]}]}
    llm = ScriptedLLM(replies=[
        call("search_values", dimension="major.college", text="engineering"),
        call("submit_answer", query=bad),
        call("submit_answer", query=good),
    ])
    h = Harness(llm, CubeClient())
    res = h.run(DB, "How many students in the Student_Club are from the College of Engineering?")
    assert res.error is None and res.steps == 3 and res.query == good, res
    assert res.prompt_tokens == 300 and res.completion_tokens == 60
    tool_out = [t["content"] for t in res.trace if t["role"] == "tool"]
    assert "College of Engineering" in tool_out[0]
    assert "did you mean member.count" in tool_out[1]
    assert "cube member" in llm.seen[0].config.system_instruction  # schema reached the prompt
    assert results_match(cube_rows(h.cube.load(good)), gold(1313))


def test_column_order_and_set_semantics():
    q = {"dimensions": ["member.last_name", "member.first_name"],
         "filters": [{"member": "major.department", "operator": "equals", "values": ["Art and Design Department"]}]}
    assert results_match(cube_rows(CubeClient().load(q)), gold(1314))


def test_wrong_answer_is_scored_wrong():
    q = {"measures": ["member.count"]}
    assert not results_match(cube_rows(CubeClient().load(q)), gold(1313))


def test_single_mode_parses_fenced_json():
    reply = 'Here you go:\n```json\n{"measures": ["member.count"]}\n```'
    assert extract_json(reply) == {"measures": ["member.count"]}


def test_text_answer_fallback_and_retry():
    """A model that answers in prose with JSON (no tool call) still gets validated and retried."""
    llm = ScriptedLLM(replies=['Answer: {"measures": ["memb.count"]}', '{"measures": ["member.count"]}'])
    res = Harness(llm, CubeClient()).run(DB, "How many members are there?")
    assert res.query == {"measures": ["member.count"]} and res.error is None and res.steps == 2


def test_training_prompt_matches_what_adk_sends():
    """The LoRA training data must use exactly the system prompt ADK builds at run time."""
    from harness.agent import system_prompt
    from harness.prompts import user_prompt
    llm = ScriptedLLM(replies=[call("submit_answer", query={"measures": ["member.count"]})])
    h = Harness(llm, CubeClient())
    q, hint = "How many members live in zip code 28092?", "zip refers to zip"
    h.run(DB, q, hint)
    assert llm.seen[0].config.system_instruction == system_prompt("tools", DB, h.schema(DB))
    sent = llm.seen[0].contents[-1].parts[0].text
    assert sent == user_prompt(q, hint, h.schema(DB)) and "member.zip" in sent


def test_tool_schema_matches_litellm_request():
    """harness/tool_schema.json (used in training files) equals the tools LiteLLM sends."""
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer
    from harness.agent import make_model
    captured = {}

    class Capture(BaseHTTPRequestHandler):
        def do_POST(self):
            captured.update(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            msg = {"role": "assistant", "content": None, "tool_calls": [{"id": "c", "type": "function",
                   "function": {"name": "submit_answer", "arguments": json.dumps({"query": {"measures": ["member.count"]}})}}]}
            out = json.dumps({"id": "x", "object": "chat.completion", "created": 0, "model": "m",
                              "choices": [{"index": 0, "finish_reason": "tool_calls", "message": msg}],
                              "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(out)

        def log_message(self, *a):
            pass

    srv = HTTPServer(("127.0.0.1", 0), Capture)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        h = Harness(make_model("openai/capture", f"http://127.0.0.1:{srv.server_port}/v1", "x"), CubeClient())
        res = h.run(DB, "How many members are there?")
    finally:
        srv.shutdown()
    assert res.query == {"measures": ["member.count"]}
    expected = json.load(open(Path(__file__).parent.parent / "harness" / "tool_schema.json", encoding="utf-8"))
    assert captured["tools"] == expected


def test_sql_to_cube_translation_verifies():
    from harness.sql2cube import Catalog, Translator
    c = CubeClient()
    tr = Translator(Catalog(c.cubes_for(DB)))
    q = next(q for q in json.load(open(BIRD / "dev.json", encoding="utf-8")) if q["question_id"] == 1313)
    cq = tr.translate(q["SQL"])
    assert cq == {"measures": ["member.count"],
                  "filters": [{"member": "major.college", "operator": "equals", "values": ["College of Engineering"]}]}
    assert results_match(cube_rows(c.load(cq)), gold(1313))


def test_dimension_used_as_measure_gets_a_hint():
    h = Harness(ScriptedLLM(replies=[]), CubeClient())
    err = h.validate(DB, {"measures": ["event.event_name"]})
    assert "is a dimension" in err and '"dimensions"' in err
    assert "Each filter must look like" in h.validate(
        DB, {"measures": ["member.count"], "filters": [{"dimension": "major.college", "value": "x"}]})


def test_text_tool_calls_are_unwrapped_and_followed():
    """A model that writes run_query as text gets the result back and then submits."""
    q = {"measures": ["member.count"]}
    llm = ScriptedLLM(replies=[
        json.dumps({"name": "run_query", "arguments": {"query": q}}),
        call("submit_answer", query=q),
    ])
    res = Harness(llm, CubeClient()).run(DB, "How many members are there?")
    assert res.query == q and res.error is None and res.steps == 2
    assert any(t.get("name") == "run_query(text)" for t in res.trace)
    assert extract_json('{"name": "submit_answer", "arguments": {"query": {"measures": ["a.b"]}}}') == {"measures": ["a.b"]}


def test_column_check_warns_once_then_accepts():
    q = {"dimensions": ["event.event_name"], "measures": ["attendance.count"], "order": {"attendance.count": "desc"}, "limit": 1}
    good = {"dimensions": ["event.event_name"], "order": {"attendance.count": "desc"}, "limit": 1}
    llm = ScriptedLLM(replies=[call("submit_answer", query=q), call("submit_answer", query=good)])
    res = Harness(llm, CubeClient()).run(DB, "Which event had the highest attendance?")
    assert res.query == good and res.error is None and res.steps == 2
    assert "does not ask for a number" in [t for t in res.trace if t["role"] == "tool"][0]["content"]
    # a model that resubmits the same query gets it trimmed by the harness
    llm = ScriptedLLM(replies=[call("submit_answer", query=q), call("submit_answer", query=q)])
    res = Harness(llm, CubeClient()).run(DB, "Which event had the highest attendance?")
    assert res.query == good and res.error is None


def test_repeated_calls_and_numeric_search():
    llm = ScriptedLLM(replies=[call("search_values", dimension="member.zip", text="null"),
                               call("search_values", dimension="member.zip", text="null"),
                               call("submit_answer", query={"measures": ["member.count"]})])
    res = Harness(llm, CubeClient()).run(DB, "How many members are there?")
    outs = [t["content"] for t in res.trace if t["role"] == "tool"]
    assert "number dimension" in outs[0] and "already made this exact call" in outs[1]


def test_filter_type_check_and_linking():
    from harness.linking import linked_fields
    h = Harness(ScriptedLLM(replies=[]), CubeClient())
    err = h.validate(DB, {"measures": ["member.count"],
                          "filters": [{"member": "member.zip", "operator": "equals", "values": ["abc"]}]})
    assert err and "is a number" in err
    assert h.validate(DB, {"measures": ["member.count"],
                           "filters": [{"member": "member.zip", "operator": "equals", "values": ["28092"]}]}) is None
    fields = linked_fields(h.schema(DB), "List the first names of members", "first name refers to first_name")
    assert fields[0] == "member.first_name"


def test_text_answer_gets_column_check():
    import json as _json
    q = {"dimensions": ["member.first_name", "member.position"],
         "filters": [{"member": "member.position", "operator": "equals", "values": ["President"]}]}
    llm = ScriptedLLM(replies=[_json.dumps(q), _json.dumps(q)])
    res = Harness(llm, CubeClient()).run(DB, "What is the first name of the president?")
    assert res.error is None and len(llm.seen) == 2
    assert res.query["dimensions"] == ["member.first_name"]  # unchanged resubmit gets trimmed


def test_time_dimension_must_be_time():
    h = Harness(ScriptedLLM(replies=[]), CubeClient())
    err = h.validate(DB, {"measures": ["member.count"],
                          "timeDimensions": [{"dimension": "member.first_name", "dateRange": "2021"}]})
    assert err and "not a time dimension" in err
