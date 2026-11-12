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
    q = next(q for q in json.load(open(BIRD / "dev.json")) if q["question_id"] == qid)
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
