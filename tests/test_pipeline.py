"""End-to-end check of harness + evaluator with a scripted stand-in for the LLM.

Needs Cube running on localhost:4000 with the generated model. Run: python -m pytest tests -q
"""
import json
from pathlib import Path

from harness.agent import Harness, extract_json
from harness.cube_client import CubeClient
from harness.evaluate import cube_rows, results_match, run_gold
from harness.llm import LLMResponse

BIRD = Path("data/bird")
DB = "student_club"


def call(name, **args):
    return {"id": name, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


class ScriptedLLM:
    """Replays a fixed list of assistant messages, like a model that follows the tool protocol."""

    def __init__(self, replies):
        self.replies = list(replies)

    def chat(self, messages, tools=None):
        return LLMResponse(message=self.replies.pop(0), prompt_tokens=100, completion_tokens=20)


def gold(qid):
    q = next(q for q in json.load(open(BIRD / "dev.json")) if q["question_id"] == qid)
    return run_gold(BIRD / "dev_databases" / DB / f"{DB}.sqlite", q["SQL"])


def test_tool_loop_with_retry_matches_gold():
    bad = {"measures": ["members.count"]}  # typo: harness should reject it with a hint
    good = {"measures": ["member.count"],
            "filters": [{"member": "major.college", "operator": "equals", "values": ["College of Engineering"]}]}
    llm = ScriptedLLM([
        {"content": "", "tool_calls": [call("search_values", dimension="major.college", text="engineering")]},
        {"content": "", "tool_calls": [call("submit_answer", query=bad)]},
        {"content": "", "tool_calls": [call("submit_answer", query=good)]},
    ])
    h = Harness(llm, CubeClient())
    res = h.run(DB, "How many students in the Student_Club are from the College of Engineering?")
    assert res.error is None and res.steps == 3 and res.query == good
    assert "College of Engineering" in res.trace[1]["content"]
    assert "did you mean member.count" in res.trace[3]["content"]
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
