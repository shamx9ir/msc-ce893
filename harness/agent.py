"""The harness: an ADK agent (prompt + Cube tools) with a validate/retry loop.

The model is any LiteLLM-supported endpoint, e.g. a local Ollama or vLLM server, so the same
code serves the base model and the LoRA model. The loop itself is ADK's: the model calls tools
until it submits an accepted query or hits the LLM-call limit.
"""
import asyncio
import difflib
import functools
import json
import re
import time
import uuid
from dataclasses import dataclass, field

from google.adk.agents import LlmAgent
from google.adk.agents.run_config import RunConfig
from google.adk.models.base_llm import BaseLlm
from google.adk.models.lite_llm import LiteLlm
from google.adk.runners import InMemoryRunner
from google.adk.tools import ToolContext
from google.genai import types

from .cube_client import CubeClient, CubeError, dumps
from .prompts import QUERY_GUIDE, SYSTEM_SINGLE, SYSTEM_TOOLS, user_prompt

QUERY_KEYS = {"measures", "dimensions", "filters", "timeDimensions", "order", "limit",
              "segments", "ungrouped", "offset", "total"}


@dataclass
class AgentResult:
    query: dict | None
    steps: int = 0                 # LLM calls
    prompt_tokens: int = 0
    completion_tokens: int = 0
    llm_latency_s: float = 0.0     # wall time minus time spent inside tools
    wall_s: float = 0.0
    error: str | None = None
    trace: list = field(default_factory=list)


def make_model(model: str, base_url: str | None = None, api_key: str | None = None,
               temperature: float = 0.0) -> LiteLlm:
    """`model` is a LiteLLM id, e.g. "ollama_chat/qwen2.5-coder:7b" or "openai/my-lora" (vLLM).
    A bare name like "qwen2.5-coder:7b" is treated as an Ollama model."""
    if "/" not in model:
        model = f"ollama_chat/{model}"
    kwargs = {"temperature": temperature}
    if base_url:
        kwargs["api_base"] = base_url
    if api_key:
        kwargs["api_key"] = api_key
    return LiteLlm(model=model, **kwargs)


def extract_json(text: str) -> dict | None:
    """Pull the first JSON object out of free text (models often wrap it in prose or ``` fences)."""
    if not text:
        return None
    text = re.sub(r"```(?:json)?", "", text)
    start = text.find("{")
    while start != -1:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(text[start:i + 1])
                        if isinstance(obj, dict):
                            return obj["query"] if isinstance(obj.get("query"), dict) else obj
                    except json.JSONDecodeError:
                        pass
                    break
        start = text.find("{", start + 1)
    return None


def _members_in(query: dict) -> list[str]:
    names = list(query.get("measures") or []) + list(query.get("dimensions") or [])
    names += [t.get("dimension") for t in query.get("timeDimensions") or [] if isinstance(t, dict)]
    order = query.get("order") or {}
    names += list(order.keys()) if isinstance(order, dict) else [o[0] for o in order if o]

    def walk(fs):
        for f in fs or []:
            if not isinstance(f, dict):
                continue
            if "member" in f:
                names.append(f["member"])
            walk(f.get("or"))
            walk(f.get("and"))
    walk(query.get("filters"))
    return [n for n in names if isinstance(n, str)]


class Harness:
    def __init__(self, model: BaseLlm, cube: CubeClient, mode: str = "tools",
                 max_steps: int = 8, preview_rows: int = 10):
        assert mode in ("tools", "single")
        self.model, self.cube, self.mode = model, cube, mode
        self.max_steps, self.preview_rows = max_steps, preview_rows
        self._schema_cache: dict[str, str] = {}

    # ---- prompts -------------------------------------------------------
    def schema(self, db: str) -> str:
        if db not in self._schema_cache:
            self._schema_cache[db] = self.cube.describe(db)
        return self._schema_cache[db]

    def instruction(self, db: str) -> str:
        tmpl = SYSTEM_TOOLS if self.mode == "tools" else SYSTEM_SINGLE
        return tmpl.format(db=db, guide=QUERY_GUIDE, schema=self.schema(db))

    # ---- validation ----------------------------------------------------
    def validate(self, db: str, query) -> str | None:
        """Return an error message a model can act on, or None if the query looks well formed."""
        if not isinstance(query, dict):
            return "The query must be a JSON object."
        unknown_keys = set(query) - QUERY_KEYS
        if unknown_keys:
            return f"Unknown query keys {sorted(unknown_keys)}. Allowed: {sorted(QUERY_KEYS)}."
        known = self.cube.member_names(db)
        bad = [m for m in _members_in(query) if m not in known]
        if bad:
            hints = {m: difflib.get_close_matches(m, known, n=3) for m in bad}
            return "Unknown members: " + "; ".join(
                f"{m} (did you mean {', '.join(h) or 'nothing close'}?)" for m, h in hints.items())
        if not query.get("measures") and not query.get("dimensions") and not query.get("timeDimensions"):
            return "The query needs at least one measure or dimension."
        return None

    def try_run(self, db: str, query) -> tuple[list | None, str | None]:
        err = self.validate(db, query)
        if err:
            return None, err
        try:
            return self.cube.load(query), None
        except CubeError as e:
            return None, f"Cube error: {e}"
        except Exception as e:  # network etc.
            return None, f"Error: {e}"

    # ---- tools ---------------------------------------------------------
    def _tools(self, db: str, res: AgentResult, tool_time: list):
        def timed(fn):
            @functools.wraps(fn)  # keep the signature: ADK builds the tool schema from it
            def wrapper(*a, **k):
                t = time.time()
                try:
                    return fn(*a, **k)
                finally:
                    tool_time[0] += time.time() - t
            return wrapper

        @timed
        def search_values(dimension: str, text: str) -> str:
            """List stored values of a dimension that contain the text (case-insensitive).

            Args:
                dimension: a dimension name such as drivers.surname
                text: part of the value to look for
            """
            if dimension not in self.cube.member_names(db):
                return f"Unknown dimension {dimension}."
            try:
                return dumps(self.cube.dimension_values(dimension, text or None))
            except Exception as e:
                return f"Error: {e}"

        @timed
        def run_query(query: dict) -> str:
            """Run a Cube query and return up to 10 rows and the row count, or the error.

            Args:
                query: the Cube query JSON object
            """
            query = extract_json(query) if isinstance(query, str) else query
            rows, err = self.try_run(db, query)
            return err or dumps({"row_count": len(rows), "rows": rows[: self.preview_rows]})

        @timed
        def submit_answer(query: dict, tool_context: ToolContext) -> str:
            """Submit the final Cube query. Ends the task if the query runs.

            Args:
                query: the Cube query JSON object
            """
            query = extract_json(query) if isinstance(query, str) else query
            res.query = query
            _, err = self.try_run(db, query)
            if err:
                res.error = err
                return f"Not accepted: {err}\nFix the query and call submit_answer again."
            res.error = None
            tool_context.actions.skip_summarization = True  # stop here, no extra LLM call
            return "Accepted."

        return [search_values, run_query, submit_answer]

    # ---- main loop -----------------------------------------------------
    def run(self, db: str, question: str, evidence: str | None = None) -> AgentResult:
        return asyncio.run(self.run_async(db, question, evidence))

    async def run_async(self, db: str, question: str, evidence: str | None = None) -> AgentResult:
        t0 = time.time()
        res, tool_time = AgentResult(query=None), [0.0]
        agent = LlmAgent(
            name="cube_analyst", model=self.model, instruction=self.instruction(db),
            tools=self._tools(db, res, tool_time) if self.mode == "tools" else [],
            generate_content_config=types.GenerateContentConfig(temperature=0.0))
        runner = InMemoryRunner(agent=agent, app_name="bird_cube")
        session = await runner.session_service.create_session(app_name="bird_cube", user_id="eval")
        last_text = ""
        submitted = False
        prompt = user_prompt(question, evidence)
        for attempt in range(2):  # second pass only if the model never submitted anything usable
            try:
                async for ev in runner.run_async(
                        user_id="eval", session_id=session.id,
                        new_message=types.Content(role="user", parts=[types.Part(text=prompt)]),
                        run_config=RunConfig(max_llm_calls=self.max_steps)):
                    self._record(ev, res)
                    for p in (ev.content.parts if ev.content else []) or []:
                        if p.text and ev.author == agent.name:
                            last_text = p.text
                        if p.function_response and p.function_response.name == "submit_answer":
                            submitted = True
            except Exception as e:  # LLM-call limit, endpoint errors, malformed tool calls
                res.error = f"{type(e).__name__}: {e}"[:500]
                break
            if res.query is not None and res.error is None:
                break
            # Fallback: a JSON object in the final text counts as the answer (small models often do this).
            query = extract_json(last_text)
            if query is not None:
                _, err = self.try_run(db, query)
                res.query, res.error = query, err
                if err is None or self.mode == "single" or res.steps >= self.max_steps:
                    break
                prompt = f"That query failed: {err}\nFix it and call submit_answer."
            elif self.mode == "single" or submitted or res.steps >= self.max_steps:
                res.error = res.error or "No query submitted"
                break
            else:
                prompt = "Call submit_answer with your final Cube query."
        res.wall_s = time.time() - t0
        res.llm_latency_s = max(0.0, res.wall_s - tool_time[0])
        return res

    @staticmethod
    def _record(ev, res: AgentResult) -> None:
        um = getattr(ev, "usage_metadata", None)
        if um:
            res.prompt_tokens += um.prompt_token_count or 0
            res.completion_tokens += um.candidates_token_count or 0
        if not ev.content or not ev.content.parts:
            return
        for p in ev.content.parts:
            if p.function_call:
                res.trace.append({"role": "assistant", "tool": p.function_call.name,
                                  "args": dict(p.function_call.args or {})})
            elif p.function_response:
                out = p.function_response.response or {}
                res.trace.append({"role": "tool", "name": p.function_response.name,
                                  "content": str(out.get("result", out))[:2000]})
            elif p.text:
                res.trace.append({"role": ev.author, "content": p.text[:2000]})
        if ev.author != "user" and any(p.function_call or p.text for p in ev.content.parts) \
                and ev.content.role == "model":
            res.steps += 1
