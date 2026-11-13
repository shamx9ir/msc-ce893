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
               temperature: float = 0.0, num_ctx: int | None = 16384,
               max_tokens: int | None = 1024) -> LiteLlm:
    """`model` is a LiteLLM id, e.g. "ollama_chat/qwen2.5-coder:7b" or "openai/my-lora" (vLLM).
    A bare name like "qwen2.5-coder:7b" is treated as an Ollama model.

    num_ctx: Ollama's context window. Its default (4,096) is shorter than the formula_1 prompt,
    which silently truncates the conversation, so both arms run with the same larger window.
    max_tokens caps one reply, so a runaway generation cannot stall a question."""
    if "/" not in model:
        model = f"ollama_chat/{model}"
    kwargs = {"temperature": temperature}
    if max_tokens:
        kwargs["max_tokens"] = max_tokens
    if num_ctx and model.startswith("ollama"):
        kwargs["num_ctx"] = num_ctx
    if base_url:
        kwargs["api_base"] = base_url
    if api_key:
        kwargs["api_key"] = api_key
    return LiteLlm(model=model, **kwargs)


def extract_json(text: str, unwrap: bool = True) -> dict | None:
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
                            return unwrap_query(obj) if unwrap else obj
                    except json.JSONDecodeError:
                        pass
                    break
        start = text.find("{", start + 1)
    return None


def unwrap_query(obj: dict) -> dict:
    """Small models often write a tool call as text: {"name": "run_query", "arguments": {"query": {...}}}
    or {"query": {...}}. Return the Cube query inside."""
    args = obj.get("arguments") or obj.get("parameters")
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            args = None
    if isinstance(args, dict):
        obj = args
    if isinstance(obj.get("query"), dict):
        obj = obj["query"]
    elif isinstance(obj.get("query"), str):
        try:
            inner = json.loads(obj["query"])
            obj = inner if isinstance(inner, dict) else obj
        except json.JSONDecodeError:
            pass
    return obj


QUANTITY_WORDS = re.compile(
    r"\b(how many|how much|number of|count|total|sum|average|avg|mean|amount|percentage|percent|ratio|"
    r"rate|difference|max(imum)?|min(imum)?|score[sd]?|points)\b", re.I)


def column_warning(query: dict, question: str) -> str | None:
    """Heuristics for the most common wrong answer: returning columns the question did not ask for."""
    if not question:
        return None
    single_value_filters = set()
    for f in query.get("filters") or []:
        if isinstance(f, dict) and f.get("operator") == "equals" and len(f.get("values") or []) == 1:
            single_value_filters.add(f.get("member"))
    redundant = [d for d in query.get("dimensions") or [] if d in single_value_filters]
    if redundant:
        return (f"{', '.join(redundant)} is filtered to one value and also returned as a column, "
                "so it adds nothing to the answer.")
    measures = query.get("measures") or []
    if measures and query.get("dimensions") and not QUANTITY_WORDS.search(question):
        return (f"the question does not ask for a number, but the query returns {', '.join(measures)}. "
                "To rank by a measure, use it in \"order\" without listing it in \"measures\".")
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


AGENT_NAME = "cube_analyst"
# ADK appends this to the instruction; training prompts must carry it too (checked in tests).
ADK_SUFFIX = f'\n\nYou are an agent. Your internal name is "{AGENT_NAME}".'


def render_instruction(mode: str, db: str, schema: str) -> str:
    tmpl = SYSTEM_TOOLS if mode == "tools" else SYSTEM_SINGLE
    return tmpl.format(db=db, guide=QUERY_GUIDE, schema=schema)


def system_prompt(mode: str, db: str, schema: str) -> str:
    """The exact system message the model receives through ADK."""
    return render_instruction(mode, db, schema) + ADK_SUFFIX


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
        return render_instruction(self.mode, db, self.schema(db))

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
        kinds = self.cube.member_kinds(db)
        wrong = [m for m in query.get("measures") or [] if kinds.get(m) == "dimension"]
        if wrong:
            return (f"{', '.join(wrong)} {'is a dimension' if len(wrong) == 1 else 'are dimensions'}, not "
                    f"{'a measure' if len(wrong) == 1 else 'measures'}: put it under \"dimensions\", or use a "
                    f"measure such as count, sum_x or avg_x.")
        wrong = [m for m in query.get("dimensions") or [] if kinds.get(m) == "measure"]
        if wrong:
            return f"{', '.join(wrong)} is a measure, not a dimension: put it under \"measures\"."
        bad_filter = self._check_filters(query.get("filters"))
        if bad_filter:
            return bad_filter
        if not query.get("measures") and not query.get("dimensions") and not query.get("timeDimensions"):
            return "The query needs at least one measure or dimension."
        return None

    @staticmethod
    def _check_filters(filters) -> str | None:
        shape = 'Each filter must look like {"member": "cube.dim", "operator": "equals", "values": ["x"]}.'
        if filters is None:
            return None
        if not isinstance(filters, list):
            return "\"filters\" must be a list. " + shape
        for f in filters:
            if not isinstance(f, dict):
                return shape
            if "or" in f or "and" in f:
                err = Harness._check_filters(f.get("or") or f.get("and"))
                if err:
                    return err
                continue
            if "member" not in f or "operator" not in f:
                return f"Bad filter {json.dumps(f)}. " + shape
            if f["operator"] not in ("set", "notSet") and not isinstance(f.get("values"), list):
                return f"Filter on {f['member']} needs \"values\" as a list of strings. " + shape
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
    def _tools(self, db: str, res: AgentResult, tool_time: list, question: str = ""):
        warned: set[str] = set()
        seen: dict[str, str] = {}

        def timed(fn):
            @functools.wraps(fn)  # keep the signature: ADK builds the tool schema from it
            def wrapper(*a, **k):
                t = time.time()
                key = fn.__name__ + dumps({n: v for n, v in k.items() if n != "tool_context"})
                try:
                    if key in seen and fn.__name__ != "submit_answer":
                        return ("You already made this exact call. Its result was: " + seen[key][:1500] +
                                "\nDo not repeat it; change the call or submit your answer.")
                    out = fn(*a, **k)
                    seen[key] = out
                    return out
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
            kind = self.cube.member_types().get(dimension, "string")
            try:
                values = self.cube.dimension_values(dimension, text or None)
            except Exception as e:
                return f"Error: {e}"
            if kind != "string":
                return (f"{dimension} is a {kind} dimension, so it cannot be searched as text. Sample values: "
                        f"{dumps(values)}. Filter on it directly (equals, gt, lt, inDateRange).")
            if not values:
                return f"No value of {dimension} contains {text!r}. Try a shorter or different spelling."
            return dumps(values)

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
            warning = column_warning(query, question)
            if warning and dumps(query) not in warned:
                warned.add(dumps(query))
                res.error = "column check"
                return (f"Check before submitting: {warning} Return only the columns the question asks for. "
                        "If the query is already right, call submit_answer again with the same query.")
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
            name=AGENT_NAME, model=self.model, instruction=self.instruction(db),
            tools=self._tools(db, res, tool_time, question) if self.mode == "tools" else [],
            generate_content_config=types.GenerateContentConfig(temperature=0.0))
        runner = InMemoryRunner(agent=agent, app_name="bird_cube")
        session = await runner.session_service.create_session(app_name="bird_cube", user_id="eval")
        last_text = ""
        submitted = False
        prompt = user_prompt(question, evidence)
        for attempt in range(4):  # extra passes only when the model answered in text instead of tools
            try:
                async for ev in runner.run_async(
                        user_id="eval", session_id=session.id,
                        new_message=types.Content(role="user", parts=[types.Part(text=prompt)]),
                        run_config=RunConfig(max_llm_calls=max(1, self.max_steps - res.steps))):
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
            if self.mode == "single" or res.steps >= self.max_steps:
                if res.query is None:
                    raw = extract_json(last_text, unwrap=False)
                    query = unwrap_query(raw) if raw else None
                    if query is not None:
                        _, err = self.try_run(db, query)
                        res.query, res.error = query, err
                res.error = res.error if res.query is not None else (res.error or "No query submitted")
                break
            # Fallback for tool calls written as text (common with small models).
            raw = extract_json(last_text, unwrap=False)
            if raw is None:
                if submitted:
                    break
                prompt = "Call submit_answer with your final Cube query."
                continue
            if raw.get("name") == "search_values":
                args = unwrap_query(raw)
                dim = args.get("dimension", "")
                try:
                    found = (dumps(self.cube.dimension_values(dim, args.get("text") or None))
                             if dim in self.cube.member_names(db) else f"Unknown dimension {dim}.")
                except Exception as e:
                    found = f"Error: {e}"
                res.trace.append({"role": "tool", "name": "search_values(text)", "content": found[:2000]})
                prompt = f"search_values result: {found}\nContinue, and finish by calling submit_answer."
                continue
            query = unwrap_query(raw)
            rows, err = self.try_run(db, query)
            res.query, res.error = query, err
            if err is None and raw.get("name") != "run_query":
                break  # a final answer given as text
            if err is None:
                preview = dumps({"row_count": len(rows), "rows": rows[: self.preview_rows]})
                res.trace.append({"role": "tool", "name": "run_query(text)", "content": preview[:2000]})
                prompt = f"run_query result: {preview}\nIf this answers the question, call submit_answer with the query."
            else:
                prompt = f"That query failed: {err}\nFix it and call submit_answer."
        if res.error == "column check":
            res.error = None  # only a warning: the last submitted query ran fine
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
