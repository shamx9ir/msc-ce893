"""The harness: prompt + Cube tools + validate/retry loop around any chat model."""
import difflib
import json
import re
import time
from dataclasses import dataclass, field

from .cube_client import CubeClient, CubeError, dumps
from .llm import LLMClient
from .prompts import QUERY_GUIDE, SYSTEM_SINGLE, SYSTEM_TOOLS, TOOLS, user_prompt

QUERY_KEYS = {"measures", "dimensions", "filters", "timeDimensions", "order", "limit",
              "segments", "ungrouped", "offset", "total"}


@dataclass
class AgentResult:
    query: dict | None
    steps: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    llm_latency_s: float = 0.0
    wall_s: float = 0.0
    error: str | None = None
    trace: list = field(default_factory=list)


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
                            return obj.get("query", obj) if "query" in obj else obj
                    except json.JSONDecodeError:
                        break
        start = text.find("{", start + 1)
    return None


def _members_in(query: dict) -> list[str]:
    names = list(query.get("measures", [])) + list(query.get("dimensions", []))
    names += [t.get("dimension") for t in query.get("timeDimensions", []) if isinstance(t, dict)]
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
    return [n for n in names if n]


class Harness:
    def __init__(self, llm: LLMClient, cube: CubeClient, mode: str = "tools",
                 max_steps: int = 8, preview_rows: int = 10):
        assert mode in ("tools", "single")
        self.llm, self.cube, self.mode = llm, cube, mode
        self.max_steps, self.preview_rows = max_steps, preview_rows
        self._schema_cache: dict[str, str] = {}

    # ---- prompts -------------------------------------------------------
    def schema(self, db: str) -> str:
        if db not in self._schema_cache:
            self._schema_cache[db] = self.cube.describe(db)
        return self._schema_cache[db]

    def messages(self, db: str, question: str, evidence: str | None) -> list[dict]:
        tmpl = SYSTEM_TOOLS if self.mode == "tools" else SYSTEM_SINGLE
        return [{"role": "system", "content": tmpl.format(db=db, guide=QUERY_GUIDE, schema=self.schema(db))},
                {"role": "user", "content": user_prompt(question, evidence)}]

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

    # ---- main loop -----------------------------------------------------
    def run(self, db: str, question: str, evidence: str | None = None) -> AgentResult:
        t0 = time.time()
        res = AgentResult(query=None)
        msgs = self.messages(db, question, evidence)
        tools = TOOLS if self.mode == "tools" else None
        for step in range(1, self.max_steps + 1):
            res.steps = step
            try:
                out = self.llm.chat(msgs, tools=tools)
            except Exception as e:
                res.error = f"LLM error: {e}"
                break
            res.prompt_tokens += out.prompt_tokens
            res.completion_tokens += out.completion_tokens
            res.llm_latency_s += out.latency_s
            msg = out.message
            res.trace.append({"role": "assistant", "content": msg.get("content"),
                              "tool_calls": msg.get("tool_calls")})
            calls = msg.get("tool_calls") or []

            if not calls:
                # No tool call: treat a JSON object in the text as the answer.
                query = extract_json(msg.get("content") or "")
                if query is None:
                    if self.mode == "single":
                        res.error = "No JSON query in reply"
                        break
                    msgs += [{"role": "assistant", "content": msg.get("content") or ""},
                             {"role": "user", "content": "Call submit_answer with your final Cube query."}]
                    continue
                _, err = self.try_run(db, query)
                res.query = query
                if err is None or self.mode == "single":
                    res.error = err
                    break
                msgs += [{"role": "assistant", "content": msg.get("content") or ""},
                         {"role": "user", "content": f"That query failed: {err}\nFix it and answer again."}]
                continue

            msgs.append({"role": "assistant", "content": msg.get("content") or "", "tool_calls": calls})
            done = False
            for call in calls:
                name = call["function"]["name"]
                try:
                    args = call["function"].get("arguments") or "{}"
                    args = json.loads(args) if isinstance(args, str) else args
                except json.JSONDecodeError:
                    args = {}
                query = args.get("query")
                if isinstance(query, str):
                    query = extract_json(query)
                if name == "search_values":
                    dim = args.get("dimension", "")
                    if dim not in self.cube.member_names(db):
                        result = f"Unknown dimension {dim}."
                    else:
                        try:
                            result = dumps(self.cube.dimension_values(dim, args.get("text") or None))
                        except Exception as e:
                            result = f"Error: {e}"
                elif name == "run_query":
                    rows, err = self.try_run(db, query)
                    result = err or dumps({"row_count": len(rows), "rows": rows[: self.preview_rows]})
                elif name == "submit_answer":
                    rows, err = self.try_run(db, query)
                    res.query = query
                    if err:
                        result = f"Not accepted: {err}\nFix the query and call submit_answer again."
                    else:
                        result, done = "Accepted.", True
                else:
                    result = f"Unknown tool {name}."
                res.trace.append({"role": "tool", "name": name, "content": result[:2000]})
                msgs.append({"role": "tool", "tool_call_id": call.get("id", name), "content": result[:4000]})
            if done:
                res.error = None
                break
        else:
            res.error = res.error or "Step limit reached"
        res.wall_s = time.time() - t0
        return res
