"""Loads the stored evaluation runs and training logs and computes every number in the report.

Used by report/make_figures.py and notebook/CE893_evaluation.ipynb. Needs pandas, sqlglot, scipy.
All inputs are in results/ (committed), so nothing here needs Cube, Ollama or a GPU.
"""
import gzip
import json
import re
from functools import lru_cache
from pathlib import Path

import pandas as pd
import sqlglot
from sqlglot import exp
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "results" / "runs"
TRAINING = ROOT / "results" / "training"

# name -> (file stem, label used in tables and figures)
# All reported test runs hide BIRD's evidence hint from the model (evaluate.py --no-evidence).
TEST_RUNS = {
    "harness_7b": ("test_nohint_harness_7b", "7B harness only"),
    "harness_14b": ("test_nohint_harness_14b", "14B harness only"),
    "lora_7b": ("test_nohint_lora_7b", "7B LoRA + harness"),
    "lora_14b": ("test_nohint_lora_14b", "14B LoRA + harness"),
    "lora_on_lean_7b": ("test_nohint_lora_on_lean_prompt_7b", "7B full LoRA, lean prompt"),
    "lean_7b": ("test_nohint_lean_7b", "7B lean LoRA"),
    "lean_14b": ("test_nohint_lean_14b", "14B lean LoRA"),
    "noschema_7b": ("test_nohint_noschema_7b", "7B no-schema LoRA, no schema"),
    "noschema_14b": ("test_nohint_noschema_14b", "14B no-schema LoRA, no schema"),
    "noschema_lean_7b": ("test_nohint_noschema_lean_prompt_7b", "7B no-schema LoRA, lean prompt"),
    "noschema_lean_14b": ("test_nohint_noschema_lean_prompt_14b", "14B no-schema LoRA, lean prompt"),
    # earlier runs WITH the hint (BIRD's standard setting); kept for reference and for harness development
    "hint_harness_v1_7b": ("test_harness_v1_7b", "7B harness v1 (hint)"),
    "hint_harness_v2_7b": ("test_harness_v2_7b", "7B harness v2 (hint)"),
    "hint_harness_7b": ("test_harness_7b", "7B harness only (hint)"),
    "hint_harness_14b": ("test_harness_14b", "14B harness only (hint)"),
    "hint_lora_7b": ("test_lora_7b", "7B LoRA + harness (hint)"),
    "hint_lora_14b": ("test_lora_14b", "14B LoRA + harness (hint)"),
    "hint_lora_on_lean_7b": ("test_lora_on_lean_prompt_7b", "7B full LoRA, lean prompt (hint)"),
    "hint_lora_on_lean_14b": ("test_lora_on_lean_prompt_14b_partial", "14B full LoRA, lean prompt (hint, 127 q)"),
    "hint_lean_7b": ("test_lean_7b", "7B lean LoRA (hint)"),
    "hint_lean_14b": ("test_lean_14b", "14B lean LoRA (hint)"),
    "hint_noschema_7b": ("test_noschema_7b", "7B no-schema LoRA, no schema (hint)"),
    "hint_noschema_14b": ("test_noschema_14b", "14B no-schema LoRA, no schema (hint)"),
    "hint_noschema_lean_7b": ("test_noschema_lean_prompt_7b", "7B no-schema LoRA, lean prompt (hint)"),
    "hint_noschema_lean_14b": ("test_noschema_lean_prompt_14b", "14B no-schema LoRA, lean prompt (hint)"),
}
MAIN_ARMS = ["harness_7b", "lora_7b", "harness_14b", "lora_14b"]
TUNE_RUNS = {
    "v2": "tune120_harness_v2_7b", "v3": "tune120_harness_v3_7b", "v4": "tune120_harness_v4_7b",
    "v4.1": "tune120_harness_v4_1_7b",
}
TUNE363_RUNS = {"v3 7B": "tune363_harness_v3_7b", "v4.2 7B": "tune363_harness_v4_2_7b",
                "v4.2 14B": "tune363_harness_v4_2_14b"}
TRAIN_LOGS = ["lora_full_7b", "lora_full_14b", "lora_lean_7b", "lora_lean_14b",
              "lora_noschema_7b", "lora_noschema_14b"]


@lru_cache(None)
def load_run(stem: str) -> pd.DataFrame:
    with gzip.open(RUNS / f"{stem}.jsonl.gz", "rt", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    df = pd.DataFrame(rows).set_index("question_id", drop=False)
    df["has_join"] = df["gold_sql"].str.contains(r"\bJOIN\b", case=False, regex=True)
    df["called_search_values"] = df["trace"].apply(
        lambda t: any(s.get("tool") == "search_values" for s in t or []))
    df["n_tool_calls"] = df["trace"].apply(lambda t: sum(1 for s in t or [] if s.get("role") == "assistant"))
    return df


def test_run(name: str) -> pd.DataFrame:
    return load_run(TEST_RUNS[name][0])


def summary(df: pd.DataFrame) -> dict:
    return {
        "n": len(df),
        "strict": 100 * df["correct"].mean(),
        "relaxed": 100 * df["correct_relaxed"].mean(),
        "prompt_tokens": df["prompt_tokens"].mean(),
        "completion_tokens": df["completion_tokens"].mean(),
        "steps": df["steps"].mean(),
        "latency_s": df["llm_latency_s"].mean(),
        "latency_median_s": df["llm_latency_s"].median(),
        "latency_p90_s": df["llm_latency_s"].quantile(0.9),
        # output tokens per second of model time (decoding speed incl. prompt processing)
        "tokens_per_s": df["completion_tokens"].sum() / df["llm_latency_s"].sum(),
        "search_values_pct": 100 * df["called_search_values"].mean(),
    }


def _tool_steps(trace):
    """(tool, reply) for each tool reply, plus a final text-written submit (which gets no reply)."""
    out, pending = [], None
    for s in trace or []:
        if s.get("role") == "tool":
            out.append((s["name"].split("(")[0], s.get("content") or "", s["name"].endswith("(text)")))
            pending = None
        elif s.get("role") == "assistant":
            pending = None
        elif s.get("role") == "cube_analyst" and '"submit_answer"' in (s.get("content") or ""):
            pending = ("submit_answer", "Accepted.", True)
    if pending:
        out.append(pending)
    return out


def _rejected(tool: str, reply: str) -> bool:
    """A call the harness or Cube turned back (bad member, wrong type, repeat, Cube error)."""
    if tool == "run_query":
        return not reply.startswith('{"row_count"')
    if tool == "submit_answer":
        return not reply.startswith("Accepted")
    return not (reply.startswith("[") or reply.startswith("No value"))   # search_values


def tool_metrics(df: pd.DataFrame) -> dict:
    """Tool use per question: calls per tool, rejected calls, text-written calls, empty-result answers."""
    n = len(df)
    calls = {"search_values": 0, "run_query": 0, "submit_answer": 0}
    rejected = text = empty = 0
    for _, row in df.iterrows():
        steps = _tool_steps(row["trace"])
        for tool, reply, is_text in steps:
            calls[tool] = calls.get(tool, 0) + 1
            rejected += _rejected(tool, reply)
            text += is_text
        drafts = [r for t, r, _ in steps if t == "run_query" and r.startswith('{"row_count"')]
        if row["pred_query"] and drafts and json.loads(drafts[-1].split(', "rows"')[0] + "}")["row_count"] == 0:
            empty += 1
    total = sum(calls.values())
    return {
        "search_values_per_q": calls["search_values"] / n,
        "run_query_per_q": calls["run_query"] / n,
        "submit_per_q": calls["submit_answer"] / n,
        "rejected_per_q": rejected / n,
        "text_calls_pct": 100 * text / max(total, 1),
        "no_answer_pct": 100 * (~df["pred_query"].astype(bool)).mean(),
        "empty_answer_pct": 100 * empty / n,   # last draft before the answer returned 0 rows
    }


def tool_table(names=None) -> pd.DataFrame:
    names = names or MAIN_ARMS
    return pd.DataFrame([{"run": TEST_RUNS[k][1], **tool_metrics(test_run(k))} for k in names],
                        index=names)


def available(names):
    """The run names whose files are present (the no-hint prompt-offloading runs come later)."""
    return [n for n in names if (RUNS / f"{TEST_RUNS[n][0]}.jsonl.gz").exists()]


def summary_table(names=None) -> pd.DataFrame:
    names = names or available([n for n in TEST_RUNS if not n.startswith("hint_")])
    out = []
    for n in names:
        s = summary(test_run(n))
        s["run"] = TEST_RUNS[n][1]
        s["key"] = n
        out.append(s)
    return pd.DataFrame(out).set_index("key")


def mcnemar(a: str, b: str, col: str = "correct") -> dict:
    """Exact McNemar test on paired questions of two test runs."""
    x, y = test_run(a), test_run(b)
    ids = x.index.intersection(y.index)
    x, y = x.loc[ids, col], y.loc[ids, col]
    only_a = int((x & ~y).sum())
    only_b = int((~x & y).sum())
    p = binomtest(only_a, only_a + only_b, 0.5).pvalue if only_a + only_b else 1.0
    return {"a": a, "b": b, "only_a": only_a, "only_b": only_b, "p": p, "n": len(ids)}


def breakdown(name: str, by: str) -> pd.Series:
    df = test_run(name)
    return 100 * df.groupby(by)["correct"].mean()


# ---------------------------------------------------------------- error categories
ERROR_ORDER = ["No valid answer (error / call limit)", "Needs SQL Cube cannot express",
               "Extra columns only", "Wrong filter value", "Aggregation missing or wrong",
               "Wrong columns returned", "Ranking / limit missing", "Wrong field or join path"]
AGG = (exp.Count, exp.Sum, exp.Avg, exp.Min, exp.Max)


@lru_cache(None)
def _gold_facts(sql: str) -> dict:
    try:
        tree = sqlglot.parse_one(sql, read="sqlite")
    except Exception:
        return {}
    select = tree if isinstance(tree, exp.Select) else tree.find(exp.Select)
    if select is None:
        return {}
    cols = select.expressions
    where = select.args.get("where")
    literals = []
    if where is not None:
        for l in where.find_all(exp.Literal):
            if isinstance(l.parent, (exp.Like, exp.ILike)):
                literals.append(("like", l.this.strip("%").lower()))
            elif isinstance(l.parent, (exp.EQ, exp.In)) and l.this.strip():
                literals.append(("eq", l.this))
    beyond = (
        isinstance(tree, (exp.Union, exp.Except, exp.Intersect)) or tree.find(exp.With) is not None
        or len(list(tree.find_all(exp.Select))) > 1
        or tree.find(exp.Case, exp.If, exp.Substring, exp.Div, exp.Mul, exp.Sub) is not None
        or re.search(r"\b(IIF|INSTR|SUBSTR|STRFTIME|JULIANDAY)\b", sql, re.I) is not None
    )
    return {
        "n_cols": len(cols),
        "has_agg": any(c.find(*AGG) is not None for c in cols),
        "has_group": select.args.get("group") is not None,
        "has_limit": select.args.get("limit") is not None,
        "has_order": select.args.get("order") is not None,
        "literals": literals,
        "beyond_cube": beyond,
    }


def _num_eq(a, b) -> bool:
    try:
        return float(a) == float(b)
    except (TypeError, ValueError):
        return False


def _pred_values(q: dict) -> list:
    vals = []
    for f in (q or {}).get("filters", []) or []:
        for v in f.get("values", []) or []:
            vals.append(str(v))
        for sub in (f.get("and") or []) + (f.get("or") or []):
            vals += _pred_values({"filters": [sub]})
    return vals


def error_category(row) -> str | None:
    """Rule-based label for a wrong answer, from the predicted Cube query and the gold SQL.

    The rules run in order and the first match wins. They are heuristics: they say where the
    predicted query visibly differs from the gold SQL, not the model's reason for the mistake.
    """
    if row["correct"]:
        return None
    q = row["pred_query"]
    err = row.get("error")
    if (isinstance(err, str) and err) or not isinstance(q, dict) or not q:
        return ERROR_ORDER[0]
    if row["correct_relaxed"]:
        return ERROR_ORDER[2]
    g = _gold_facts(row["gold_sql"])
    if not g:
        return ERROR_ORDER[-1]
    if g["beyond_cube"]:
        return ERROR_ORDER[1]
    pv = _pred_values(q)

    def found(kind, lit):
        if kind == "like":
            return any(lit in v.lower() for v in pv)
        return any(v == lit for v in pv) or any(
            _num_eq(v, lit) for v in pv)

    if any(not found(k, l) for k, l in g["literals"]):
        return ERROR_ORDER[3]
    measures = q.get("measures") or []
    dims = q.get("dimensions") or []
    tdims = [t for t in q.get("timeDimensions") or [] if t.get("granularity")]
    if g["has_agg"] != bool(measures) and not g["has_group"]:
        return ERROR_ORDER[4]
    if g["n_cols"] != len(measures) + len(dims) + len(tdims):
        return ERROR_ORDER[5]
    if g["has_limit"] and g["has_order"] and not (q.get("limit") and q.get("order")):
        return ERROR_ORDER[6]
    return ERROR_ORDER[-1]


def error_counts(name: str) -> pd.Series:
    df = test_run(name)
    cats = df.apply(error_category, axis=1).dropna()
    return cats.value_counts().reindex(ERROR_ORDER, fill_value=0)


# ---------------------------------------------------------------- training logs
def training_log(name: str) -> dict:
    d = json.loads((TRAINING / f"{name}.json").read_text())
    train = pd.DataFrame([x for x in d["log"] if "loss" in x and "eval_loss" not in x])
    ev = pd.DataFrame([x for x in d["log"] if "eval_loss" in x])[["epoch", "eval_loss"]]
    ev = pd.concat([pd.DataFrame([{"epoch": 0.0, "eval_loss": d["eval_loss_before_training"]}]), ev],
                   ignore_index=True)
    final = d["log"][-1]
    return {"config": {k: v for k, v in d.items() if k != "log"}, "train": train, "eval": ev,
            "runtime_h": final.get("train_runtime", float("nan")) / 3600}


# ---------------------------------------------------------------- trace rendering
def render_trace(row, max_chars: int = 220) -> str:
    lines = []
    for s in row["trace"] or []:
        if s.get("role") == "assistant":
            if s.get("tool"):
                lines.append(f"> {s['tool']}({json.dumps(s.get('args', {}).get('query', s.get('args')), separators=(',', ':'))})")
            elif s.get("text"):
                lines.append(f"> (text) {s['text']}")
        elif s.get("role") == "tool":
            c = s.get("content", "")
            lines.append("  < " + (c if len(c) <= max_chars else c[:max_chars] + " ..."))
    return "\n".join(lines)
