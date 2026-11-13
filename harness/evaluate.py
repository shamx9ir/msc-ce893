"""Run the harness over BIRD questions and score execution accuracy against the gold SQL.

A prediction is correct when the rows returned by its Cube query equal the rows returned by
BIRD's gold SQL on the original SQLite database, compared as sets (the BIRD convention).
Column order may differ (Cube orders columns its own way), and values are normalised
(numbers rounded, date formats unified).

Example:
    python -m harness.evaluate --db student_club --model qwen2.5-coder:7b \
        --setup harness --out runs/harness_qwen7b.jsonl
"""
import argparse
import itertools
import json
import re
import sqlite3
import time
from pathlib import Path

from .agent import Harness, make_model
from .cube_client import CubeClient

NUM_RE = re.compile(r"^-?\d+(\.\d+)?([eE][-+]?\d+)?$")
TS_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})[T ](\d{2}:\d{2}:\d{2})(\.\d+)?Z?$")


def norm_value(v):
    if v is None:
        return None
    if isinstance(v, bool):
        return float(v)
    if isinstance(v, (int, float)):
        return round(float(v), 4)
    s = str(v).strip()
    if NUM_RE.match(s):
        return round(float(s), 4)
    m = TS_RE.match(s)
    if m:
        return m.group(1) if m.group(2) == "00:00:00" else f"{m.group(1)} {m.group(2)}"
    return s


def norm_rows(rows) -> set:
    return {tuple(norm_value(v) for v in r) for r in rows}


def results_match(pred_rows: list[tuple], gold_rows: list[tuple], allow_extra_columns: bool = False) -> bool:
    """Set equality of rows, ignoring column order.

    allow_extra_columns: the prediction may return additional columns (Cube must return any
    dimension it sorts by, which SQL can leave out); some choice of its columns must then
    reproduce the gold rows exactly, with the same number of distinct rows.
    """
    gold = norm_rows(gold_rows)
    if not pred_rows or not gold_rows:
        return norm_rows(pred_rows) == gold
    pw, gw = len(pred_rows[0]), len(gold_rows[0])
    if pw < gw or (pw > gw and not allow_extra_columns):
        return False
    if pw > 6:  # avoid combinatorial blow-up; require same order
        return pw == gw and norm_rows(pred_rows) == gold
    if pw > gw and len(norm_rows(pred_rows)) != len(gold):
        return False
    for perm in itertools.permutations(range(pw), gw):
        if norm_rows([tuple(r[i] for i in perm) for r in pred_rows]) == gold:
            return True
    return False


def run_gold(sqlite_path: Path, sql: str, timeout_s: float = 30) -> list[tuple]:
    con = sqlite3.connect(f"file:{sqlite_path}?mode=ro", uri=True)
    con.text_factory = lambda b: b.decode("utf-8", errors="replace")
    deadline = time.time() + timeout_s
    con.set_progress_handler(lambda: int(time.time() > deadline), 10_000)
    try:
        return con.execute(sql).fetchall()
    finally:
        con.close()


def cube_rows(rows: list[dict]) -> list[tuple]:
    return [tuple(r.values()) for r in rows]


def load_questions(bird_dir: Path, dbs: list[str], ids: set[int] | None, limit: int | None):
    qs = [q for q in json.load(open(bird_dir / "dev.json", encoding="utf-8")) if q["db_id"] in dbs]
    if ids:
        qs = [q for q in qs if q["question_id"] in ids]
    return qs[:limit] if limit else qs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bird-dir", type=Path, default=Path("data/bird"))
    ap.add_argument("--db", action="append", required=True)
    ap.add_argument("--ids-file", type=Path, help="JSON list of question_ids to evaluate (e.g. the test split)")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--model", required=True,
                    help='LiteLLM model id, e.g. ollama_chat/qwen2.5-coder:7b or openai/<name> for vLLM; '
                         'a bare name is treated as an Ollama model')
    ap.add_argument("--base-url", help="model server URL, e.g. http://localhost:11434 (Ollama) "
                                       "or http://localhost:8000/v1 (vLLM)")
    ap.add_argument("--api-key")
    ap.add_argument("--num-ctx", type=int, default=16384, help="Ollama context window (same for both arms)")
    ap.add_argument("--cube-url", default="http://localhost:4000/cubejs-api/v1")
    ap.add_argument("--mode", choices=["tools", "single"], default="tools")
    ap.add_argument("--max-steps", type=int, default=8)
    ap.add_argument("--no-evidence", action="store_true", help="hide BIRD's hint text from the model")
    ap.add_argument("--setup", default="harness", help="label stored with each result, e.g. harness or lora")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    ids = set(json.load(open(args.ids_file, encoding="utf-8"))) if args.ids_file else None
    questions = load_questions(args.bird_dir, args.db, ids, args.limit)
    harness = Harness(make_model(args.model, args.base_url, args.api_key, num_ctx=args.num_ctx),
                      CubeClient(args.cube_url), mode=args.mode, max_steps=args.max_steps)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if args.out.exists():  # resume
        done = {json.loads(line)["question_id"] for line in open(args.out, encoding="utf-8")}
    correct = relaxed_total = total = 0
    with open(args.out, "a", encoding="utf-8") as f:
        for q in questions:
            if q["question_id"] in done:
                continue
            db_path = args.bird_dir / "dev_databases" / q["db_id"] / f"{q['db_id']}.sqlite"
            res = harness.run(q["db_id"], q["question"], None if args.no_evidence else q.get("evidence"))
            ok = relaxed = False
            exec_error = res.error
            if res.query is not None and res.error is None:
                try:
                    pred = cube_rows(harness.cube.load(res.query))
                    gold = run_gold(db_path, q["SQL"])
                    ok = results_match(pred, gold)
                    relaxed = ok or results_match(pred, gold, allow_extra_columns=True)
                except Exception as e:
                    exec_error = str(e)
            correct += ok
            relaxed_total += relaxed
            total += 1
            rec = {"question_id": q["question_id"], "db_id": q["db_id"], "difficulty": q["difficulty"],
                   "setup": args.setup, "model": args.model, "mode": args.mode,
                   "question": q["question"], "evidence": q.get("evidence"), "gold_sql": q["SQL"],
                   "pred_query": res.query, "correct": ok, "correct_relaxed": relaxed, "error": exec_error, "steps": res.steps,
                   "prompt_tokens": res.prompt_tokens, "completion_tokens": res.completion_tokens,
                   "llm_latency_s": round(res.llm_latency_s, 3), "wall_s": round(res.wall_s, 3),
                   "trace": res.trace}
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
            f.flush()
            print(f"[{total}/{len(questions) - len(done)}] q{q['question_id']} {'OK ' if ok else 'BAD'} "
                  f"acc={correct / total:.1%} steps={res.steps} {exec_error or ''}"[:200])
    if total:
        print(f"\nExecution accuracy: {correct}/{total} = {correct / total:.1%} "
              f"(allowing extra columns: {relaxed_total}/{total} = {relaxed_total / total:.1%})")


if __name__ == "__main__":
    main()
