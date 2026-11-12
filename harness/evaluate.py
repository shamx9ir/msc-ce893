"""Run the harness over BIRD questions and score execution accuracy against the gold SQL.

A prediction is correct when the rows returned by its Cube query equal the rows returned by
BIRD's gold SQL on the original SQLite database, compared as sets (the BIRD convention).
Column order may differ (Cube orders columns its own way), and values are normalised
(numbers rounded, date formats unified).

Example:
    python -m harness.evaluate --db student_club --model qwen2.5-coder:7b \
        --base-url http://localhost:11434/v1 --setup harness --out runs/harness_qwen7b.jsonl
"""
import argparse
import itertools
import json
import re
import sqlite3
import time
from pathlib import Path

from .agent import Harness
from .cube_client import CubeClient
from .llm import LLMClient

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


def results_match(pred_rows: list[tuple], gold_rows: list[tuple]) -> bool:
    gold = norm_rows(gold_rows)
    if not pred_rows or not gold_rows:
        return norm_rows(pred_rows) == gold
    width = len(pred_rows[0])
    if width != len(gold_rows[0]):
        return False
    if width > 6:  # avoid factorial blow-up; require same order
        return norm_rows(pred_rows) == gold
    for perm in itertools.permutations(range(width)):
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
    qs = [q for q in json.load(open(bird_dir / "dev.json")) if q["db_id"] in dbs]
    if ids:
        qs = [q for q in qs if q["question_id"] in ids]
    return qs[:limit] if limit else qs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bird-dir", type=Path, default=Path("data/bird"))
    ap.add_argument("--db", action="append", required=True)
    ap.add_argument("--ids-file", type=Path, help="JSON list of question_ids to evaluate (e.g. the test split)")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--model", required=True)
    ap.add_argument("--base-url", default="http://localhost:11434/v1")
    ap.add_argument("--api-key", default="not-needed")
    ap.add_argument("--cube-url", default="http://localhost:4000/cubejs-api/v1")
    ap.add_argument("--mode", choices=["tools", "single"], default="tools")
    ap.add_argument("--max-steps", type=int, default=8)
    ap.add_argument("--no-evidence", action="store_true", help="hide BIRD's hint text from the model")
    ap.add_argument("--setup", default="harness", help="label stored with each result, e.g. harness or lora")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    ids = set(json.load(open(args.ids_file))) if args.ids_file else None
    questions = load_questions(args.bird_dir, args.db, ids, args.limit)
    harness = Harness(LLMClient(model=args.model, base_url=args.base_url, api_key=args.api_key),
                      CubeClient(args.cube_url), mode=args.mode, max_steps=args.max_steps)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if args.out.exists():  # resume
        done = {json.loads(line)["question_id"] for line in open(args.out)}
    correct = total = 0
    with open(args.out, "a") as f:
        for q in questions:
            if q["question_id"] in done:
                continue
            db_path = args.bird_dir / "dev_databases" / q["db_id"] / f"{q['db_id']}.sqlite"
            res = harness.run(q["db_id"], q["question"], None if args.no_evidence else q.get("evidence"))
            ok, exec_error = False, res.error
            if res.query is not None:
                try:
                    pred = cube_rows(harness.cube.load(res.query))
                    ok = results_match(pred, run_gold(db_path, q["SQL"]))
                except Exception as e:
                    exec_error = str(e)
            correct += ok
            total += 1
            rec = {"question_id": q["question_id"], "db_id": q["db_id"], "difficulty": q["difficulty"],
                   "setup": args.setup, "model": args.model, "mode": args.mode,
                   "question": q["question"], "evidence": q.get("evidence"), "gold_sql": q["SQL"],
                   "pred_query": res.query, "correct": ok, "error": exec_error, "steps": res.steps,
                   "prompt_tokens": res.prompt_tokens, "completion_tokens": res.completion_tokens,
                   "llm_latency_s": round(res.llm_latency_s, 3), "wall_s": round(res.wall_s, 3),
                   "trace": res.trace}
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
            f.flush()
            print(f"[{total}/{len(questions) - len(done)}] q{q['question_id']} {'OK ' if ok else 'BAD'} "
                  f"acc={correct / total:.1%} steps={res.steps} {exec_error or ''}"[:200])
    if total:
        print(f"\nExecution accuracy: {correct}/{total} = {correct / total:.1%}")


if __name__ == "__main__":
    main()
