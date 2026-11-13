"""Turn verified pairs into chat-format fine-tuning files that match the harness exactly.

Each example is the conversation the harness would have on a perfect run:
  system  - the same system prompt ADK sends (instruction + ADK's agent line), with the
            database's Cube schema
  user    - the question and BIRD's hint
  tools mode:  assistant calls run_query(query) -> tool returns the real preview ->
               assistant calls submit_answer(query) -> "Accepted."
  single mode: assistant replies with the query JSON

Output (OpenAI/TRL "messages" format, plus "tools" for the tools mode):
  <out>/sft_tools_train.jsonl, sft_tools_val.jsonl, sft_single_train.jsonl, sft_single_val.jsonl
Validation = whole databases held out (default 5% of databases), so it measures transfer.

Usage: python scripts/make_sft.py --data data/training --out data/sft
"""
import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from harness.agent import system_prompt  # noqa: E402
from harness.prompts import user_prompt  # noqa: E402

TOOLS = json.loads((ROOT / "harness" / "tool_schema.json").read_text(encoding="utf-8"))


def dumps(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, default=str)


def tool_call(i: int, name: str, args: dict) -> dict:
    return {"id": f"call_{i}", "type": "function", "function": {"name": name, "arguments": dumps(args)}}


def example(rec: dict, schema: str, mode: str, with_evidence: bool = True) -> dict:
    evidence = rec.get("evidence") if with_evidence else None
    msgs = [{"role": "system", "content": system_prompt(mode, rec["db_id"], schema)},
            {"role": "user", "content": user_prompt(rec["question"], evidence)}]
    q = rec["cube_query"]
    if mode == "single":
        msgs.append({"role": "assistant", "content": dumps(q)})
        return {"messages": msgs}
    preview = dumps({"row_count": rec["row_count"], "rows": rec["preview"]})
    msgs += [{"role": "assistant", "content": "", "tool_calls": [tool_call(0, "run_query", {"query": q})]},
             {"role": "tool", "tool_call_id": "call_0", "name": "run_query", "content": preview},
             {"role": "assistant", "content": "", "tool_calls": [tool_call(1, "submit_answer", {"query": q})]},
             {"role": "tool", "tool_call_id": "call_1", "name": "submit_answer", "content": "Accepted."}]
    return {"messages": msgs, "tools": TOOLS}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/training"))
    ap.add_argument("--out", type=Path, default=Path("data/sft"))
    ap.add_argument("--val-share", type=float, default=0.05)
    ap.add_argument("--no-evidence", action="store_true")
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    recs = list({(r["db_id"], r["question_id"]): r for r in
                 map(json.loads, open(args.data / "verified.jsonl", encoding="utf-8"))}.values())  # dedupe resumed runs
    test_dbs = {"formula_1", "student_club"} | {q["db_id"] for q in json.load(open(ROOT / "data/bird/dev.json", encoding="utf-8"))} \
        if (ROOT / "data/bird/dev.json").exists() else {"formula_1", "student_club"}
    leaked = [r for r in recs if r["db_id"] in test_dbs]
    assert not leaked, f"{len(leaked)} training pairs come from test databases"
    schemas = {p.stem: p.read_text(encoding="utf-8") for p in (args.data / "schemas").glob("*.txt")}
    dbs = sorted({r["db_id"] for r in recs})
    random.Random(args.seed).shuffle(dbs)
    val_dbs = set(dbs[: max(1, round(len(dbs) * args.val_share))])
    args.out.mkdir(parents=True, exist_ok=True)
    counts = {}
    for mode in ("tools", "single"):
        for split in ("train", "val"):
            rows = [example(r, schemas[r["db_id"]], mode, not args.no_evidence) for r in recs
                    if (r["db_id"] in val_dbs) == (split == "val")]
            path = args.out / f"sft_{mode}_{split}.jsonl"
            with open(path, "w", encoding="utf-8") as f:
                for row in rows:
                    f.write(dumps(row) + "\n")
            counts[path.name] = len(rows)
    (args.out / "split.json").write_text(dumps({"val_dbs": sorted(val_dbs), "counts": counts}), encoding="utf-8")
    print(counts, "val dbs:", sorted(val_dbs))


if __name__ == "__main__":
    main()
