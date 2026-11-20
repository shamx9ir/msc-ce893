"""Turn verified pairs into chat-format fine-tuning files that match the harness exactly.

Each example is the conversation the harness would have on a perfect run:
  system  - the same system prompt ADK sends (instruction + ADK's agent line), with the
            database's Cube schema
  user    - the question and BIRD's hint
  tools mode:  [assistant calls search_values for a filter value, when the record has "searches"] ->
               assistant calls run_query(query) -> tool returns the real preview ->
               assistant calls submit_answer(query) -> "Accepted."
  single mode: assistant replies with the query JSON

Output (OpenAI/TRL "messages" format, plus "tools" for the tools mode):
  <out>/sft_tools_train.jsonl, sft_tools_val.jsonl, sft_single_train.jsonl, sft_single_val.jsonl
Validation = whole databases held out (default 5% of databases), so it measures transfer.

Prompt profiles (schema-offloading experiment, see harness/prompts.py):
  --profile full      harness v4.2 prompt with tool schemas (default; the original LoRA data)
  --profile lean      no query guide, no tool schemas, compact schema
  --profile noschema  training databases use the lean prompt (the model still has to learn Cube on
                      schemas it sees), and the --synthetic examples for the test databases use the
                      no-schema prompt, so the model learns those schemas into its weights.
                      Synthetic examples come from scripts/synth_schema_qa.py, never from BIRD dev.

Usage: python scripts/make_sft.py --data data/training --out data/sft
       python scripts/make_sft.py --profile lean --out data/sft_lean
       python scripts/make_sft.py --profile noschema --synthetic data/synth/synth.jsonl --out data/sft_noschema
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


def example(rec: dict, schema: str, mode: str, with_evidence: bool = True, profile: str = "full") -> dict:
    evidence = rec.get("evidence") if with_evidence else None
    msgs = [{"role": "system", "content": system_prompt(mode, rec["db_id"], schema, profile)},
            {"role": "user", "content": user_prompt(rec["question"], evidence,
                                                    schema if profile != "noschema" else None)}]
    q = rec["cube_query"]
    if mode == "single":
        msgs.append({"role": "assistant", "content": dumps(q)})
        return {"id": f"{rec['db_id']}/{rec['question_id']}", "messages": msgs}
    preview = dumps({"row_count": rec["row_count"], "rows": rec["preview"]})
    i = 0
    rec_fix = rec.get("recovery")  # 0 rows -> search -> retry, added by scripts/add_value_searches_v2.py
    searches = list(rec.get("searches") or [])
    late = [s for s in searches if rec_fix and s["dimension"] == rec_fix["member"]][:1]
    searches = [s for s in searches if s not in late]

    def search_turns(srch: dict, i: int) -> list:
        return [{"role": "assistant", "content": "", "tool_calls": [
                    tool_call(i, "search_values", {"dimension": srch["dimension"], "text": srch["text"]})]},
                {"role": "tool", "tool_call_id": f"call_{i}", "name": "search_values", "content": dumps(srch["result"])}]

    for srch in searches:  # value lookups, added by scripts/add_value_searches*.py
        msgs += search_turns(srch, i)
        i += 1
    if late:
        wrong_q = json.loads(dumps(q))
        for f in wrong_q.get("filters") or []:
            if isinstance(f, dict) and f.get("member") == rec_fix["member"] and f.get("values") == [rec_fix["right"]]:
                f["values"] = [rec_fix["wrong"]]
        # the wrong guess is context only (train_lora.py skips turns marked "train": false)
        msgs += [{"role": "assistant", "content": "", "train": False,
                  "tool_calls": [tool_call(i, "run_query", {"query": wrong_q})]},
                 {"role": "tool", "tool_call_id": f"call_{i}", "name": "run_query",
                  "content": dumps({"row_count": 0, "rows": []})}]
        msgs += search_turns(late[0], i + 1)
        i += 2
    msgs += [{"role": "assistant", "content": "", "tool_calls": [tool_call(i, "run_query", {"query": q})]},
             {"role": "tool", "tool_call_id": f"call_{i}", "name": "run_query", "content": preview},
             {"role": "assistant", "content": "", "tool_calls": [tool_call(i + 1, "submit_answer", {"query": q})]},
             {"role": "tool", "tool_call_id": f"call_{i + 1}", "name": "submit_answer", "content": "Accepted."}]
    row = {"id": f"{rec['db_id']}/{rec['question_id']}", "messages": msgs}
    if rec.get("synthetic"):
        row["synthetic"] = True
    if profile == "full":
        row["tools"] = TOOLS
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/training"))
    ap.add_argument("--out", type=Path, default=Path("data/sft"))
    ap.add_argument("--val-share", type=float, default=0.05)
    ap.add_argument("--no-evidence", action="store_true")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--profile", choices=["full", "lean", "noschema"], default="full")
    ap.add_argument("--synthetic", type=Path, help="synthetic test-schema examples (noschema profile only)")
    ap.add_argument("--synthetic-val-share", type=float, default=0.05)
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
    if (args.profile == "noschema") != bool(args.synthetic):
        sys.exit("--synthetic is needed for, and only used by, --profile noschema")
    synth = {"train": [], "val": []}
    if args.synthetic:
        srecs = [json.loads(l) for l in open(args.synthetic, encoding="utf-8")]
        assert all(r.get("synthetic") for r in srecs), "--synthetic must hold synth_schema_qa.py output"
        random.Random(args.seed).shuffle(srecs)
        n_val = round(len(srecs) * args.synthetic_val_share)
        synth = {"val": srecs[:n_val], "train": srecs[n_val:]}
    train_profile = "lean" if args.profile == "noschema" else args.profile
    counts = {}
    for mode in (("tools", "single") if args.profile == "full" else ("tools",)):
        for split in ("train", "val"):
            rows = [example(r, schemas[r["db_id"]], mode, not args.no_evidence, train_profile) for r in recs
                    if (r["db_id"] in val_dbs) == (split == "val")]
            rows += [example(r, "", mode, not args.no_evidence, "noschema") for r in synth[split]]
            if synth[split]:
                random.Random(args.seed).shuffle(rows)  # mix synthetic and BIRD examples
            path = args.out / f"sft_{mode}_{split}.jsonl"
            with open(path, "w", encoding="utf-8") as f:
                for row in rows:
                    f.write(dumps(row) + "\n")
            counts[path.name] = len(rows)
    (args.out / "split.json").write_text(dumps({"val_dbs": sorted(val_dbs), "counts": counts, "profile": args.profile,
                                                "synthetic": {k: len(v) for k, v in synth.items()}}), encoding="utf-8")
    print(counts, "val dbs:", sorted(val_dbs))


if __name__ == "__main__":
    main()
