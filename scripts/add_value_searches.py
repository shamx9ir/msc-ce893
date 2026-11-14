"""Add search_values steps to the verified training pairs, so the LoRA model keeps the habit of
looking up exact stored values (the harness relies on it for names and categories).

For each string `equals` filter, the step is added when the value is not spelled out verbatim in
the question or hint (where a real run would need to look it up), and for a random 20% of the
others. The search result is what the harness's search_values would return: up to 20 distinct
stored values containing the text, read from the BIRD-train SQLite file.

Writes a "searches" list into each record of training_data/verified.jsonl (in place).
Usage: python scripts/add_value_searches.py --bird-train data/bird_train/train_databases
"""
import argparse
import json
import random
import re
import sqlite3
from pathlib import Path


def ident(name: str) -> str:  # same as scripts/gen_cube_models.py
    s = re.sub(r"\W+", "_", name.strip()).strip("_") or "col"
    return s if not s[0].isdigit() else f"c_{s}"


def member_map(con) -> dict:
    out = {}
    for (t,) in con.execute("SELECT name FROM sqlite_master WHERE type='table'"):
        for row in con.execute(f'PRAGMA table_info("{t}")'):
            out[f"{ident(t)}.{ident(row[1])}"] = (t, row[1])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("training_data/verified.jsonl"))
    ap.add_argument("--bird-train", type=Path, default=Path("data/bird_train/train_databases"))
    ap.add_argument("--share", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    recs = [json.loads(l) for l in open(args.data, encoding="utf-8")]
    cons, maps, added = {}, {}, 0
    for r in recs:
        r["searches"] = []
        db = r["db_id"]
        if db not in cons:
            cons[db] = sqlite3.connect(args.bird_train / db / f"{db}.sqlite")
            cons[db].text_factory = lambda b: b.decode("utf-8", "replace")
            maps[db] = member_map(cons[db])
        text = r["question"] + " " + (r.get("evidence") or "")
        for f in r["cube_query"].get("filters") or []:
            if not (isinstance(f, dict) and f.get("operator") == "equals" and len(f.get("values") or []) == 1):
                continue
            v, m = f["values"][0], f.get("member")
            if not isinstance(v, str) or m not in maps[db] or len(r["searches"]) >= 2:
                continue
            t, c = maps[db][m]
            try:
                sample = cons[db].execute(f'SELECT typeof("{c}") FROM "{t}" WHERE "{c}" IS NOT NULL LIMIT 1').fetchone()
            except sqlite3.Error:
                continue
            if not sample or sample[0] != "text":
                continue
            if v in text and rng.random() >= args.share:
                continue
            rows = cons[db].execute(f'SELECT DISTINCT "{c}" FROM "{t}" WHERE "{c}" LIKE ? LIMIT 20',
                                    (f"%{v}%",)).fetchall()
            values = [x[0] for x in rows]
            if v not in values:
                continue  # keep only searches that find the value the query uses
            r["searches"].append({"dimension": m, "text": v, "result": values})
            added += 1
    with open(args.data, "w", encoding="utf-8") as fh:
        for r in recs:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{added} searches in {sum(bool(r['searches']) for r in recs)} of {len(recs)} examples")


if __name__ == "__main__":
    main()
