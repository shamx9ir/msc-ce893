"""search_values v2: teach the LoRA model to look values up instead of guessing them.

The first LoRA data (scripts/add_value_searches.py) had a search_values step in only 22% of the
examples, and the trained models almost stopped calling the tool. This version:
  1. adds a lookup for every string `equals` filter on a text column (up to 2 per example),
  2. searches with a short, lower-case piece of the wording (as a model would type it after
     reading the question), not with the exact stored value; it falls back to longer text when
     the short piece does not bring the right value into the first 20 results,
  3. turns a share of the examples into recovery examples: the first run_query uses a plausible
     wrong spelling (the question's own casing, or a case change), gets 0 rows, then the model
     calls search_values and reruns with the stored value. Only queries with dimensions are used,
     so an empty result is certain, and the wrong spelling is checked not to be stored.

The search result mimics the harness: up to 20 distinct stored values containing the text,
case-insensitive (SQLite LIKE), read from the BIRD-train SQLite files.

Reads <data>/verified.jsonl, writes <out>/verified.jsonl (+ copies schemas/) with:
  "searches": [{"dimension", "text", "result"}]
  "recovery": {"member", "wrong", "right"} or absent   (make_sft.py builds the turns)
Usage: python scripts/add_value_searches_v2.py --data training_data --out data/training_sv2 \
           --bird-train data/bird_train/train_databases
"""
import argparse
import json
import random
import re
import shutil
import sqlite3
from pathlib import Path

from add_value_searches import member_map

STOP = {"the", "and", "of", "in", "for", "de", "la", "le", "der", "inc", "ltd", "co", "&"}


def fragments(value: str, text: str) -> list[str]:
    """Search texts to try, shortest first: a word of the value that the question/hint also uses,
    then the value's longest word, then the whole value, all lower case."""
    words = [w for w in re.findall(r"[\w'.]+", value) if len(w) >= 3 and w.lower() not in STOP]
    low = text.lower()
    seen, out = set(), []
    in_text = sorted((w for w in words if w.lower() in low), key=len, reverse=True)
    longest = sorted(words, key=len, reverse=True)
    for cand in in_text[:1] + longest[:1] + [value]:
        c = cand.lower()
        if c and c not in seen:
            seen.add(c)
            out.append(c)
    return out


def wrong_spelling(value: str, text: str, stored) -> str | None:
    """A spelling a model would plausibly guess that is not stored: the question's own casing
    first, then case changes."""
    cands = []
    m = re.search(re.escape(value), text, re.IGNORECASE)
    if m:
        cands.append(m.group(0))
    cands += [value.title(), value.lower(), value.upper(), value.capitalize()]
    for c in cands:
        if c != value and not stored(c):
            return c
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("training_data"))
    ap.add_argument("--out", type=Path, default=Path("data/training_sv2"))
    ap.add_argument("--bird-train", type=Path, default=Path("data/bird_train/train_databases"))
    ap.add_argument("--recovery-share", type=float, default=0.12,
                    help="share of eligible examples turned into 0-rows -> search -> retry examples")
    ap.add_argument("--max-searches", type=int, default=2)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    recs = [json.loads(l) for l in open(args.data / "verified.jsonl", encoding="utf-8")]
    cons, maps = {}, {}
    stats = {"filters": 0, "searched": 0, "fallback_full": 0, "not_found": 0, "recovery": 0}
    for r in recs:
        r["searches"] = []
        r.pop("recovery", None)
        db = r["db_id"]
        if db not in cons:
            cons[db] = sqlite3.connect(args.bird_train / db / f"{db}.sqlite")
            cons[db].text_factory = lambda b: b.decode("utf-8", "replace")
            maps[db] = member_map(cons[db])
        con = cons[db]
        text = r["question"] + " " + (r.get("evidence") or "")
        for f in r["cube_query"].get("filters") or []:
            if len(r["searches"]) >= args.max_searches:
                break
            if not (isinstance(f, dict) and f.get("operator") == "equals" and len(f.get("values") or []) == 1):
                continue
            v, m = f["values"][0], f.get("member")
            if not isinstance(v, str) or m not in maps[db] or any(s["dimension"] == m for s in r["searches"]):
                continue
            t, c = maps[db][m]
            try:
                sample = con.execute(f'SELECT typeof("{c}") FROM "{t}" WHERE "{c}" IS NOT NULL LIMIT 1').fetchone()
            except sqlite3.Error:
                continue
            if not sample or sample[0] != "text":
                continue
            stats["filters"] += 1
            for i, frag in enumerate(fragments(v, text)):
                rows = con.execute(f'SELECT DISTINCT "{c}" FROM "{t}" WHERE "{c}" LIKE ? LIMIT 20',
                                   (f"%{frag}%",)).fetchall()
                values = [x[0] for x in rows]
                if v in values:
                    r["searches"].append({"dimension": m, "text": frag, "result": values})
                    stats["searched"] += 1
                    stats["fallback_full"] += frag == v.lower() and i > 0
                    break
            else:
                stats["not_found"] += 1
        # recovery: only for non-aggregate queries (dimensions, no measures), so 0 rows is certain
        q = r["cube_query"]
        if r["searches"] and q.get("dimensions") and not q.get("measures") and rng.random() < args.recovery_share:
            s = r["searches"][0]
            t, c = maps[db][s["dimension"]]
            right = next(f["values"][0] for f in q["filters"] if isinstance(f, dict) and f.get("member") == s["dimension"]
                         and f.get("operator") == "equals")
            stored = lambda x: con.execute(f'SELECT 1 FROM "{t}" WHERE "{c}" = ? LIMIT 1', (x,)).fetchone() is not None
            wrong = wrong_spelling(right, text, stored)
            if wrong:
                r["recovery"] = {"member": s["dimension"], "wrong": wrong, "right": right}
                stats["recovery"] += 1
    args.out.mkdir(parents=True, exist_ok=True)
    with open(args.out / "verified.jsonl", "w", encoding="utf-8") as fh:
        for r in recs:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    if (args.data / "schemas").resolve() != (args.out / "schemas").resolve():
        shutil.copytree(args.data / "schemas", args.out / "schemas", dirs_exist_ok=True)
    n = sum(bool(r["searches"]) for r in recs)
    print(f"{stats} -> {n} of {len(recs)} examples have a search_values step "
          f"({100 * n / len(recs):.0f}%), {stats['recovery']} recovery examples")


if __name__ == "__main__":
    main()
