"""Synthetic question -> Cube query pairs that teach a LoRA the schema of the test databases.

For the "noschema" prompt profile (harness/prompts.py) the model gets no schema in its prompt, so it
has to know formula_1's and student_club's members from training. These two databases have no
training questions (their BIRD dev questions are the test set), so this script writes questions
from the schema alone: templates over the cubes, columns, column descriptions, joins and sampled
stored values. No BIRD dev question, hint or SQL is read, except by the optional --audit, which
only reports near-duplicates afterwards.

Every query is run against Cube and kept only if it runs and returns rows; the record holds the
same preview (and, for some string filters, the same search_values step) as training_data/verified.jsonl.

Usage (Cube running):
  python scripts/synth_schema_qa.py --db formula_1 --n 800 --db student_club --n 500 --out data/synth/synth.jsonl --audit
"""
import argparse
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from harness.cube_client import CubeClient, CubeError  # noqa: E402

AGG_WORD = {"avg": ["average", "mean"], "sum": ["total", "sum of the"], "max": ["highest", "maximum"],
            "min": ["lowest", "minimum"]}


def words(name: str) -> str:
    name = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", name)
    return name.replace("_", " ").lower().strip()


class Schema:
    def __init__(self, db: str):
        y = yaml.safe_load(open(ROOT / "cube" / "model" / f"{db}.yml", encoding="utf-8"))
        self.cubes = {}
        for c in y["cubes"]:
            measures = {m["name"] for m in c.get("measures", [])}
            dims = [d for d in c["dimensions"] if d["name"] != "row_key"]
            for d in dims:
                d["agg"] = d["type"] == "number" and f"sum_{d['name']}" in measures
            self.cubes[c["name"]] = {"dims": dims, "joins": [j["name"] for j in c.get("joins", [])]}

    def label(self, rng, cube: str, d: dict) -> str:
        """How a question names a column: its description when short, else its name in words."""
        desc = (d.get("description") or "").strip().rstrip(".")
        if desc and len(desc) <= 45 and "\n" not in desc and rng.random() < 0.5:
            desc = desc[0].lower() + desc[1:]
            return desc[4:] if desc.startswith("the ") else desc
        return words(d["name"])


class Generator:
    def __init__(self, cube: CubeClient, db: str, rng: random.Random):
        self.cube, self.db, self.rng, self.s = cube, db, rng, Schema(db)
        self._values = {}

    def values(self, member: str) -> list:
        if member not in self._values:
            try:
                rows = self.cube.load({"dimensions": [member], "limit": 300})
            except CubeError:
                rows = []
            vals = [r[member] for r in rows if r.get(member) not in (None, "")]
            self._values[member] = [v for v in vals if len(str(v)) <= 60]
        return self._values[member]

    def pick_value(self, member: str):
        vals = self.values(member)
        return self.rng.choice(vals) if vals else None

    def eq_filter(self, member: str, kind: str):
        v = self.pick_value(member)
        if v is None:
            return None, None
        if kind == "time":
            year = str(v)[:4]
            if not year.isdigit():
                return None, None
            return ({"member": member, "operator": "inDateRange", "values": [f"{year}-01-01", f"{year}-12-31"]},
                    f"in {year}")
        return ({"member": member, "operator": "equals", "values": [str(v)]},
                str(v) if kind == "number" else f'"{v}"')

    def make(self, cube: str, d: dict):
        """One question about column d of cube (several templates, chosen at random)."""
        r, s = self.rng, self.s
        ent = words(cube)
        m = f"{cube}.{d['name']}"
        lab = s.label(r, cube, d)
        others = [o for o in s.cubes[cube]["dims"] if o is not d]
        templates = ["count_eq", "list_eq", "list_by"]
        if d["agg"]:
            templates += ["agg", "agg", "agg_eq", "num_cmp", "top_n"]
        if d["type"] == "string":
            templates += ["top_group", "count_eq"]
        if s.cubes[cube]["joins"]:
            templates += ["join_count", "join_list"]
        t = r.choice(templates)
        if t == "count_eq":
            f, txt = self.eq_filter(m, d["type"])
            if not f:
                return None
            q = {"measures": [f"{cube}.count"], "filters": [f]}
            if d["type"] == "time":
                question = r.choice([f"How many {ent} have a {lab} {txt}?", f"Count the {ent} with {lab} {txt}."])
            else:
                question = r.choice([f"How many {ent} have {lab} {txt}?", f"Count the {ent} whose {lab} is {txt}.",
                                     f"What is the number of {ent} with {lab} {txt}?"])
        elif t == "list_eq":  # return d, filter on another column
            o = r.choice(others) if others else None
            if not o:
                return None
            f, txt = self.eq_filter(f"{cube}.{o['name']}", o["type"])
            if not f:
                return None
            q = {"dimensions": [m], "filters": [f]}
            question = r.choice([f"List the {lab} of {ent} whose {s.label(r, cube, o)} is {txt}.",
                                 f"What is the {lab} of the {ent} with {s.label(r, cube, o)} {txt}?"])
        elif t == "list_by":  # filter on d, return another column
            o = r.choice(others) if others else None
            if not o:
                return None
            f, txt = self.eq_filter(m, d["type"])
            if not f:
                return None
            q = {"dimensions": [f"{cube}.{o['name']}"], "filters": [f]}
            question = f"Give the {s.label(r, cube, o)} of every {ent} whose {lab} is {txt}."
        elif t in ("agg", "agg_eq"):
            agg = r.choice(list(AGG_WORD))
            q = {"measures": [f"{cube}.{agg}_{d['name']}"]}
            question = f"What is the {r.choice(AGG_WORD[agg])} {lab} of {ent}"
            if t == "agg_eq":
                o = r.choice([o for o in others if o["type"] == "string"] or [None])
                if not o:
                    return None
                f, txt = self.eq_filter(f"{cube}.{o['name']}", "string")
                if not f:
                    return None
                q["filters"] = [f]
                question += f" whose {s.label(r, cube, o)} is {txt}"
            question += "?"
        elif t == "num_cmp":
            v = self.pick_value(m)
            if v is None:
                return None
            op, word = r.choice([("gt", "greater than"), ("lt", "less than"), ("gte", "at least")])
            q = {"measures": [f"{cube}.count"], "filters": [{"member": m, "operator": op, "values": [str(v)]}]}
            question = f"How many {ent} have {lab} {word} {v}?"
        elif t == "top_n":
            o = r.choice([o for o in others if o["type"] == "string"] or [None])
            if not o:
                return None
            n, dirn, word = r.choice([1, 3, 5]), *r.choice([("desc", "highest"), ("asc", "lowest")])
            om = f"{cube}.{o['name']}"
            q = {"dimensions": [om, m], "order": {m: dirn}, "limit": n}
            what = f"the {ent}" if n == 1 else f"the {n} {ent}"
            question = f"List the {s.label(r, cube, o)} and {lab} of {what} with the {word} {lab}."
        elif t == "top_group":
            q = {"dimensions": [m], "order": {f"{cube}.count": "desc"}, "limit": 1}
            question = r.choice([f"Which {lab} has the most {ent}?", f"What is the most common {lab} among {ent}?"])
        else:  # join_count / join_list: filter on a column of a joined cube
            other = r.choice(s.cubes[cube]["joins"])
            cand = [o for o in s.cubes[other]["dims"] if o["type"] in ("string", "time")]
            if not cand:
                return None
            o = r.choice(cand)
            f, txt = self.eq_filter(f"{other}.{o['name']}", o["type"])
            if not f:
                return None
            olab = f"{words(other)} {s.label(r, other, o)}"
            if t == "join_count":
                q = {"measures": [f"{cube}.count"], "filters": [f]}
                question = f"How many {ent} are linked to a {olab} {txt}?"
            else:
                q = {"dimensions": [m], "filters": [f]}
                question = f"List the {lab} of {ent} that have a {olab} {txt}."
        return {"template": t, "question": question, "cube_query": q}

    def verify(self, rec: dict) -> dict | None:
        try:
            rows = self.cube.load(rec["cube_query"])
        except CubeError:
            return None
        if not rows:
            return None
        if rec["cube_query"].get("measures") and not rec["cube_query"].get("dimensions"):
            val = next(iter(rows[0].values()))
            if val in (None, "0", 0):
                return None
        if len(rows) > 2000:
            return None
        rec.update(row_count=len(rows), preview=rows[:10], searches=[])
        for f in rec["cube_query"].get("filters") or []:  # value lookups like add_value_searches.py
            v = (f.get("values") or [None])[0]
            if (f.get("operator") == "equals" and isinstance(v, str) and not v.isdigit()
                    and self.rng.random() < 0.4):
                found = self.cube.dimension_values(f["member"], v)
                if v in found:
                    rec["searches"].append({"dimension": f["member"], "text": v, "result": found})
        return rec

    def run(self, n: int) -> list[dict]:
        cols = [(c, d) for c, v in self.s.cubes.items() for d in v["dims"]]
        out, seen_q, seen_query, tries = [], set(), set(), 0
        while len(out) < n and tries < n * 20:
            tries += 1
            cube, d = cols[tries % len(cols)]  # round-robin, so every column is asked about
            rec = self.make(cube, d)
            if not rec:
                continue
            key = json.dumps(rec["cube_query"], sort_keys=True)
            if rec["question"] in seen_q or key in seen_query:
                continue
            rec = self.verify(rec)
            if not rec:
                continue
            seen_q.add(rec["question"])
            seen_query.add(key)
            rec.update(db_id=self.db, question_id=f"synth-{self.db}-{len(out)}", evidence=None, synthetic=True)
            out.append(rec)
        return out


def audit(recs: list[dict], bird_dir: Path) -> None:
    """Report synthetic questions that read almost like a test question (word Jaccard >= 0.7)."""
    dev = json.load(open(bird_dir / "dev.json", encoding="utf-8"))
    test_ids = set(json.load(open(ROOT / "splits" / "test_ids.json", encoding="utf-8")))
    tok = lambda s: set(re.findall(r"[a-z0-9]+", s.lower()))
    tests = [(q["question_id"], tok(q["question"])) for q in dev if q["question_id"] in test_ids]
    close = 0
    for r in recs:
        a = tok(r["question"])
        best = max(len(a & b) / len(a | b) for _, b in tests)
        close += best >= 0.7
    print(f"audit: {close} of {len(recs)} synthetic questions have word overlap >= 0.7 with a test question")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", action="append", required=True)
    ap.add_argument("--n", type=int, action="append", required=True, help="examples per --db, in order")
    ap.add_argument("--out", type=Path, default=Path("data/synth/synth.jsonl"))
    ap.add_argument("--cube-url", default="http://localhost:4000/cubejs-api/v1")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--audit", action="store_true", help="report near-duplicates of test questions")
    ap.add_argument("--bird-dir", type=Path, default=Path("data/bird"))
    args = ap.parse_args()
    assert len(args.db) == len(args.n)
    cube = CubeClient(args.cube_url)
    recs = []
    for db, n in zip(args.db, args.n):
        got = Generator(cube, db, random.Random(f"{args.seed}-{db}")).run(n)
        print(db, len(got), dict(Counter(r["template"] for r in got)),
              "with search:", sum(bool(r["searches"]) for r in got), flush=True)
        recs += got
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
    print(f"wrote {len(recs)} to {args.out}")
    if args.audit:
        audit(recs, args.bird_dir)


if __name__ == "__main__":
    main()
