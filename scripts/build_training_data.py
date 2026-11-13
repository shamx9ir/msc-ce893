"""Build verified (question -> Cube query) pairs from BIRD's train split for LoRA training.

For each train database:
  1. copy it into its own DuckDB file and generate its Cube model (same generator as the test DBs)
  2. start a Cube server for it (port 4100 by default)
  3. translate every gold SQL with harness.sql2cube and keep the pair only if the Cube query
     returns the same rows as the SQL on the original SQLite file
  4. save the database's schema text, which goes into the training prompts

BIRD's train databases are disjoint from the dev (test) databases, so nothing here can leak
test answers. Stop the main Cube server first (they share Cube's SQL/Store ports).

Usage:
    python scripts/build_training_data.py --bird-dir data/bird_train --out data/training
    python scripts/build_training_data.py --bird-dir data/bird_train --out data/training --db movie_platform
"""
import argparse
import collections
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import duckdb
import requests
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from build_duckdb import copy_db  # noqa: E402
from gen_cube_models import build_db  # noqa: E402
from harness.cube_client import CubeClient  # noqa: E402
from harness.evaluate import cube_rows, results_match, run_gold  # noqa: E402
from harness.sql2cube import Catalog, Translator, Unsupported  # noqa: E402


def start_cube(duck: Path, model_dir: Path, port: int, log: Path) -> subprocess.Popen:
    env = {**os.environ, "CUBEJS_DB_TYPE": "duckdb", "CUBEJS_DB_DUCKDB_DATABASE_PATH": str(duck),
           "CUBEJS_SCHEMA_PATH": str(model_dir), "PORT": str(port), "CUBEJS_DEV_MODE": "true",
           "CUBEJS_TELEMETRY": "false", "CUBEJS_CACHE_AND_QUEUE_DRIVER": "memory"}
    npx = shutil.which("npx") or "npx"
    proc = subprocess.Popen([npx, "cubejs-server"], cwd=ROOT / "cube", env=env,
                            stdout=open(log, "w"), stderr=subprocess.STDOUT)
    base = f"http://localhost:{port}"
    for _ in range(90):
        try:
            if requests.get(f"{base}/readyz", timeout=2).ok and \
                    requests.get(f"{base}/cubejs-api/v1/meta", timeout=10).ok:
                return proc
        except requests.RequestException:
            pass
        if proc.poll() is not None:
            break
        time.sleep(1)
    proc.terminate()
    raise RuntimeError(f"Cube did not start; see {log}")


def stop_cube(proc: subprocess.Popen) -> None:
    proc.terminate()
    try:
        proc.wait(15)
    except subprocess.TimeoutExpired:
        proc.kill()


def process_db(db: str, questions: list, args) -> collections.Counter:
    stats = collections.Counter()
    work = args.out / "work"
    work.mkdir(parents=True, exist_ok=True)
    sqlite_path = args.bird_dir / "train_databases" / db / f"{db}.sqlite"
    if sqlite_path.stat().st_size > args.max_db_mb * 2**20:
        stats["skipped_large_db"] = len(questions)
        return stats
    duck = work / f"{db}.duckdb"
    duck.unlink(missing_ok=True)
    con = duckdb.connect(str(duck))
    copy_db(con, sqlite_path, db, quiet=True)
    con.close()
    model_dir = work / f"model_{db}"
    model_dir.mkdir(exist_ok=True)
    (model_dir / f"{db}.yml").write_text(yaml.safe_dump(build_db(args.bird_dir, db, "train"),
                                                        sort_keys=False, allow_unicode=True, width=120))
    proc = start_cube(duck, model_dir, args.port, work / f"cube_{db}.log")
    try:
        cube = CubeClient(f"http://localhost:{args.port}/cubejs-api/v1", timeout=args.timeout)
        (args.out / "schemas").mkdir(exist_ok=True)
        (args.out / "schemas" / f"{db}.txt").write_text(cube.describe(db))
        tr = Translator(Catalog(cube.cubes_for(db)))
        with open(args.out / "verified.jsonl", "a") as f:
            for q in questions:
                try:
                    cq = tr.translate(q["SQL"])
                except Unsupported:
                    stats["unsupported"] += 1
                    continue
                try:
                    raw = cube.load(cq)
                    pred = cube_rows(raw)
                    gold = run_gold(sqlite_path, q["SQL"], timeout_s=args.timeout)
                except Exception:
                    stats["exec_error"] += 1
                    continue
                strict = results_match(pred, gold)
                if not (strict or results_match(pred, gold, allow_extra_columns=True)):
                    stats["mismatch"] += 1
                    continue
                if not gold:
                    stats["empty_result"] += 1  # an empty answer verifies nothing
                    continue
                stats["verified"] += 1
                f.write(json.dumps({"db_id": db, "question_id": q["question_id"], "question": q["question"],
                                    "evidence": q.get("evidence", ""), "sql": q["SQL"], "cube_query": cq,
                                    "strict_match": strict, "row_count": len(raw), "preview": raw[:10]},
                                   ensure_ascii=False, default=str) + "\n")
    finally:
        stop_cube(proc)
        if not args.keep_work:
            duck.unlink(missing_ok=True)
    return stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bird-dir", type=Path, default=Path("data/bird_train"))
    ap.add_argument("--out", type=Path, default=Path("data/training"))
    ap.add_argument("--db", action="append", help="only these databases (default: all)")
    ap.add_argument("--port", type=int, default=4100)
    ap.add_argument("--timeout", type=float, default=30)
    ap.add_argument("--max-db-mb", type=float, default=2000, help="skip SQLite files larger than this")
    ap.add_argument("--keep-work", action="store_true")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    by_db = collections.defaultdict(list)
    for q in json.load(open(args.bird_dir / "train.json")):
        q.setdefault("question_id", len(by_db[q["db_id"]]) + sum(map(len, by_db.values())))
        by_db[q["db_id"]].append(q)
    done_path = args.out / "stats.json"
    all_stats = json.loads(done_path.read_text()) if done_path.exists() else {}
    for db in sorted(args.db or by_db):
        if db in all_stats:
            continue
        t0 = time.time()
        try:
            stats = process_db(db, by_db[db], args)
        except Exception as e:
            stats = collections.Counter({"db_error": len(by_db[db])})
            print(f"{db}: ERROR {e}")
        stats["questions"] = len(by_db[db])
        all_stats[db] = dict(stats)
        done_path.write_text(json.dumps(all_stats, indent=1))
        print(f"{db}: {dict(stats)} ({time.time() - t0:.0f}s)", flush=True)
    total = collections.Counter()
    for s in all_stats.values():
        total.update(s)
    print("TOTAL", dict(total))


if __name__ == "__main__":
    main()
