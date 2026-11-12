"""Copy selected BIRD SQLite databases into one DuckDB file that Cube reads.

Each BIRD database becomes a DuckDB schema of the same name
(e.g. formula_1.races, student_club.member). Column types follow the
SQLite declarations; values that do not fit (e.g. date '0000-00-00')
become NULL via TRY_CAST.

Usage:
    python scripts/build_duckdb.py --bird-dir data/bird/dev_20240627 \
        --out data/bird.duckdb --db formula_1 --db student_club
"""
import argparse
import sqlite3
from pathlib import Path

import duckdb
import pandas as pd

TYPE_MAP = {"INTEGER": "BIGINT", "INT": "BIGINT", "REAL": "DOUBLE", "FLOAT": "DOUBLE",
            "DATE": "DATE", "DATETIME": "TIMESTAMP", "TEXT": "VARCHAR"}


def duck_type(decl: str) -> str:
    decl = (decl or "TEXT").upper().split("(")[0].strip()
    return TYPE_MAP.get(decl, "VARCHAR")


def copy_db(con: duckdb.DuckDBPyConnection, sqlite_path: Path, schema: str) -> None:
    src = sqlite3.connect(sqlite_path)
    src.text_factory = lambda b: b.decode("utf-8", errors="replace")
    con.execute(f'CREATE SCHEMA IF NOT EXISTS "{schema}"')
    tables = [r[0] for r in src.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")]
    for table in tables:
        cols = src.execute(f'PRAGMA table_info("{table}")').fetchall()
        df = pd.read_sql_query(f'SELECT * FROM "{table}"', src)
        # Keep everything as text first so DuckDB does the typed cast.
        df = df.astype(object).where(df.notna(), None).astype(str).where(df.notna(), None)
        select = ", ".join(
            f'TRY_CAST("{c[1]}" AS {duck_type(c[2])}) AS "{c[1]}"' for c in cols)
        con.register("df_src", df)
        con.execute(f'CREATE OR REPLACE TABLE "{schema}"."{table}" AS SELECT {select} FROM df_src')
        con.unregister("df_src")
        n = con.execute(f'SELECT count(*) FROM "{schema}"."{table}"').fetchone()[0]
        print(f"  {schema}.{table}: {n} rows")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bird-dir", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--db", action="append", required=True)
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(args.out))
    for db in args.db:
        print(db)
        copy_db(con, args.bird_dir / "dev_databases" / db / f"{db}.sqlite", db)
    con.close()


if __name__ == "__main__":
    main()
