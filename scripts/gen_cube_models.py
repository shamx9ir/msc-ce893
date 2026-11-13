"""Generate Cube data models (YAML) for BIRD databases.

The model is derived mechanically from what BIRD ships with each database:
  * dev_tables.json            -> tables, columns, types, primary/foreign keys
  * database_description/*.csv -> column descriptions and value notes
so that the semantic layer carries the same information the SQL schema does.

For every table we create one cube with
  * a dimension per column (date-like TEXT columns become `time` dimensions)
  * a `count` measure
  * sum / avg / min / max measures for numeric, non-key columns
  * joins in both directions for every foreign key

Usage:
    python scripts/gen_cube_models.py --bird-dir data/bird --out cube/model --db formula_1 --db student_club
"""
import argparse
import csv
import io
import json
import re
from collections import defaultdict
from pathlib import Path

import sqlite3
import yaml

NUMERIC = {"integer", "real", "number"}
TEMPORAL = {"date", "datetime", "time"}
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}([ T]\d{2}:\d{2}(:\d{2})?)?$")


def ident(name: str) -> str:
    """Cube member names must be identifiers."""
    s = re.sub(r"\W+", "_", name.strip()).strip("_")
    return s if not s[0].isdigit() else f"c_{s}"


def read_descriptions(desc_dir: Path, table: str) -> dict:
    path = desc_dir / f"{table}.csv"
    if not path.exists():
        return {}
    raw = path.read_bytes()
    for enc in ("utf-8-sig", "latin-1"):
        try:
            text = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    out = {}
    for row in csv.DictReader(io.StringIO(text)):
        row = {(k or "").strip(): (v or "").strip() for k, v in row.items()}
        col = row.get("original_column_name", "")
        if not col:
            continue
        parts = [row.get("column_description", "")]
        vd = row.get("value_description", "")
        if vd:
            vd = re.sub(r"(?i)commonsense evidence:\s*", "", vd)
            parts.append(vd)
        desc = " ".join(p for p in parts if p)
        out[col.lower()] = re.sub(r"\s+", " ", desc)[:400]
    return out


def is_date_text(con, table: str, col: str) -> bool:
    rows = con.execute(f'SELECT CAST("{col}" AS TEXT) FROM "{table}" WHERE "{col}" IS NOT NULL LIMIT 50').fetchall()
    return bool(rows) and all(DATE_RE.match(r[0]) for r in rows)


def build_db(bird_dir: Path, db: str, split: str = "dev") -> dict:
    con = sqlite3.connect(f"file:{bird_dir / f'{split}_databases' / db / f'{db}.sqlite'}?mode=ro", uri=True)
    con.text_factory = lambda b: b.decode('utf-8', errors='replace')
    meta = next(t for t in json.load(open(bird_dir / f"{split}_tables.json")) if t["db_id"] == db)
    tables = meta["table_names_original"]
    cols = meta["column_names_original"]  # [table_idx, name]
    types = meta["column_types"]
    desc_dir = bird_dir / f"{split}_databases" / db / "database_description"

    pk_of = defaultdict(list)
    for pk in meta["primary_keys"]:
        for c in (pk if isinstance(pk, list) else [pk]):
            pk_of[cols[c][0]].append(cols[c][1])

    fk_cols = set()
    joins = defaultdict(dict)  # table -> {target_table: (sql, relationship)}
    for child, parent in meta["foreign_keys"]:
        ct, cc = cols[child]
        pt, pc = cols[parent]
        if ct == pt:
            continue
        fk_cols.add((ct, cc.lower()))
        cn, pn = tables[ct], tables[pt]
        on = f'{{{cn}}}."{cc}" = {{{pn}}}."{pc}"'
        joins[cn].setdefault(pn, (on, "many_to_one"))
        joins[pn].setdefault(cn, (on, "one_to_many"))

    cubes = []
    for ti, table in enumerate(tables):
        descs = read_descriptions(desc_dir, table)
        pks = [p.lower() for p in pk_of.get(ti, [])]
        dims, measures = [], [{"name": "count", "type": "count",
                               "description": f"Number of {table} rows"}]
        for ci, (t_idx, col) in enumerate(cols):
            if t_idx != ti:
                continue
            ctype = types[ci]
            dname = ident(col)
            dim = {"name": dname, "sql": f'{{CUBE}}."{col}"'}
            if ctype in NUMERIC:
                dim["type"] = "number"
            elif ctype in TEMPORAL or (ctype == "text" and is_date_text(con, table, col)):
                dim["type"] = "time"
                dim["sql"] = f'TRY_CAST({{CUBE}}."{col}" AS TIMESTAMP)'
            else:
                dim["type"] = "string"
            if len(pks) == 1 and col.lower() == pks[0]:
                dim["primary_key"] = True
                dim["public"] = True
            if descs.get(col.lower()):
                dim["description"] = descs[col.lower()]
            dims.append(dim)

            is_key = col.lower() in pks or (ti, col.lower()) in fk_cols or col.lower().endswith("id")
            if ctype in NUMERIC and not is_key:
                for agg in ("sum", "avg", "min", "max"):
                    measures.append({"name": f"{agg}_{dname}", "type": agg,
                                     "sql": f'{{CUBE}}."{col}"',
                                     "description": f"{agg.upper()} of {table}.{col}"})
        if len(pks) != 1:
            key_cols = pks or [c for t, c in cols if t == ti]
            concat = " || '|' || ".join(f'CAST({{CUBE}}."{c}" AS VARCHAR)' for c in key_cols)
            dims.insert(0, {"name": "row_key", "sql": concat, "type": "string",
                            "primary_key": True, "public": False})
        # sql_alias avoids clashes such as table zip_code with a column zip_code in DuckDB
        cube = {"name": table, "sql_table": f'{db}."{table}"', "sql_alias": f"t_{ident(table)}",
                "meta": {"db": db}, "dimensions": dims, "measures": measures}
        if joins.get(table):
            # {CUBE} for the cube's own side: inside cube zip_code, {zip_code} would mean its
            # zip_code dimension, not the cube.
            cube["joins"] = [{"name": tgt, "sql": on.replace(f"{{{table}}}", "{CUBE}"), "relationship": rel}
                             for tgt, (on, rel) in joins[table].items()]
        cubes.append(cube)
    return {"cubes": cubes}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bird-dir", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--db", action="append", required=True)
    ap.add_argument("--split", default="dev", choices=["dev", "train"])
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for db in args.db:
        model = build_db(args.bird_dir, db, args.split)
        path = args.out / f"{db}.yml"
        path.write_text("# Generated by scripts/gen_cube_models.py - edit the generator, not this file.\n"
                        + yaml.safe_dump(model, sort_keys=False, allow_unicode=True, width=120))
        print(f"{path}: {len(model['cubes'])} cubes")


if __name__ == "__main__":
    main()
