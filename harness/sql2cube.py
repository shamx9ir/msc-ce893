"""Translate simple BIRD gold SQL into an equivalent Cube query.

Covers the SQL shapes a semantic layer can express: one SELECT over joined tables with
column projections, COUNT/SUM/AVG/MIN/MAX, WHERE conditions (=, <>, <, >, IN, LIKE, BETWEEN,
IS NULL, year/date matches written as STRFTIME or SUBSTR), GROUP BY, HAVING on an aggregate,
ORDER BY and LIMIT. Join conditions are dropped because Cube joins cubes itself.

Anything else raises Unsupported. A translation is only trusted after `verify` shows that the
Cube query returns the same rows as the SQL (see build_training_data.py).
"""
import re

import sqlglot
from sqlglot import exp

AGG = {exp.Sum: "sum", exp.Avg: "avg", exp.Min: "min", exp.Max: "max"}


class Unsupported(Exception):
    pass


class Catalog:
    """Maps tables/columns of one database to cube members, using Cube's /meta."""

    def __init__(self, cubes: list[dict]):
        self.dims, self.measures, self.types, self.pks = {}, {}, {}, {}
        for c in cubes:
            t = c["name"].lower()
            for d in c["dimensions"]:
                short = d["name"].split(".", 1)[1]
                self.dims[(t, short.lower())] = d["name"]
                self.types[d["name"]] = d["type"]
            for m in c["measures"]:
                self.measures[(t, m["name"].split(".", 1)[1].lower())] = m["name"]
            self.cube_names = getattr(self, "cube_names", set()) | {t}

    def dim(self, table: str, col: str) -> str:
        key = (table.lower(), re.sub(r"\W+", "_", col.strip()).strip("_").lower())
        if key not in self.dims:
            raise Unsupported(f"no dimension for {table}.{col}")
        return self.dims[key]

    def measure(self, table: str, name: str) -> str:
        key = (table.lower(), name.lower())
        if key not in self.measures:
            raise Unsupported(f"no measure {table}.{name}")
        return self.measures[key]


def _lit(node) -> str:
    if isinstance(node, exp.Literal):
        return node.this
    if isinstance(node, exp.Neg) and isinstance(node.this, exp.Literal):
        return "-" + node.this.this
    if isinstance(node, exp.Boolean):
        return "true" if node.this else "false"
    raise Unsupported(f"non-literal value {node.sql()}")


class Translator:
    def __init__(self, catalog: Catalog):
        self.cat = catalog

    # ---- name resolution ----------------------------------------------
    def _tables(self, select: exp.Select) -> dict:
        aliases = {}
        for t in select.find_all(exp.Table):
            name = t.name
            if name.lower() not in self.cat.cube_names:
                raise Unsupported(f"unknown table {name}")
            aliases[(t.alias or name).lower()] = name
        return aliases

    def _col(self, col: exp.Column) -> str:
        if col.table:
            table = self.aliases.get(col.table.lower())
            if not table:
                raise Unsupported(f"unknown alias {col.table}")
            return self.cat.dim(table, col.name)
        hits = []
        for table in set(self.aliases.values()):
            try:
                hits.append(self.cat.dim(table, col.name))
            except Unsupported:
                pass
        if len(hits) != 1:
            raise Unsupported(f"ambiguous or unknown column {col.name}")
        return hits[0]

    def _table_of(self, col: exp.Column) -> str:
        return self._col(col).split(".", 1)[0]

    def _measure(self, node) -> str:
        if isinstance(node, exp.Count):
            arg = node.this
            if isinstance(arg, exp.Distinct):
                inner = arg.expressions[0] if arg.expressions else None
                if isinstance(inner, exp.Column):
                    # COUNT(DISTINCT pk) == count of that cube
                    member = self._col(inner)
                    cube = member.split(".")[0]
                    return self.cat.measure(cube, "count")
                raise Unsupported("COUNT(DISTINCT expr)")
            if isinstance(arg, exp.Star) or arg is None:
                if len(set(self.aliases.values())) != 1:
                    # COUNT(*) over a join counts the many-side rows; use the most granular table
                    raise Unsupported("COUNT(*) over a join")
                return self.cat.measure(next(iter(self.aliases.values())), "count")
            if isinstance(arg, exp.Column):
                return self.cat.measure(self._table_of(arg), "count")
            raise Unsupported(f"COUNT({arg.sql()})")
        for cls, name in AGG.items():
            if isinstance(node, cls) and isinstance(node.this, exp.Column):
                member = self._col(node.this)
                cube, dim = member.split(".", 1)
                return self.cat.measure(cube, f"{name}_{dim}")
        raise Unsupported(f"expression {node.sql()}")

    def _member(self, node) -> tuple[str, str]:
        """Return (kind, member) for a select/order item."""
        node = node.this if isinstance(node, (exp.Alias, exp.Paren)) else node
        if isinstance(node, exp.Column):
            return "dimension", self._col(node)
        return "measure", self._measure(node)

    # ---- conditions -----------------------------------------------------
    def _year_or_date(self, node):
        """STRFTIME('%Y', col) / SUBSTR(col, 1, 4|7|10) -> (time member, precision)."""
        if isinstance(node, exp.Anonymous) and node.name.upper() == "STRFTIME":
            fmt, col = node.expressions[0], node.expressions[1]
            prec = {"%Y": 4, "%Y-%m": 7, "%Y-%m-%d": 10}.get(_lit(fmt))
        elif isinstance(node, (exp.TimeToStr,)):
            col, fmt = node.this, node.args.get("format")
            prec = {"%Y": 4, "%Y-%m": 7, "%Y-%m-%d": 10}.get(_lit(fmt))
        elif isinstance(node, exp.Substring):
            col = node.this
            start, length = node.args.get("start"), node.args.get("length")
            prec = 4 if _lit(start) == "1" and length and _lit(length) in ("4", "7", "10") else None
            prec = int(_lit(length)) if prec else None
        else:
            return None
        if not isinstance(col, exp.Column) or prec is None:
            raise Unsupported(f"date function {node.sql()}")
        member = self._col(col)
        if self.cat.types.get(member) != "time":
            raise Unsupported("date function on non-time column")
        return member, prec

    @staticmethod
    def _range(value: str, prec: int) -> list[str]:
        if prec == 4:
            return [f"{value}-01-01", f"{value}-12-31"]
        if prec == 7:
            y, m = value.split("-")
            last = {2: 29 if int(y) % 4 == 0 else 28}.get(int(m), 30 if int(m) in (4, 6, 9, 11) else 31)
            return [f"{value}-01", f"{value}-{last:02d}"]
        return [value, value]

    def _cond(self, node) -> list[dict]:
        node = node.this if isinstance(node, exp.Paren) else node
        if isinstance(node, exp.And):
            return self._cond(node.left) + self._cond(node.right)
        if isinstance(node, exp.Or):
            parts = []
            for side in (node.left, node.right):
                f = self._cond(side)
                parts.append(f[0] if len(f) == 1 else {"and": f})
            return [{"or": parts}]
        if isinstance(node, exp.Not) and isinstance(node.this, exp.Is):
            return [{"member": self._lhs(node.this.this), "operator": "set"}]
        if isinstance(node, exp.Is):
            return [{"member": self._lhs(node.this), "operator": "notSet"}]
        if isinstance(node, exp.In):
            return [{"member": self._lhs(node.this), "operator": "equals",
                     "values": [_lit(v) for v in node.expressions]}]
        if isinstance(node, exp.Between):
            m = self._lhs(node.this)
            if self.cat.types.get(m) == "time":
                return [{"member": m, "operator": "inDateRange",
                         "values": [_lit(node.args["low"]), _lit(node.args["high"])]}]
            return [{"member": m, "operator": "gte", "values": [_lit(node.args["low"])]},
                    {"member": m, "operator": "lte", "values": [_lit(node.args["high"])]}]
        if isinstance(node, exp.Like):
            pat = _lit(node.expression)
            core = pat.strip("%")
            member = self._lhs(node.this)
            if self.cat.types.get(member) == "time":
                if pat.endswith("%") and not pat.startswith("%") and re.fullmatch(r"\d{4}(-\d{2}){0,2}", core):
                    return [{"member": member, "operator": "inDateRange",
                             "values": self._range(core, len(core))}]
                raise Unsupported("LIKE on a time column")
            if "%" in core or "_" in core:
                raise Unsupported("complex LIKE")
            op = ("contains" if pat.startswith("%") and pat.endswith("%") else
                  "endsWith" if pat.startswith("%") else
                  "startsWith" if pat.endswith("%") else "equals")
            return [{"member": self._lhs(node.this), "operator": op, "values": [core]}]
        ops = {exp.EQ: "equals", exp.NEQ: "notEquals", exp.GT: "gt", exp.GTE: "gte",
               exp.LT: "lt", exp.LTE: "lte"}
        for cls, op in ops.items():
            if isinstance(node, cls):
                left, right = node.left, node.right
                if isinstance(left, exp.Literal) and not isinstance(right, exp.Literal):
                    left, right = right, left
                    op = {"gt": "lt", "gte": "lte", "lt": "gt", "lte": "gte"}.get(op, op)
                if isinstance(right, exp.Column):
                    return []  # a join condition written in WHERE
                ymd = self._year_or_date(left)
                if ymd:
                    member, prec = ymd
                    rng = self._range(_lit(right), prec)
                    if op == "equals":
                        return [{"member": member, "operator": "inDateRange", "values": rng}]
                    if op in ("gt", "gte"):
                        return [{"member": member, "operator": "afterDate" if op == "gt" else "afterOrOnDate",
                                 "values": [rng[1] if op == "gt" else rng[0]]}]
                    if op in ("lt", "lte"):
                        return [{"member": member, "operator": "beforeDate" if op == "lt" else "beforeOrOnDate",
                                 "values": [rng[0] if op == "lt" else rng[1]]}]
                    raise Unsupported("date inequality")
                member = self._lhs(left)
                val = _lit(right)
                if self.cat.types.get(member) == "time":
                    if op == "equals" and re.fullmatch(r"\d{4}-\d{2}-\d{2}", val):
                        return [{"member": member, "operator": "inDateRange", "values": [val, val]}]
                    raise Unsupported("comparison on a time column")
                return [{"member": member, "operator": op, "values": [val]}]
        raise Unsupported(f"condition {node.sql()}")

    def _lhs(self, node) -> str:
        if isinstance(node, exp.Column):
            return self._col(node)
        return self._measure(node)  # HAVING-style filter on an aggregate

    # ---- entry point ------------------------------------------------------
    def translate(self, sql: str) -> dict:
        try:
            tree = sqlglot.parse_one(sql, read="sqlite")
        except Exception as e:
            raise Unsupported(f"parse error: {e}")
        if not isinstance(tree, exp.Select):
            raise Unsupported(type(tree).__name__)
        if len(list(tree.find_all(exp.Select))) > 1:
            raise Unsupported("subquery")
        for j in tree.args.get("joins") or []:
            if j.side or j.kind not in ("", None, "INNER"):
                raise Unsupported("outer join")
        self.aliases = self._tables(tree)

        q: dict = {}
        for item in tree.expressions:
            kind, member = self._member(item)
            key = "dimensions" if kind == "dimension" else "measures"
            if member not in q.get(key, []):
                q.setdefault(key, []).append(member)
        group = tree.args.get("group")
        if group:
            for g in group.expressions:
                if not isinstance(g, exp.Column):
                    raise Unsupported("GROUP BY expression")
                if self._col(g) not in q.get("dimensions", []):
                    # Cube would add the column to the output; only fine if it is the grain key
                    raise Unsupported("GROUP BY a column that is not selected")
        filters = []
        if tree.args.get("where"):
            filters += self._cond(tree.args["where"].this)
        if tree.args.get("having"):
            filters += self._cond(tree.args["having"].this)
        if filters:
            q["filters"] = filters
        if tree.args.get("order"):
            order = {}
            for o in tree.args["order"].expressions:
                _, member = self._member(o.this)
                order[member] = "desc" if o.args.get("desc") else "asc"
            q["order"] = order
        if tree.args.get("limit"):
            q["limit"] = int(_lit(tree.args["limit"].expression))
            if tree.args.get("offset"):
                q["offset"] = int(_lit(tree.args["offset"].expression))
        if not q.get("measures") and not q.get("dimensions"):
            raise Unsupported("nothing selected")
        for member in q.get("order", {}):
            # Cube silently ignores ORDER BY on a dimension that is not in the query, so it has
            # to be returned as an extra column (the evaluator's relaxed score allows that).
            if member in self.cat.types and member not in q.get("dimensions", []):
                q.setdefault("dimensions", []).append(member)
        return q
