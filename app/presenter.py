"""Turns an agent answer (Cube query + rows) into an A2UI v0.9 surface.

The surface uses one catalog: the A2UI basic catalog (Text, Card, Column, Row, Tabs) plus two
components of our own, `HighchartsChart` (a Highcharts options object) and `DataTable`.
The chart form is picked from the query's shape, so the same answer always looks the same,
whichever model wrote the query:

  one row of measures only               -> stat tiles (no chart)
  time dimension + measure(s)            -> line chart, one line per measure or per category
  one category + measure(s)              -> bar chart (horizontal when labels are long or many)
  two categories + one measure           -> stacked bars, second category as series
  label + one numeric column, no measure -> bar chart of that column
  anything else                          -> table only

The client renders whatever this module sends: it holds no chart logic of its own.
"""
from __future__ import annotations

import json
import re
from datetime import datetime

A2UI_VERSION = "v0.9"
CATALOG_ID = "https://github.com/shamx9ir/msc-ce893/app/catalog.json"

MAX_CHART_POINTS = 40      # categories on a bar chart (the table keeps all rows)
MAX_NUMERIC_POINTS = 150  # columns on a numeric axis such as years
MAX_SERIES = 8             # categorical palette size; more series go to the table only
MAX_TABLE_ROWS = 500


# ---- members and columns -----------------------------------------------------------------

def member_info(meta: dict) -> dict[str, dict]:
    """name -> {kind, type, title} for every member in Cube's /meta."""
    info = {}
    for c in meta.get("cubes", []):
        for d in c.get("dimensions", []):
            info[d["name"]] = {"kind": "dimension", "type": d.get("type", "string"),
                               "title": d.get("shortTitle") or d.get("title") or d["name"]}
        for m in c.get("measures", []):
            info[m["name"]] = {"kind": "measure", "type": "number",
                               "title": m.get("shortTitle") or m.get("title") or m["name"]}
    return info


def _title(name: str) -> str:
    """drivers.forename -> "Drivers forename"; used when Cube has no title."""
    words = re.sub(r"([a-z])([A-Z])", r"\1 \2", name.replace(".", " ").replace("_", " ")).split()
    return " ".join(words).capitalize()


def columns_for(query: dict, rows: list[dict], info: dict[str, dict]) -> list[dict]:
    """Columns in query order: [{key, title, kind, type}]. Time dimensions with a granularity come
    back from Cube as "<member>.<granularity>"."""
    keys: list[str] = []
    for td in query.get("timeDimensions") or []:
        if isinstance(td, dict) and td.get("granularity"):
            keys.append(f"{td['dimension']}.{td['granularity']}")
    keys += list(query.get("dimensions") or []) + list(query.get("measures") or [])
    if rows:  # keep anything Cube returned that the query did not name (e.g. order-by members),
        # except the bare time member Cube adds next to "<member>.<granularity>"
        grains = {k.rsplit(".", 1)[0] for k in keys if k.count(".") == 2}
        keys += [k for k in rows[0] if k not in keys and k not in grains]
    cols = []
    for k in dict.fromkeys(keys):
        base = k
        if k not in info and k.rsplit(".", 1)[0] in info:
            base = k.rsplit(".", 1)[0]
        i = info.get(base, {"kind": "dimension", "type": "string", "title": _title(k)})
        title = i["title"] if base == k else f"{i['title']} ({k.rsplit('.', 1)[1]})"
        cols.append({"key": k, "title": title, "kind": i["kind"], "type": i["type"]})
    return cols


def _num(v):
    if v is None or isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return v
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return int(f) if f.is_integer() and abs(f) < 2**53 else f


def _ms(v) -> int | None:
    """Cube time value -> epoch milliseconds (Highcharts datetime axis)."""
    if not v:
        return None
    try:
        dt = datetime.fromisoformat(str(v).replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        from datetime import timezone
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def _label(v) -> str:
    return "(empty)" if v is None or v == "" else str(v)


def _fmt(v) -> str:
    n = _num(v)
    if n is None:
        return _label(v)
    if isinstance(n, int):
        return f"{n:,}"
    return f"{n:,.2f}" if abs(n) < 1e6 else f"{n:,.0f}"


# ---- chart choice ------------------------------------------------------------------------

def _base_options(kind: str, x_title: str | None, y_title: str | None) -> dict:
    return {
        "chart": {"type": kind},
        "title": {"text": None},
        "xAxis": {"title": {"text": x_title}},
        "yAxis": {"title": {"text": y_title}},
        "legend": {"enabled": False},
        "tooltip": {"shared": kind == "line"},
        "credits": {"enabled": False},
    }


def choose_chart(query: dict, rows: list[dict], cols: list[dict]) -> dict:
    """Returns {"form": "kpi"|"chart"|"table"|"empty", "options": {...}?, "kpis": [...]?}."""
    if not rows:
        return {"form": "empty"}
    measures = [c for c in cols if c["kind"] == "measure"]
    times = [c for c in cols if c["kind"] == "dimension" and c["type"] == "time"]
    cats = [c for c in cols if c["kind"] == "dimension" and c["type"] in ("string", "boolean")]
    nums = [c for c in cols if c["kind"] == "dimension" and c["type"] == "number"]

    if measures and len(rows) == 1 and not times and not cats and not nums:
        return {"form": "kpi", "kpis": [{"title": m["title"], "value": _fmt(rows[0].get(m["key"]))}
                                        for m in measures]}

    # Time series: x = time, one line per measure, or per category when there is one measure.
    if times and measures and len(rows) >= 2 and len(times) == 1 and not nums:
        t = times[0]
        rows_sorted = sorted(rows, key=lambda r: _ms(r.get(t["key"])) or 0)
        if not cats and len(measures) <= MAX_SERIES:
            series = [{"name": m["title"], "data": [[_ms(r.get(t["key"])), _num(r.get(m["key"]))]
                                                    for r in rows_sorted]} for m in measures]
        elif len(cats) == 1 and len(measures) == 1:
            groups: dict[str, list] = {}
            for r in rows_sorted:
                groups.setdefault(_label(r.get(cats[0]["key"])), []).append(
                    [_ms(r.get(t["key"])), _num(r.get(measures[0]["key"]))])
            if len(groups) > MAX_SERIES:
                return {"form": "table"}
            series = [{"name": k, "data": v} for k, v in groups.items()]
        else:
            return {"form": "table"}
        opts = _base_options("line", t["title"], measures[0]["title"] if len(measures) == 1 else None)
        opts["xAxis"]["type"] = "datetime"
        opts["legend"]["enabled"] = len(series) > 1
        opts["series"] = series
        return {"form": "chart", "options": opts}

    # Category (or a numeric dimension used as a category, e.g. a year) + measure(s).
    keys = cats + nums
    if measures and len(keys) == 1 and len(rows) >= 2 and len(measures) <= MAX_SERIES:
        x = keys[0]
        if x["type"] == "number":  # e.g. a year: keep every point, in axis order unless ordered
            shown = rows if query.get("order") else sorted(rows, key=lambda r: _num(r.get(x["key"])) or 0)
            shown = shown[:MAX_NUMERIC_POINTS]
        else:
            shown = rows[:MAX_CHART_POINTS]
        labels = [_label(r.get(x["key"])) for r in shown]
        horizontal = x["type"] != "number" and (len(labels) > 12 or max(map(len, labels)) > 14)
        opts = _base_options("bar" if horizontal else "column", x["title"],
                             measures[0]["title"] if len(measures) == 1 else None)
        opts["xAxis"]["categories"] = labels
        opts["series"] = [{"name": m["title"], "data": [_num(r.get(m["key"])) for r in shown]}
                          for m in measures]
        opts["legend"]["enabled"] = len(measures) > 1
        return {"form": "chart", "options": opts, "truncated": len(rows) > len(shown)}

    if len(measures) == 1 and len(keys) == 2 and len(rows) >= 2 and not times:
        x, s = keys
        xs = list(dict.fromkeys(_label(r.get(x["key"])) for r in rows))
        ss = list(dict.fromkeys(_label(r.get(s["key"])) for r in rows))
        if len(ss) > MAX_SERIES and len(xs) <= MAX_SERIES:
            x, s, xs, ss = s, x, ss, xs
        if len(ss) <= MAX_SERIES and len(xs) <= MAX_CHART_POINTS:
            grid = {(_label(r.get(x["key"])), _label(r.get(s["key"]))): _num(r.get(measures[0]["key"]))
                    for r in rows}
            horizontal = len(xs) > 12 or max(map(len, xs)) > 14
            opts = _base_options("bar" if horizontal else "column", x["title"], measures[0]["title"])
            opts["xAxis"]["categories"] = xs
            opts["plotOptions"] = {"series": {"stacking": "normal"}}
            opts["legend"] = {"enabled": True, "title": {"text": s["title"]}}
            opts["series"] = [{"name": sv, "data": [grid.get((xv, sv)) for xv in xs]} for sv in ss]
            return {"form": "chart", "options": opts}
        # Too many values to stack (e.g. forename + surname): one bar per row, labels joined.
        shown = rows[:MAX_CHART_POINTS]
        opts = _base_options("bar", f"{keys[0]['title']}, {keys[1]['title']}", measures[0]["title"])
        opts["xAxis"]["categories"] = [f"{_label(r.get(keys[0]['key']))} {_label(r.get(keys[1]['key']))}"
                                       for r in shown]
        opts["series"] = [{"name": measures[0]["title"], "data": [_num(r.get(measures[0]["key"])) for r in shown]}]
        return {"form": "chart", "options": opts, "truncated": len(rows) > len(shown)}

    # No measure: a label column plus one numeric column ("list drivers and their points").
    if not measures and len(nums) == 1 and len(cats) >= 1 and 2 <= len(rows):
        label_cols = cats
        shown = rows[:MAX_CHART_POINTS]
        labels = [" ".join(_label(r.get(c["key"])) for c in label_cols) for r in shown]
        opts = _base_options("bar", " ".join(c["title"] for c in label_cols), nums[0]["title"])
        opts["xAxis"]["categories"] = labels
        opts["series"] = [{"name": nums[0]["title"], "data": [_num(r.get(nums[0]["key"])) for r in shown]}]
        return {"form": "chart", "options": opts, "truncated": len(rows) > len(shown)}

    return {"form": "table"}


def display_query(query: dict, info: dict[str, dict]) -> dict:
    """The answer query plus any measure it sorts by but does not return. "Which 10 countries have
    the most circuits?" is answered with country names only (BIRD scores the columns asked for),
    but the chart needs the circuit counts it ranks by. Returns the query unchanged otherwise."""
    order = query.get("order") or {}
    keys = list(order) if isinstance(order, dict) else [o[0] for o in order if o]
    extra = [k for k in keys if info.get(k, {}).get("kind") == "measure"
             and k not in (query.get("measures") or [])]
    # A date listed both as a plain dimension and with a granularity ("each month") gives one row
    # per day in Cube; for display, group by the granularity only.
    grained = {t.get("dimension") for t in query.get("timeDimensions") or []
               if isinstance(t, dict) and t.get("granularity")}
    dims = [d for d in query.get("dimensions") or [] if d not in grained]
    if not extra and len(dims) == len(query.get("dimensions") or []):
        return query
    shown = dict(query, measures=list(query.get("measures") or []) + extra, dimensions=dims)
    if isinstance(shown.get("order"), dict):
        shown["order"] = {k: v for k, v in shown["order"].items() if k not in grained} or None
    return {k: v for k, v in shown.items() if v not in ([], None)}


# ---- A2UI messages -----------------------------------------------------------------------

def _msg(kind: str, body: dict) -> dict:
    return {"version": A2UI_VERSION, kind: body}


def _text(cid: str, text, variant: str = "body") -> dict:
    return {"id": cid, "component": "Text", "text": text, "variant": variant}


def build_surface(surface_id: str, *, question: str, query: dict | None, rows: list[dict] | None,
                  meta: dict, model_label: str, stats: dict, error: str | None = None,
                  trace: list | None = None, display: tuple[dict, list[dict]] | None = None,
                  sql: str | None = None) -> list[dict]:
    """All A2UI messages for one answer: createSurface, updateComponents, updateDataModel.
    `display` is (display_query, its rows) when it differs from the answer query."""
    info = member_info(meta)
    rows = rows or []
    comps: list[dict] = []
    data: dict = {}
    # The page shows the question above the answers, so the surface starts with the run's stats.
    header = [_text("stats", stats_line(model_label, stats, len(rows) if query and not error else None),
                    "caption")]
    tabs = []

    if error or query is None:
        comps.append(_text("error", f"**No answer.** {error or 'The model did not submit a query.'}"))
        body = ["error"]
    else:
        shown_q, shown_rows = display if display else (query, rows)
        cols = columns_for(shown_q, shown_rows, info)
        choice = choose_chart(shown_q, shown_rows, cols)
        added = [info.get(m, {}).get("title", m) for m in shown_q.get("measures") or []
                 if m not in (query.get("measures") or [])]
        data["table"] = {"columns": [{"key": c["key"], "title": c["title"],
                                      "numeric": c["kind"] == "measure" or c["type"] == "number",
                                      "grouping": c["kind"] == "measure",   # 1,234 for measures, not years
                                      "time": c["type"] == "time"}
                                     for c in cols],
                         "rows": shown_rows[:MAX_TABLE_ROWS]}
        comps.append({"id": "table", "component": "DataTable",
                      "columns": {"path": "/table/columns"}, "rows": {"path": "/table/rows"}})
        if choice["form"] == "empty":
            comps.append(_text("answer", "The query ran but returned **no rows**."))
            main = "answer"
        elif choice["form"] == "kpi":
            tiles = []
            for i, k in enumerate(choice["kpis"]):
                comps += [
                    {"id": f"kpi{i}", "component": "Card", "child": f"kpi{i}_col"},
                    {"id": f"kpi{i}_col", "component": "Column", "children": [f"kpi{i}_v", f"kpi{i}_t"]},
                    _text(f"kpi{i}_v", {"path": f"/kpis/{i}/value"}, "h2"),
                    _text(f"kpi{i}_t", {"path": f"/kpis/{i}/title"}, "caption"),
                ]
                tiles.append(f"kpi{i}")
            data["kpis"] = choice["kpis"]
            comps.append({"id": "answer", "component": "Row", "children": tiles})
            main = "answer"
        elif choice["form"] == "chart":
            data["chart"] = choice["options"]
            comps.append({"id": "chart", "component": "HighchartsChart", "options": {"path": "/chart"},
                          "height": 340})
            if choice.get("truncated"):
                comps += [_text("chart_note", f"The chart shows the first {len(choice['options']['xAxis']['categories'])} rows; "
                                              "the table has all of them.", "caption"),
                          {"id": "chart_col", "component": "Column", "children": ["chart", "chart_note"]}]
                main = "chart_col"
            else:
                main = "chart"
        else:
            main = "table"
        dropped = [info.get(d, {}).get("title", d) for d in query.get("dimensions") or []
                   if d not in (shown_q.get("dimensions") or [])]
        if dropped and main != "table" and choice["form"] != "empty":
            grain = next((t.get("granularity") for t in query.get("timeDimensions") or []
                          if isinstance(t, dict) and t.get("granularity")), "period")
            comps += [_text("grain_note", f"Shown per {grain}. The submitted query also lists "
                                          f"{', '.join(dropped)} as a plain column, which gives one row per date.",
                            "caption"),
                      {"id": "grain_col", "component": "Column", "children": [main, "grain_note"]}]
            main = "grain_col"
        if added:
            comps += [_text("added_note", f"{', '.join(added)} is shown because the query sorts by it; "
                                          "the submitted answer leaves it out.", "caption"),
                      {"id": "main_col", "component": "Column", "children": [main, "added_note"]}]
            main = "main_col"
        if main != "table":
            tabs.append({"title": "Chart" if choice["form"] == "chart" else "Answer", "child": main})
        tabs.append({"title": "Table", "child": "table"})
        body = []

    if query is not None:
        cubes = cubes_in(query)
        text = ("**Cubes used:** " + ", ".join(cubes) + (" (Cube joins them)" if len(cubes) > 1 else "") +
                "\n\n```json\n" + json.dumps(query, indent=2) + "\n```")
        if sql:
            text += "\n\n**SQL that Cube generated:**\n\n```sql\n" + sql + "\n```"
        comps.append(_text("query", text))
        tabs.append({"title": "Cube query", "child": "query"})
    if trace:
        comps.append(_text("trace", trace_markdown(trace)))
        tabs.append({"title": "Agent steps", "child": "trace"})
    if tabs:
        comps.append({"id": "tabs", "component": "Tabs", "tabs": tabs})
        body.append("tabs")

    comps += header
    comps.append({"id": "root", "component": "Column", "children": ["stats"] + body})
    msgs = [
        _msg("createSurface", {"surfaceId": surface_id, "catalogId": CATALOG_ID}),
        _msg("updateComponents", {"surfaceId": surface_id, "components": comps}),
    ]
    if data:
        msgs.append(_msg("updateDataModel", {"surfaceId": surface_id, "path": "/", "value": data}))
    return msgs


def cubes_in(query: dict) -> list[str]:
    """The cubes a query touches, in first-use order; more than one means Cube writes a join."""
    names = list(query.get("measures") or []) + list(query.get("dimensions") or [])
    names += [t.get("dimension", "") for t in query.get("timeDimensions") or [] if isinstance(t, dict)]

    def walk(fs):
        for f in fs or []:
            if isinstance(f, dict):
                names.append(f.get("member", ""))
                walk(f.get("or"))
                walk(f.get("and"))
    walk(query.get("filters"))
    order = query.get("order") or {}
    names += list(order) if isinstance(order, dict) else [o[0] for o in order if o]
    return list(dict.fromkeys(n.split(".")[0] for n in names if isinstance(n, str) and "." in n))


def stats_line(model_label: str, stats: dict, n_rows: int | None) -> str:
    parts = [model_label]
    if stats.get("steps") is not None:
        parts.append(f"{stats['steps']} LLM call{'s' if stats['steps'] != 1 else ''}")
    if stats.get("wall_s") is not None:
        parts.append(f"{stats['wall_s']:.1f} s")
    if stats.get("prompt_tokens"):
        parts.append(f"{stats['prompt_tokens'] + stats.get('completion_tokens', 0):,} tokens")
    if n_rows is not None:
        parts.append(f"{n_rows:,} row{'s' if n_rows != 1 else ''}")
    return " · ".join(parts)


def trace_markdown(trace: list) -> str:
    """The harness trace as a numbered list, so a viewer can see the tool calls the model made."""
    lines = []
    for t in trace:
        if t.get("role") == "assistant" and t.get("tool"):
            args = json.dumps(t.get("args") or {}, ensure_ascii=False)
            lines.append(f"1. **{t['tool']}** `{_clip(args, 300)}`")
        elif t.get("role") == "tool":
            lines.append(f"   ↳ {_clip(str(t.get('content', '')), 300).replace('`', 'ʹ')}")
        elif t.get("role") == "harness":
            lines.append(f"1. *harness:* {t.get('content')}")
        elif t.get("content"):
            lines.append(f"1. *model text:* {_clip(str(t['content']), 300)}")
    return "\n".join(lines) or "No steps recorded."


def _clip(s: str, n: int) -> str:
    s = s.replace("\n", " ")
    return s if len(s) <= n else s[: n - 1] + "…"
