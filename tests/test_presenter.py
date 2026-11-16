"""Chart choice and A2UI messages of the demo app (no Cube or model needed)."""
from app.presenter import CATALOG_ID, build_surface, choose_chart, columns_for, member_info

META = {"cubes": [
    {"name": "races", "dimensions": [
        {"name": "races.year", "type": "number", "shortTitle": "Year"},
        {"name": "races.name", "type": "string", "shortTitle": "Name"},
        {"name": "races.date", "type": "time", "shortTitle": "Date"}],
     "measures": [{"name": "races.count", "type": "number", "shortTitle": "Count"},
                  {"name": "races.avg_round", "type": "number", "shortTitle": "Avg round"}]},
    {"name": "circuits", "dimensions": [{"name": "circuits.country", "type": "string", "shortTitle": "Country"}],
     "measures": [{"name": "circuits.count", "type": "number", "shortTitle": "Count"}]},
]}
INFO = member_info(META)


def form(query, rows):
    return choose_chart(query, rows, columns_for(query, rows, INFO))


def test_single_value_is_a_stat_tile():
    c = form({"measures": ["races.count"]}, [{"races.count": "976"}])
    assert c["form"] == "kpi" and c["kpis"] == [{"title": "Count", "value": "976"}]


def test_category_and_measure_is_a_bar_chart():
    rows = [{"circuits.country": c, "circuits.count": str(n)} for c, n in [("USA", 8), ("France", 7), ("UK", 4)]]
    c = form({"dimensions": ["circuits.country"], "measures": ["circuits.count"]}, rows)
    o = c["options"]
    assert c["form"] == "chart" and o["chart"]["type"] == "column"
    assert o["xAxis"]["categories"] == ["USA", "France", "UK"] and o["series"][0]["data"] == [8, 7, 4]


def test_numeric_dimension_is_sorted_categories():
    rows = [{"races.year": "2009", "races.count": "17"}, {"races.year": "2008", "races.count": "18"}]
    o = form({"dimensions": ["races.year"], "measures": ["races.count"]}, rows)["options"]
    assert o["xAxis"]["categories"] == ["2008", "2009"]


def test_time_granularity_is_a_line_chart():
    q = {"measures": ["races.count"], "timeDimensions": [{"dimension": "races.date", "granularity": "year"}]}
    rows = [{"races.date.year": "2009-01-01T00:00:00.000", "races.count": "17"},
            {"races.date.year": "2008-01-01T00:00:00.000", "races.count": "18"}]
    c = form(q, rows)
    assert c["options"]["chart"]["type"] == "line" and c["options"]["xAxis"]["type"] == "datetime"
    assert [p[1] for p in c["options"]["series"][0]["data"]] == [18, 17]  # sorted by time


def test_dimensions_only_is_a_table():
    rows = [{"races.name": "A", "circuits.country": "B"}] * 3
    assert form({"dimensions": ["races.name", "circuits.country"]}, rows)["form"] == "table"


def test_surface_messages():
    q = {"dimensions": ["circuits.country"], "measures": ["circuits.count"]}
    rows = [{"circuits.country": "USA", "circuits.count": "8"}, {"circuits.country": "UK", "circuits.count": "4"}]
    msgs = build_surface("s1", question="Circuits per country?", query=q, rows=rows, meta=META,
                         model_label="7B", stats={"steps": 2, "wall_s": 1.5},
                         trace=[{"role": "assistant", "tool": "submit_answer", "args": {"query": q}}])
    assert [m["version"] for m in msgs] == ["v0.9"] * 3
    assert msgs[0]["createSurface"] == {"surfaceId": "s1", "catalogId": CATALOG_ID}
    comps = {c["id"]: c for c in msgs[1]["updateComponents"]["components"]}
    assert "root" in comps and comps["chart"]["component"] == "HighchartsChart"
    for c in comps.values():  # every child reference resolves
        for ref in c.get("children", []) + [c.get("child")] + [t["child"] for t in c.get("tabs", [])]:
            assert ref is None or ref in comps, ref
    data = msgs[2]["updateDataModel"]
    assert data["path"] == "/" and data["value"]["chart"]["series"][0]["data"] == [8, 4]


def test_failed_answer_has_no_data_model():
    msgs = build_surface("s2", question="q", query=None, rows=None, meta=META, model_label="7B",
                         stats={}, error="No query submitted")
    assert len(msgs) == 2
    comps = {c["id"]: c for c in msgs[1]["updateComponents"]["components"]}
    assert "No query submitted" in comps["error"]["text"]


def test_measure_used_for_ranking_is_added_for_display():
    from app.presenter import display_query
    q = {"dimensions": ["circuits.country"], "order": {"circuits.count": "desc"}, "limit": 3}
    assert display_query(q, INFO)["measures"] == ["circuits.count"]
    assert display_query({"dimensions": ["circuits.country"]}, INFO) == {"dimensions": ["circuits.country"]}
    rows = [{"circuits.country": "USA", "circuits.count": "8"}, {"circuits.country": "UK", "circuits.count": "4"}]
    msgs = build_surface("s3", question="q", query=q, rows=[{"circuits.country": "USA"}], meta=META,
                         model_label="7B", stats={}, display=(display_query(q, INFO), rows))
    comps = {c["id"]: c for c in msgs[1]["updateComponents"]["components"]}
    assert comps["chart"]["component"] == "HighchartsChart" and "Count" in comps["added_note"]["text"]


def test_join_query_lists_cubes_and_sql():
    from app.presenter import cubes_in
    q = {"dimensions": ["races.year"], "measures": ["circuits.count"],
         "filters": [{"member": "races.name", "operator": "contains", "values": ["Grand"]}]}
    assert cubes_in(q) == ["circuits", "races"]
    msgs = build_surface("s4", question="q", query=q, rows=[], meta=META, model_label="7B", stats={},
                         sql="SELECT 1 FROM a LEFT JOIN b")
    comps = {c["id"]: c for c in msgs[1]["updateComponents"]["components"]}
    assert "circuits, races (Cube joins them)" in comps["query"]["text"] and "LEFT JOIN" in comps["query"]["text"]


def test_two_labels_with_many_values_become_one_bar_per_row():
    rows = [{"races.name": f"GP{i}", "circuits.country": f"C{i}", "races.count": str(i)} for i in range(12)]
    c = form({"dimensions": ["races.name", "circuits.country"], "measures": ["races.count"]}, rows)
    assert c["form"] == "chart" and c["options"]["xAxis"]["categories"][0] == "GP0 C0"


def test_date_listed_twice_is_charted_by_its_granularity():
    from app.presenter import display_query
    q = {"dimensions": ["races.date"], "measures": ["races.count"], "order": {"races.date": "asc"},
         "timeDimensions": [{"dimension": "races.date", "granularity": "month"}]}
    shown = display_query(q, INFO)
    assert "dimensions" not in shown and "order" not in shown and shown["measures"] == ["races.count"]
    rows = [{"races.date.month": "2009-01-01T00:00:00.000", "races.date": "2009-01-01T00:00:00.000", "races.count": "2"},
            {"races.date.month": "2009-02-01T00:00:00.000", "races.date": "2009-02-01T00:00:00.000", "races.count": "3"}]
    msgs = build_surface("s5", question="q", query=q, rows=rows, meta=META, model_label="7B", stats={},
                         display=(shown, rows))
    comps = {c["id"]: c for c in msgs[1]["updateComponents"]["components"]}
    assert msgs[2]["updateDataModel"]["value"]["chart"]["chart"]["type"] == "line"
    assert "per month" in comps["grain_note"]["text"]
    for c in comps.values():
        for ref in c.get("children", []) + [c.get("child")] + [t["child"] for t in c.get("tabs", [])]:
            assert ref is None or ref in comps, ref


def test_two_numbers_per_item_is_a_scatter_plot():
    meta = {"cubes": [{"name": "circuits", "dimensions": [
        {"name": "circuits.name", "type": "string", "shortTitle": "Name"},
        {"name": "circuits.lat", "type": "number", "shortTitle": "Lat"},
        {"name": "circuits.lng", "type": "number", "shortTitle": "Lng"}],
        "measures": [{"name": "circuits.count", "shortTitle": "Count"}, {"name": "circuits.avg_alt", "shortTitle": "Avg alt"}]}]}
    info = member_info(meta)
    q = {"dimensions": ["circuits.name", "circuits.lat", "circuits.lng"]}
    rows = [{"circuits.name": f"c{i}", "circuits.lat": str(i), "circuits.lng": str(-i)} for i in range(5)]
    c = choose_chart(q, rows, columns_for(q, rows, info))
    assert c["options"]["chart"]["type"] == "scatter"
    assert c["options"]["series"][0]["data"][1] == {"x": 1, "y": -1, "name": "c1"}
    q = {"dimensions": ["circuits.name"], "measures": ["circuits.count", "circuits.avg_alt"]}
    many = [{"circuits.name": f"c{i}", "circuits.count": "1", "circuits.avg_alt": str(i)} for i in range(30)]
    assert choose_chart(q, many, columns_for(q, many, info))["options"]["chart"]["type"] == "scatter"
    assert choose_chart(q, many[:5], columns_for(q, many[:5], info))["options"]["chart"]["type"] == "column"


def test_id_columns_are_not_scatter_axes():
    meta = {"cubes": [{"name": "r", "dimensions": [{"name": "r.driverId", "type": "number"},
                                                    {"name": "r.points", "type": "number"}], "measures": []}]}
    q = {"dimensions": ["r.driverId", "r.points"]}
    rows = [{"r.driverId": str(i), "r.points": "25"} for i in range(5)]
    assert choose_chart(q, rows, columns_for(q, rows, member_info(meta)))["form"] == "table"


def test_dates_grouped_without_a_measure_are_counted_for_the_chart():
    from app.presenter import display_query
    q = {"dimensions": ["races.date"], "filters": [{"member": "races.year", "operator": "equals", "values": ["2009"]}],
         "timeDimensions": [{"dimension": "races.date", "granularity": "month"}]}
    shown = display_query(q, INFO)
    assert shown["measures"] == ["races.count"] and "dimensions" not in shown and shown["filters"] == q["filters"]
    rows = [{"races.date.month": f"2009-0{m}-01T00:00:00.000", "races.count": "2"} for m in (3, 4, 5)]
    msgs = build_surface("s6", question="q", query=q, rows=rows, meta=META, model_label="7B", stats={},
                         display=(shown, rows))
    comps = {c["id"]: c for c in msgs[1]["updateComponents"]["components"]}
    assert msgs[2]["updateDataModel"]["value"]["chart"]["chart"]["type"] == "line"
    assert "does not count" in comps["added_note"]["text"]
    for c in comps.values():
        for ref in c.get("children", []) + [c.get("child")] + [t["child"] for t in c.get("tabs", [])]:
            assert ref is None or ref in comps, ref
