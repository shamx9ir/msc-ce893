"""search_values v2 training data: lookups for text filters, and 0-rows -> search -> retry examples."""
import json
import sqlite3
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from add_value_searches_v2 import fragments, wrong_spelling  # noqa: E402


def test_fragments_prefer_question_wording():
    assert fragments("Circuit de Barcelona-Catalunya", "races at the barcelona track")[0] == "barcelona"
    assert fragments("ARECIBO", '"ARECIBO" is the county') == ["arecibo"]
    assert fragments("Italy", "drivers from italy")[-1] == "italy"


def test_wrong_spelling_is_not_stored():
    stored = {"ARECIBO"}.__contains__
    assert wrong_spelling("ARECIBO", "households in Arecibo county", stored) == "Arecibo"
    assert wrong_spelling("ARECIBO", "nothing here", stored) == "Arecibo"
    assert wrong_spelling("ab", "x", lambda s: True) is None


def test_end_to_end(tmp_path):
    db = tmp_path / "bird" / "shop"
    db.mkdir(parents=True)
    con = sqlite3.connect(db / "shop.sqlite")
    con.execute("CREATE TABLE city (name TEXT, pop INT)")
    con.executemany("INSERT INTO city VALUES (?, ?)", [("ARECIBO", 1), ("ARECIBO WEST", 2), ("Ponce", 3)])
    con.commit()
    data = tmp_path / "data"
    (data / "schemas").mkdir(parents=True)
    (data / "schemas" / "shop.txt").write_text("cube city: name (string), pop (number)", encoding="utf-8")
    rec = {"db_id": "shop", "question_id": 1, "question": "Population of Arecibo?", "evidence": "",
           "cube_query": {"dimensions": ["city.pop"],
                          "filters": [{"member": "city.name", "operator": "equals", "values": ["ARECIBO"]}]},
           "row_count": 1, "preview": [{"city.pop": 1}]}
    (data / "verified.jsonl").write_text(json.dumps(rec) + "\n", encoding="utf-8")
    out = tmp_path / "out"
    subprocess.run([sys.executable, str(ROOT / "scripts/add_value_searches_v2.py"), "--data", str(data),
                    "--out", str(out), "--bird-train", str(tmp_path / "bird"), "--recovery-share", "1"], check=True)
    r = json.loads((out / "verified.jsonl").read_text(encoding="utf-8"))
    assert r["searches"] == [{"dimension": "city.name", "text": "arecibo", "result": ["ARECIBO", "ARECIBO WEST"]}]
    assert r["recovery"] == {"member": "city.name", "wrong": "Arecibo", "right": "ARECIBO"}

    sys.path.insert(0, str(ROOT))
    from make_sft import example
    msgs = example(r, "schema", "tools")["messages"]
    calls = [(m["tool_calls"][0]["function"]["name"], m.get("train", True)) for m in msgs if m.get("tool_calls")]
    assert calls == [("run_query", False), ("search_values", True), ("run_query", True), ("submit_answer", True)]
    first = json.loads(msgs[2]["tool_calls"][0]["function"]["arguments"])["query"]
    assert first["filters"][0]["values"] == ["Arecibo"] and msgs[3]["content"] == '{"row_count": 0, "rows": []}'
    assert rec["cube_query"]["filters"][0]["values"] == ["ARECIBO"]  # the right query is untouched
