"""Demo app: ask a question in plain English, the harness agent (ADK) writes a Cube query, and
the answer comes back as an A2UI surface with a Highcharts chart.

    python -m app.server                      # http://localhost:8000 (serves app/web/dist)
    python -m app.server --port 8000 --cube-url http://localhost:4000/cubejs-api/v1

The agent is harness v4.2 unchanged (harness.agent.Harness). Only --model differs between the
four experiment arms, and the app lets you switch between them or run all four on one question.
"""
from __future__ import annotations

import argparse
import json
import os
import threading
import time
import uuid
from pathlib import Path

import requests
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from harness.agent import Harness, make_model
from harness.cube_client import CubeClient

from .presenter import build_surface, display_query, member_info

ROOT = Path(__file__).resolve().parent.parent
DIST = Path(__file__).resolve().parent / "web" / "dist"

# The four experiment arms (README "LoRA arm"); test-set strict accuracy from the frozen runs.
MODELS = [
    {"id": "ollama_chat/qwen2.5-coder:7b", "label": "Qwen 7B · harness", "short": "7B", "lora": False,
     "strict": 27.7},
    {"id": "ollama_chat/qwen7b-cube-lora", "label": "Qwen 7B + LoRA · harness", "short": "7B+LoRA",
     "lora": True, "strict": 46.4},
    {"id": "ollama_chat/qwen2.5-coder:14b", "label": "Qwen 14B · harness", "short": "14B", "lora": False,
     "strict": 28.9},
    {"id": "ollama_chat/qwen14b-cube-lora", "label": "Qwen 14B + LoRA · harness", "short": "14B+LoRA",
     "lora": True, "strict": 49.7},
]
DATABASES = ["formula_1", "student_club"]

# Questions that make good charts. BIRD test questions are added from data/bird/dev.json if present.
# (question, joins tables?, chart type it is meant to show or None)
CHART_EXAMPLES = {
    "formula_1": [
        ("How many races were held in each year?", False, "Columns"),
        ("How many drivers are there of each nationality? Show the top 15.", False, "Horizontal bars"),
        ("How many races were held in each month of 2009?", False, "Line"),
        ("Show the latitude and longitude of every circuit, with its name.", False, "Scatter"),
        ("How many drivers are there?", False, "Single number"),
        ("List the names and nationalities of all constructors.", False, "Table"),
        ("Which 10 drivers have scored the most points in their careers?", True, None),      # drivers + results
        ("Which constructors have won the most races? Show the top 10.", True, None),        # constructors + results
        ("Which countries have hosted the most races?", True, None),                         # circuits + races
        ("How many points did Lewis Hamilton score in each season?", True, None),            # drivers + results + races
        ("What are the 10 most common finishing statuses in race results?", True, None),     # status + results
    ],
    "student_club": [
        ("How much was spent in each budget category?", False, "Columns"),
        ("How many events of each type are there, split by status?", False, "Stacked bars"),
        ("What is the budgeted amount and the amount spent for each budget category?", False, "Two series"),
        ("How much income was received each month?", False, "Line"),
        ("Compare each event's budgeted amount with the amount it spent.", True, "Scatter"),
        ("How many members are there?", False, "Single number"),
        ("List the members' names and emails.", False, "Table"),
        ("How many events of each type are there?", False, None),
        ("How much did each event spend? Show the top 10.", True, None),                     # event + budget
        ("How many event attendances came from each college?", True, None),                  # attendance + member + major
        ("Which 10 members have spent the most on expenses?", True, None),                   # member + expense
        ("How many attendances did each type of event have?", True, None),                   # attendance + event
        ("How much was spent on expenses in each budget category?", True, None),             # expense + budget
    ],
}


class AskBody(BaseModel):
    question: str
    db: str
    model: str
    hint: str | None = None


def create_app(cube_url: str, ollama_url: str, num_ctx: int, max_steps: int,
               bird_dir: Path) -> FastAPI:
    app = FastAPI(title="Text-to-Cube demo")
    cube = CubeClient(cube_url)
    harnesses: dict[str, Harness] = {}
    gpu_lock = threading.Lock()  # one question at a time: the four models share one GPU

    def harness_for(model_id: str) -> Harness:
        if model_id not in harnesses:
            harnesses[model_id] = Harness(make_model(model_id, ollama_url, num_ctx=num_ctx), cube,
                                          max_steps=max_steps)
        return harnesses[model_id]

    def bird_examples() -> dict[str, list[dict]]:
        dev, ids = bird_dir / "dev.json", ROOT / "splits" / "test_ids.json"
        if not dev.exists():
            return {}
        test = set(json.loads(ids.read_text())) if ids.exists() else None
        out: dict[str, list[dict]] = {}
        for q in json.loads(dev.read_text(encoding="utf-8")):
            if q["db_id"] in DATABASES and (test is None or q["question_id"] in test):
                lst = out.setdefault(q["db_id"], [])
                if len(lst) < 8 and q["difficulty"] != "challenging":
                    lst.append({"question": q["question"], "hint": q.get("evidence") or None})
        return out

    @app.get("/api/config")
    def config():
        available = None
        try:
            tags = requests.get(f"{ollama_url.rstrip('/')}/api/tags", timeout=3).json()
            names = {m["name"] for m in tags.get("models", [])}
            names |= {n.removesuffix(":latest") for n in names}
            available = {m["id"]: m["id"].split("/", 1)[1] in names for m in MODELS}
        except Exception:
            pass
        try:
            cube.meta()
            cube_ok = True
        except Exception:
            cube_ok = False
        bird = bird_examples()
        return {
            "models": [dict(m, available=None if available is None else available[m["id"]]) for m in MODELS],
            "databases": DATABASES,
            "examples": {db: [{"question": q, "hint": None, "join": j, "chart": c}
                              for q, j, c in CHART_EXAMPLES[db]]
                         for db in DATABASES},
            "birdExamples": bird,
            "cube": cube_ok,
            "ollama": available is not None,
        }

    @app.post("/api/ask")
    def ask(body: AskBody):
        if body.db not in DATABASES:
            raise HTTPException(400, f"Unknown database {body.db}")
        model = next((m for m in MODELS if m["id"] == body.model), None)
        if model is None:
            raise HTTPException(400, f"Unknown model {body.model}")
        surface_id = f"answer-{uuid.uuid4().hex[:10]}"
        with gpu_lock:
            t0 = time.time()
            try:
                res = harness_for(model["id"]).run(body.db, body.question.strip(), body.hint or None)
            except Exception as e:  # endpoint down, model missing
                res = None
                err = f"{type(e).__name__}: {e}"
            wall = time.time() - t0
        rows, error, display, sql = None, None, None, None
        if res is None:
            error = err
        elif res.query is None or res.error:
            error = res.error or "The model did not submit a query."
        else:
            try:
                rows = cube.load(res.query)
                shown = display_query(res.query, member_info(cube.meta()))
                if shown is not res.query:
                    display = (shown, cube.load(shown))
                try:
                    sql = cube.sql(res.query)
                except Exception:
                    sql = None
            except Exception as e:
                error = f"Cube error: {e}"
        stats = {"steps": res.steps if res else None, "wall_s": wall,
                 "prompt_tokens": res.prompt_tokens if res else 0,
                 "completion_tokens": res.completion_tokens if res else 0}
        messages = build_surface(surface_id, question=body.question, query=res.query if res else None,
                                 rows=rows, meta=cube.meta(), model_label=model["label"], stats=stats,
                                 error=error, trace=res.trace if res else None, display=display,
                                 sql=sql)
        return {"surfaceId": surface_id, "model": model["id"], "messages": messages,
                "query": res.query if res else None, "error": error, "stats": stats,
                "rowCount": None if rows is None else len(rows)}

    if DIST.exists():
        app.mount("/assets", StaticFiles(directory=DIST / "assets"), name="assets")

        @app.get("/")
        def index():
            return FileResponse(DIST / "index.html")

    return app


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--cube-url", default=os.environ.get("CUBE_URL", "http://localhost:4000/cubejs-api/v1"))
    ap.add_argument("--ollama-url", default=os.environ.get("OLLAMA_HOST", "http://localhost:11434"))
    ap.add_argument("--num-ctx", type=int, default=16384, help="same context window as the experiments")
    ap.add_argument("--max-steps", type=int, default=8)
    ap.add_argument("--bird-dir", type=Path, default=ROOT / "data" / "bird")
    args = ap.parse_args()
    if not args.ollama_url.startswith("http"):
        args.ollama_url = "http://" + args.ollama_url
    import uvicorn
    if not DIST.exists():
        print("app/web/dist not found: run `npm install && npm run build` in app/web, "
              "or use `npm run dev` there (http://localhost:5173).")
    uvicorn.run(create_app(args.cube_url, args.ollama_url, args.num_ctx, args.max_steps, args.bird_dir),
                host=args.host, port=args.port)


if __name__ == "__main__":
    main()
