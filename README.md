# Text-to-Cube on BIRD: pure harness vs LoRA + harness

MSc Generative AI module project. A local LLM answers questions from the
[BIRD](https://bird-bench.github.io/) benchmark by writing **Cube semantic-layer
queries** instead of SQL. We compare two setups on identical prompts and tools:

1. **Pure harness**: base model + prompt + Cube tools (this part is built)
2. **LoRA + harness**: the same model with a LoRA adapter (next)

Accuracy is *execution accuracy*: the Cube query's result must equal the result
of BIRD's gold SQL on the original SQLite database (set comparison, column order ignored).

## Layout

| Path | What it is |
|---|---|
| `scripts/download_bird.py` | downloads the BIRD dev set into `data/bird` |
| `scripts/build_duckdb.py` | copies the chosen BIRD SQLite databases into `data/bird.duckdb` (one schema per database) for Cube |
| `scripts/gen_cube_models.py` | generates Cube models from BIRD's schema and column descriptions (`cube/model/*.yml`) |
| `cube/` | Cube server (DuckDB driver) |
| `harness/` | the harness: prompts, Cube client, tool loop, evaluator |
| `tests/` | end-to-end test with a scripted stand-in for the LLM |

## How the harness works

The system prompt holds a short guide to Cube's query JSON and a compact listing of every
cube, dimension (with BIRD's column descriptions) and measure for the question's database.
The model then uses three tools:

- `search_values(dimension, text)` finds exact stored values (names, categories)
- `run_query(query)` runs a draft query and shows the first rows or Cube's error
- `submit_answer(query)` submits the final query

Before any query reaches Cube, the harness checks every member name and replies with
"did you mean ..." hints, so the model can correct itself. There is also a `--mode single`
baseline (one shot, no tools), which is useful as an ablation.

## Setup (Windows, macOS or Linux)

Needs Python 3.11+, Node.js 20+, and [Ollama](https://ollama.com) (or any OpenAI-compatible server).

```bash
python -m venv .venv && .venv\Scripts\activate      # Windows (source .venv/bin/activate elsewhere)
pip install -r requirements.txt

python scripts/download_bird.py
python scripts/build_duckdb.py --bird-dir data/bird --out data/bird.duckdb --db formula_1 --db student_club
python scripts/gen_cube_models.py --bird-dir data/bird --duckdb data/bird.duckdb --out cube/model --db formula_1 --db student_club

cd cube && copy .env.example .env && npm install && npm run dev      # cp on macOS/Linux; Cube on :4000
```

In another terminal:

```bash
ollama pull qwen2.5-coder:7b
python -m pytest tests -q                       # sanity check (needs Cube running)
python -m harness.evaluate --db student_club --db formula_1 --model qwen2.5-coder:7b \
    --setup harness --out runs/harness_qwen7b.jsonl
```

Each line of the output file records the question, the predicted Cube query, whether it was
correct, the number of steps, tokens and latency, and the full tool trace. Runs resume if interrupted.
Use `--limit 10` for a quick trial and `--mode single` for the no-tools baseline.
