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
| `harness/` | the harness: an ADK agent with Cube tools, prompts, evaluator |
| `harness/sql2cube.py` | translates simple gold SQL into Cube queries (used to build training data) |
| `scripts/build_training_data.py` | converts BIRD train questions to Cube queries and keeps only pairs whose results match the SQL |
| `scripts/make_sft.py` | writes chat-format fine-tuning files that reproduce the harness's exact prompts |
| `splits/test_ids.json` | the frozen test set: all 332 dev questions on formula_1 and student_club |
| `splits/tune_ids.json` | tuning set for harness changes: 40 questions each from superhero, toxicology and california_schools (no overlap with the test databases) |
| `tests/` | end-to-end tests with a scripted stand-in for the LLM |

## How the harness works

The agent loop is built on Google's [Agent Development Kit (ADK)](https://google.github.io/adk-docs/).
ADK calls the model through LiteLLM, so any local server works: Ollama for the base model, and
Ollama or vLLM with the adapter for the LoRA model. Both setups use exactly the same agent.

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
python scripts/gen_cube_models.py --bird-dir data/bird --out cube/model --db formula_1 --db student_club

cd cube && copy .env.example .env && npm install && npm run dev      # cp on macOS/Linux; Cube on :4000
```

In another terminal:

```bash
ollama pull qwen2.5-coder:7b
python -m pytest tests -q                       # sanity check (needs Cube running)
python -m harness.evaluate --db student_club --db formula_1 --model ollama_chat/qwen2.5-coder:7b \
    --setup harness --out runs/harness_qwen7b.jsonl
```

Each line of the output file records the question, the predicted Cube query, whether it was
correct, the number of steps, tokens and latency, and the full tool trace. Runs resume if interrupted.
`--model` is a LiteLLM id (`ollama_chat/...` for Ollama, `openai/<name>` with `--base-url http://localhost:8000/v1` for vLLM).
Use `--limit 10` for a quick trial and `--mode single` for the no-tools baseline.

## Scoring

`correct` is strict: same rows, same number of columns. `correct_relaxed` also accepts extra
columns, because Cube must return any dimension it sorts by while SQL can sort by a column it
does not return (e.g. "the driver with the earliest birth date"). Report both.

As a ceiling, the rule-based translator reproduces 231 of the 332 test answers (70%) with relaxed
scoring; the other 101 need SQL a semantic layer cannot express (subqueries, CASE arithmetic,
string slicing) or differ in edge cases such as ties.

## LoRA training data

BIRD's train split (about 9,400 questions on 69 databases, none shared with the test set):

```bash
# download train.zip from https://bird-bench.oss-cn-beijing.aliyuncs.com/train.zip and unpack to data/bird_train
python scripts/build_training_data.py --bird-dir data/bird_train --out data/training   # stop the main Cube first
python scripts/make_sft.py --data data/training --out data/sft
```

Result of the first run (59 of 69 train databases; the 10 largest, over 600 MB each, were skipped):

| | Questions |
|---|---|
| BIRD train questions on the 59 databases | 8,003 |
| translated **and** verified by execution | **3,531** |
| not expressible by the rule-based translator | 2,782 |
| translated but result differed | 682 |
| Cube error (mostly many-to-many joins without a declared key) | 900 |
| empty gold result (verifies nothing, dropped) | 108 |

`make_sft.py` holds out 3 whole databases for validation (3,273 train / 258 val examples). Prompt
length: 2,695 training examples fit in 4,096 tokens and 2,929 in 8,192 (large schemas make the rest long).

## Harness versions

Tune on `splits/tune_ids.json`, report on `splits/test_ids.json`. The tuning databases need their own
DuckDB copy and Cube models:

```bash
python scripts/build_duckdb.py --bird-dir data/bird --out data/bird.duckdb --db formula_1 --db student_club --db superhero --db toxicology --db california_schools
python scripts/gen_cube_models.py --bird-dir data/bird --out cube/model --db formula_1 --db student_club --db superhero --db toxicology --db california_schools
python -m harness.evaluate --db superhero --db toxicology --db california_schools --ids-file splits/tune_ids.json \
    --model ollama_chat/qwen2.5-coder:7b --setup harness-v3 --out runs/tune_v3.jsonl
```

| Version | Commit | Change | Test strict / relaxed |
|---|---|---|---|
| v1 | 0c95b22 | first harness | 13.3% / 26.8% |
| v2 | bb618ff | measure/dimension and filter checks, text tool calls unwrapped, rules + 4 examples in the prompt | 20.5% / 34.3% |
| v3 | 53c8093 | value lookup on number/time columns, repeated-call guard, one-time column check at submit | tune 23.3% (v2 22.5%) |
| v4 | e9d1d12 | schema linking: the question message lists fields the question/hint name (`--no-link` turns it off); column check also on text answers; text values on number filters caught | tune 22.5% (california_schools 0/40, toxicology 15/40) |
| v4.1 | 309104f | linked fields weighted by word rarity, at most 5, labelled "not a list of columns to return"; formula-like members get a clear error | tune 24.2% / 30.8% |
| v4.2 | (this) | an unchanged resubmit after the column warning is trimmed by the harness; text answers get the same check; timeDimensions must be time fields | not yet run |

## LoRA arm

`training_data/` holds the 3,531 verified BIRD-train question → Cube query pairs and their schemas.
Rebuild the chat-format files after any prompt change (`data/sft/` is not in git):

```bash
python scripts/make_sft.py --data training_data --out data/sft
pip install torch --index-url https://download.pytorch.org/whl/cu128 && pip install -r requirements-train.txt
python scripts/train_lora.py --base Qwen/Qwen2.5-Coder-7B-Instruct --out lora/qwen7b --dry-run   # check masking
python scripts/train_lora.py --base Qwen/Qwen2.5-Coder-7B-Instruct --out lora/qwen7b
python scripts/export_ollama.py --base Qwen/Qwen2.5-Coder-7B-Instruct --adapter lora/qwen7b/final \
    --llama-cpp ../llama.cpp --quantize ../llama.cpp/build/bin/llama-quantize --ollama-base qwen2.5-coder:7b --name qwen7b-cube-lora
python -m harness.evaluate --db formula_1 --db student_club --ids-file splits/test_ids.json \
    --model ollama_chat/qwen7b-cube-lora --setup lora-v4.2-7b --out runs/test_lora_7b.jsonl
```

The same with `Qwen2.5-Coder-14B-Instruct` / `qwen2.5-coder:14b` for the 14B arm. Both arms use harness v4.2,
examples up to 4,096 tokens, and Q4_K_M weights in Ollama, so each LoRA model differs from its baseline only by the adapter.
