# Text-to-Cube on BIRD: pure harness vs LoRA + harness

MSc Generative AI module project. A local LLM answers questions from the
[BIRD](https://bird-bench.github.io/) benchmark by writing **Cube semantic-layer
queries** instead of SQL. We compare two setups on identical prompts and tools:

1. **Pure harness**: base model + prompt + Cube tools (harness v4.2, frozen)
2. **LoRA + harness**: the same model with a LoRA adapter, the same harness

Both were run with Qwen2.5-Coder 7B and 14B. On the 332 test questions, strict / relaxed accuracy:
7B 27.7% / 34.0% → 46.4% / 52.1% with LoRA; 14B 28.9% / 41.3% → 49.7% / 56.9% with LoRA.
The write-up is in `report/` and the notebook that reproduces its numbers is `notebook/CE893_evaluation.ipynb`.

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
| `app/` | the demo app: Python server (`app/server.py`) and web page (`app/web`, A2UI + Highcharts) |
| `scripts/add_value_searches.py`, `scripts/synth_schema_qa.py` | add search_values steps to training examples; synthetic questions on the test schemas (noschema arm) |
| `scripts/train_lora.py`, `scripts/export_ollama.py` | QLoRA training; merge + GGUF Q4_K_M + Ollama model |
| `training_data/` | the 3,531 verified training pairs and their schemas |
| `results/` | stored test and tuning runs (`runs/*.jsonl.gz`) and LoRA training logs (`training/`) |
| `report/`, `notebook/` | final report (LaTeX, full draft + outline) and the notebook that reproduces its numbers and figures |
| `proposal/` | the coursework proposal |
| `tests/` | end-to-end tests with a scripted stand-in for the LLM |

## How the harness works

The agent loop is built on Google's [Agent Development Kit (ADK)](https://google.github.io/adk-docs/).
ADK calls the model through LiteLLM, so any local server works: Ollama for the base model, and
Ollama with the adapter merged in for the LoRA model (`scripts/export_ollama.py`). Both setups use exactly the same agent.

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
length (Qwen tokenizer, with the search_values steps): 2,621 training examples fit in 4,096 tokens and 2,929 in 8,192
(large schemas make the rest long).

## Harness versions

Tune on the three tuning databases, report on `splits/test_ids.json`. Early comparisons (v2 to v4.1) used the
120-question sample `splits/tune_ids.json`; it proved too noisy, so v3 and v4.2 were also compared on all 363 dev
questions of those databases (drop `--ids-file`). The tuning databases need their own DuckDB copy and Cube models:

```bash
python scripts/build_duckdb.py --bird-dir data/bird --out data/bird.duckdb --db formula_1 --db student_club --db superhero --db toxicology --db california_schools
python scripts/gen_cube_models.py --bird-dir data/bird --out cube/model --db formula_1 --db student_club --db superhero --db toxicology --db california_schools
python -m harness.evaluate --db superhero --db toxicology --db california_schools --ids-file splits/tune_ids.json \
    --model ollama_chat/qwen2.5-coder:7b --setup harness-v3 --out runs/tune_v3.jsonl
```

| Version | Commit | Change | Result, strict / relaxed (7B unless noted) |
|---|---|---|---|
| v1 | 0c95b22 | first harness | 13.3% / 26.8% |
| v2 | bb618ff | measure/dimension and filter checks, text tool calls unwrapped, rules + 4 examples in the prompt | 20.5% / 34.3% |
| v3 | 53c8093 | value lookup on number/time columns, repeated-call guard, one-time column check at submit | tune-120 23.3% (v2 22.5%); tune-363 26.2% / 33.6% |
| v4 | e9d1d12 | schema linking: the question message lists fields the question/hint name (`--no-link` turns it off); column check also on text answers; text values on number filters caught | tune 22.5% (california_schools 0/40, toxicology 15/40) |
| v4.1 | 309104f | linked fields weighted by word rarity, at most 5, labelled "not a list of columns to return"; formula-like members get a clear error | tune 24.2% / 30.8% |
| **v4.2** (frozen) | 33da679 | an unchanged resubmit after the column warning is trimmed by the harness; text answers get the same check; timeDimensions must be time fields | tune-363 32.0% / 35.3% (vs v3 McNemar p = 0.011), 14B 38.8% / 47.9%; **test 27.7% / 34.0%, 14B 28.9% / 41.3%** |

## LoRA arm

`training_data/` holds the 3,531 verified BIRD-train question → Cube query pairs and their schemas.
Rebuild the chat-format files after any prompt change (`data/sft/` is not in git):

```bash
python scripts/make_sft.py --data training_data --out data/sft      # training_data already has search_values steps (scripts/add_value_searches.py, needs BIRD train)
pip install torch --index-url https://download.pytorch.org/whl/cu128 && pip install -r requirements-train.txt
python scripts/train_lora.py --base Qwen/Qwen2.5-Coder-7B-Instruct --out lora/qwen7b --dry-run   # check masking
python scripts/train_lora.py --base Qwen/Qwen2.5-Coder-7B-Instruct --out lora/qwen7b
python scripts/export_ollama.py --base Qwen/Qwen2.5-Coder-7B-Instruct --adapter lora/qwen7b/final \
    --llama-cpp ../llama.cpp --quantize ../llama.cpp/build/bin/llama-quantize --ollama-base qwen2.5-coder:7b --name qwen7b-cube-lora
python -m harness.evaluate --db formula_1 --db student_club --ids-file splits/test_ids.json \
    --model ollama_chat/qwen7b-cube-lora --setup lora-v4.2-7b --out runs/test_lora_7b.jsonl
```

The same with `Qwen2.5-Coder-14B-Instruct` / `qwen2.5-coder:14b` for the 14B arm. Both arms use harness v4.2,
Q4_K_M weights in Ollama, and the same chat template (checked by `export_ollama.py`), so each LoRA model differs from its
baseline only by the adapter. The 7B was trained on examples up to 8,192 tokens; the 14B only fits 16 GB at
`--max-len 4096` (2,621 train / 139 val examples), so it saw about 10% fewer examples. `--quantize` is optional:
without it Ollama would quantise, but Ollama 0.40 cannot make Q4_K_M, so we used llama.cpp's `llama-quantize` (release b12376).

### search_values v2 (7B retrain, 2026-11-21: no gain)

The first adapters almost stopped calling `search_values` (7B: 1 of 332 test questions, 14B: 0; harness-only 92%),
because only 769 of the 3,531 training examples (22%) had a lookup. On 7B, wrong answers whose final query returned
0 rows went from 10 to 25. `scripts/add_value_searches_v2.py` rebuilds the lookups: one for every string `equals`
filter on a text column (68% of examples, up from 22%) (up to 2 per example), searched with a short lower-case word from the question/value rather
than the exact stored value, plus 312 recovery examples (first `run_query` with a plausible wrong spelling,
0 rows, `search_values`, corrected query). The wrong first query is marked `"train": false`, so `train_lora.py`
keeps it as context but does not train the model to write it.

```bash
# training_data_sv2/ is in git (2,386 of 3,531 examples with a lookup, 312 recovery examples); rebuild with:
# python scripts/add_value_searches_v2.py --data training_data --out training_data_sv2 --bird-train data/bird_train/train_databases --recovery-share 0.18
python scripts/make_sft.py --data training_data_sv2 --out data/sft_sv2
python scripts/train_lora.py --base Qwen/Qwen2.5-Coder-7B-Instruct --data data/sft_sv2 --out lora/qwen7b-sv2
```

Result (7B, same settings as the first 7B LoRA: 2,929 train examples at max-len 8,192, 2 epochs; Ollama
`qwen7b-cube-sv2`; test set, 332 questions):

| 7B LoRA | Hint | Strict | Relaxed | Calls search_values | Wrong with 0-row last query |
|---|---|---|---|---|---|
| original | no | 36.7% | 40.7% | 0.9% | 43 |
| **sv2** | no | **36.1%** (McNemar p = 0.88) | 37.7% | 38.0% | 21 |
| original | yes | 46.4% | 52.1% | 0.3% | 25 |
| sv2 | yes | 43.4% (p = 0.12) | 45.2% | 57.5% | 21 |

The model learned the habit (searches on 38-58% of questions, empty-result failures halved without the hint), but
accuracy did not improve: the extra steps brought new failures (9 malformed tool calls, e.g. search_values
arguments sent to run_query, 4 step-limit hits, 3 unanswered) and cost answer shape (relaxed falls). With the hint
most literal values are already given, so the lookups add little. Validation loss was flat from epoch 0.5
(0.293). The 14B was not retrained (it had no empty-result failures to fix). Runs:
`results/runs/test_nohint_lora_sv2_7b.jsonl.gz`, `results/runs/test_lora_sv2_7b.jsonl.gz`.

## Schema-offloading arms (prompt profiles)

Can the LoRA carry what the prompt spells out, so each question costs fewer tokens?
`--profile` (in `harness.evaluate` and `scripts/make_sft.py`) picks the prompt:

| profile | query guide | tool JSON schemas | data schema | linked fields |
|---|---|---|---|---|
| `full` (harness v4.2) | yes | yes | full `describe()` text | yes |
| `lean` | no | no (one line with tool names) | compact: no measure lines, no cube prefix | yes |
| `noschema` | no | no | none, only the database name | no |

`lean` and `noschema` send no tool declarations to the model (`LiteLlmNoToolSchemas`); ADK still runs
the tools and parses Qwen's `<tool_call>` text back into calls.

The test databases (formula_1, student_club) are not in the LoRA training data, so a model can only
work without their schema if it is trained on it. `scripts/synth_schema_qa.py` writes template
questions from the two schemas and sampled values (no BIRD dev question, hint or SQL is used) and
checks each query against Cube. The `noschema` training set = the 58 training databases (55 train + 3 validation) with the
`lean` prompt + these synthetic examples with the `noschema` prompt. It is a single-database setting
(a model fine-tuned for the databases it serves), unlike `lean`, which stays database-agnostic.

```
python scripts/synth_schema_qa.py --db formula_1 --n 800 --db student_club --n 500 --out data/synth/synth.jsonl --audit
python scripts/make_sft.py --data training_data --profile lean --out data/sft_lean
python scripts/make_sft.py --data training_data --profile noschema --synthetic data/synth/synth.jsonl --out data/sft_noschema
python scripts/train_lora.py --data data/sft_lean --out lora/qwen7b-lean
python -m harness.evaluate ... --profile lean --model ollama_chat/qwen7b-cube-lean
```

### Results (test set, 332 questions, 2026-11-17/19)

Prompt tokens are summed over all model calls of a question. Strict / relaxed execution accuracy.

| Model | Arm | Strict | Relaxed | Prompt tokens / q | Steps |
|---|---|---|---|---|---|
| 7B | harness only (full prompt) | 27.7 | 34.0 | 15,742 | 4.0 |
| 7B | LoRA, full prompt | 46.4 | 52.1 | 7,798 | 2.1 |
| 7B | full-prompt LoRA given lean prompt (no retrain) | 0.0 | 0.0 | 5,409 | 3.5 |
| 7B | LoRA trained + run on lean prompt | 41.9 | 45.5 | 3,257 | 2.2 |
| 7B | noschema LoRA, no schema in prompt | 31.0 | 33.7 | 1,312 | 3.8 |
| 7B | noschema LoRA, lean prompt | 44.3 | 49.7 | 3,364 | 2.2 |
| 14B | harness only (full prompt) | 28.9 | 41.3 | 10,766 | 2.9 |
| 14B | LoRA, full prompt | 49.7 | 56.9 | 7,341 | 2.0 |
| 14B | full-prompt LoRA given lean prompt (127 q, stopped) | 0.0 | 0.0 | 10,169 | 3.2 |
| 14B | LoRA trained + run on lean prompt | 47.6 | 57.2 | 3,153 | 2.1 |
| 14B | noschema LoRA, no schema in prompt | 38.3 | 42.8 | 1,024 | 2.6 |
| 14B | noschema LoRA, lean prompt | 47.9 | 56.0 | 3,086 | 2.0 |

McNemar exact (strict): 7B lean vs full p=0.008; 14B lean vs full p=0.21; 14B noschema+lean vs full p=0.35;
14B noschema (no schema) vs full p=4e-6; 7B vs 14B noschema p=0.004. Without the schema the model invents
members (an "Unknown members" error in 120 of 332 14B runs), mostly on student_club (14B 44.9% vs 63.9% lean).

## Demo app

Ask a question in plain English, pick a model (or compare the four main arms, or the four
short-prompt LoRAs), and see the answer as a chart. The menu has the four main arms plus the
schema-offloading LoRAs that score best on the test set (lean, and noschema run with the lean prompt);
these run with the prompt profile they were trained on, as `evaluate.py --profile lean` does. The flow:

1. `app/server.py` runs harness v4.2 unchanged (the ADK agent with Cube tools) on the chosen model.
2. The submitted Cube query is run on Cube, and `app/presenter.py` turns query + rows into
   [A2UI](https://github.com/a2ui-project/a2ui) v0.9 messages: one surface per answer, with tabs for
   the chart, the table, the Cube query and the agent's tool calls.
3. The page (`app/web`, Lit) feeds those messages to A2UI's renderer. The surface uses the A2UI basic
   catalog plus two components of ours: `HighchartsChart` (a Highcharts options object bound to the
   surface's data model) and `DataTable`.

The chart form follows the query's shape: a single number becomes a stat tile, a time dimension a line
chart, one category a bar chart, two categories stacked bars, anything else a table. When the answer
query sorts by a measure it does not return ("which 10 countries have the most circuits?"), the chart
adds that measure so the ranking is visible, and says so.

```bash
# Cube on :4000 and Ollama with the experiment models (missing ones show as not installed)
cd app/web && npm install && npm run build && cd ../..
python -m app.server                    # http://localhost:8000
```

For UI work, `npm run dev` in `app/web` serves the page on :5173 with hot reload and forwards `/api`
to the server on :8000. Options: `--cube-url`, `--ollama-url`, `--num-ctx` (default 16384, as in the
experiments), `--port`. If `data/bird/dev.json` is there, the page also offers questions from the test set.
Questions run one at a time, since the models share one GPU; switching model makes Ollama load
the other one, so the first question after a switch takes longer.
