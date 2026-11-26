# CE893 Generative AI project, {REG}

**Text-to-Cube on BIRD: prompt-engineering harness vs LoRA fine-tuning.** A small open-weight LLM (Qwen2.5-Coder
7B and 14B) answers questions about a database by writing Cube semantic-layer queries through a Google ADK agent.
The project compares the same agent with prompt engineering alone ("harness only") against the same agent with a
LoRA adapter, and tests how much of the prompt the adapter can take over.

Snapshot of GitHub repository `shamx9ir/msc-ce893`, commit `{COMMIT}`, packed on {DATE}.

## What is where

| Path | Contents |
|---|---|
| `report.pdf` | the final report, {REPORT} |
| `report-source/` | the LaTeX source of `report.pdf`: `main.tex`, the text file it inputs, `preamble.tex`, `references.bib`, `ce893.bst`, the template font and logo, `figures/`. Build with `latexmk -xelatex main.tex` (see `BUILD.txt`) |
| `code/` | all code, in the repository's own layout so every relative path works (see the table below) |
| `data/original/bird/` | the BIRD dev files of the five databases used: `dev.json` (their questions, hints and gold SQL), `dev_tables.json`, and `dev_databases/<db>/` (SQLite file + BIRD's column descriptions) |
| `data/processed/` | `test_questions.jsonl` (332 frozen test questions with gold SQL), `tune_questions.jsonl` (all 363 questions of the three harness-development databases; `splits/tune_ids.json` marks the 120-question subset used early on), `splits/`, and `lora_training/<adapter>/sft_tools_{train,val}.jsonl` (the exact LoRA training and validation files, with `split.json`) |
| `models/` | the trained LoRA adapters, one folder per system; `models/README.md` says which is which and how to load them |
| `data/source_and_licence.txt` | where every data file comes from, its licence, and how to get what is not included |

Inside `code/`:

| Path | Contents |
|---|---|
| `notebook/CE893_walkthrough.ipynb` | **start here**: guided walk through data, systems and results, runs from the stored results in about a minute without a GPU |
| `notebook/CE893_evaluation.ipynb` | compact notebook that recomputes every number and figure in the report |
| `harness/` | the ADK agent, tools, prompts, Cube client, evaluator (`evaluate.py`) and SQL-to-Cube translator |
| `scripts/` | data preparation, LoRA training (`make_sft.py`, `train_lora.py`), export to Ollama, and `make_submission.py` (built this ZIP) |
| `cube/` | Cube server project; `cube/model/*.yml` are the generated semantic-layer models of the five databases |
| `training_data/`, `training_data_sv3/` | the 3,531 execution-verified question to Cube query pairs (BIRD train) and their schema texts; `_sv3` adds the lookup steps used by the lookup LoRAs (`_sv2` is an earlier variant, not in the report) |
| `results/runs/` | stored output of every reported run: one line per question with the prediction, correctness and full tool trace |
| `results/training/` | LoRA settings and loss curves |
| `report/analysis.py`, `report/make_figures.py` | every number and figure of the report, computed from `results/` |
| `app/` | demo app (A2UI + Highcharts) |
| `tests/` | end-to-end tests with a scripted stand-in for the model |
| `README.md`, `docs/README_technical.md` | the repository README and the detailed technical notes |

## Reproducing the results

**From the stored results (no GPU, about a minute).** Python 3.11+:

```
cd code
python -m venv .venv
.venv\Scripts\activate            (macOS/Linux: source .venv/bin/activate)
pip install pandas sqlglot scipy matplotlib jupyter
jupyter notebook notebook/CE893_walkthrough.ipynb      then Run All
```

**Running the systems again (needs Ollama; training needs a 16 GB NVIDIA GPU).** The code expects the BIRD
files in `code/data/bird/`. Copy them there once:

```
cd code
mkdir data
xcopy /E /I ..\data\original\bird data\bird        (macOS/Linux: cp -r ../data/original/bird data/bird)
pip install -r requirements.txt
```

Then follow "Reproducing the results, 2." in `code/README.md` from the `build_duckdb.py` step (skip
`download_bird.py`). The LoRA training files that `make_sft.py` writes are the ones in
`data/processed/lora_training/`.

## Not included, and why

| Item | Size | Why left out | How to get it |
|---|---|---|---|
| Base models Qwen2.5-Coder-7B/14B-Instruct | 15 to 30 GB each | third-party weights, too large | Hugging Face `Qwen/Qwen2.5-Coder-7B-Instruct`, `-14B-Instruct`; Ollama `qwen2.5-coder:7b`, `:14b` |
| Merged/quantised Ollama exports of the adapters | tens of GB | too large; rebuilt exactly from `models/lora/` | `code/scripts/export_ollama.py` (see `models/README.md`) |
| BIRD train databases | several GB | too large; only needed to rebuild `training_data/` from scratch | `https://bird-bench.oss-cn-beijing.aliyuncs.com/train.zip` (see `data/source_and_licence.txt`) |
| Other BIRD dev databases | about 1.4 GB | not used by the project | `code/scripts/download_bird.py` |
| Generated DuckDB file, `node_modules/` | | rebuilt by the commands above | |
