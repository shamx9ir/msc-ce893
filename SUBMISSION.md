# FASER submission checklist

What the module's *Project Deliverables* guide (CE893, 2026-27) asks for, where each item lives in this
repository, and how to build the ZIP. Build it with:

```
python scripts/make_submission.py
```

It writes `dist/DIMAA21403-project.zip` (about 2.4 GB with the LoRA adapters, 39 MB without) plus an unzipped copy in `dist/DIMAA21403-project/`
to look through, and ends with a **CHECK BEFORE UPLOADING** list of anything that still needs attention.
It uses only the Python standard library (plus `pypdf` for the page count, `jupyter` for `--run-notebooks`) and
works the same on Windows. Run it on the experiment PC, where `data/bird` and `data/synth/synth.jsonl` already
exist; elsewhere add `--download-bird` (350 MB download).

| Option | Effect |
|---|---|
| `--report outline` (default) / `--report full` | which version becomes `report.pdf` (see open question 1) |
| `--run-notebooks` | executes both notebooks inside the unzipped copy before zipping (the guide's check 1; about 1.5 minutes) |
| `--download-bird` | downloads BIRD dev if `data/bird` is missing |
| `--bird-dir PATH` | BIRD dev folder somewhere else |
| `--no-sft` | leaves out the LoRA training files (smaller ZIP) |
| `--lora-dir PATH` | where the trained adapters are, repeatable (default `lora/` plus `../msc-ce893-schema/lora/` on your PC, where the lean and noschema adapters live) |
| `--weights-zip` | puts the adapters in a second ZIP, `DIMAA21403-project-weights.zip`, if one file is too big for FASER |
| `--no-weights` | leaves the adapters out |
| `--reg` | registration number in the folder name (default `DIMAA21403`) |

## Suggested ZIP structure

The guide's suggested layout, with `code/` kept in the repository's own layout so every relative path in the
code and notebooks still works:

```
DIMAA21403-project/
|-- README.md                      what is where, how to run, what is left out (from submission/README.md)
|-- report.pdf                     report/main_outline.pdf (or main.pdf with --report full)
|-- report-source/
|   |-- main.tex                   main_outline.tex renamed (or main.tex)
|   |-- report_outline.tex         the text (report.tex for the full version)
|   |-- preamble.tex, ce893.bst, essex_logo.png, Archivio/   module template
|   |-- references.bib
|   |-- figures/                   only the 10 figures the report uses (.pdf + .png)
|   `-- BUILD.txt                  latexmk -xelatex main.tex
|-- code/
|   |-- notebook/                  CE893_walkthrough.ipynb (start here), CE893_evaluation.ipynb
|   |-- harness/  scripts/  cube/  app/  tests/
|   |-- training_data/  training_data_sv2/  training_data_sv3/   verified training pairs (read by the code)
|   |-- results/runs/  results/training/                      stored outputs of every reported run
|   |-- report/analysis.py, make_figures.py, figures/         the notebooks import these
|   |-- splits/  requirements*.txt  README.md  docs/README_technical.md
|-- models/
|   |-- README.md                  which adapter is which system, sizes, how to load them (PEFT or Ollama)
|   `-- lora/<adapter>/            every trained adapter: lora/<name>/final + train_info.json (no checkpoints)
`-- data/
    |-- original/bird/             BIRD dev, only the 5 databases used: dev.json, dev_tables.json, dev_databases/<db>/
    |-- processed/
    |   |-- test_questions.jsonl   332 frozen test questions + hint + gold SQL
    |   |-- tune_questions.jsonl   363 harness-development questions
    |   |-- splits/                test_ids.json, tune_ids.json
    |   |-- lora_training/<adapter>/sft_tools_train.jsonl, sft_tools_val.jsonl, split.json
    |   `-- synthetic/synth.jsonl  (noschema adapter; only if present on the machine that builds the ZIP)
    `-- source_and_licence.txt     BIRD citation + CC BY-SA 4.0, how to get what is not included
```

## Requirement checklist

### 1. Code

| Guide requirement | Where | Status |
|---|---|---|
| Completed notebook (.ipynb) or the .py files written | `code/notebook/CE893_walkthrough.ipynb`, `CE893_evaluation.ipynb`; all `.py` under `code/` | ✅ |
| Every supporting script or configuration file | `harness/`, `scripts/`, `cube/` (package.json, `.env.example`, `model/*.yml`), `harness/tool_schema.json`, `requirements*.txt` | ✅ |
| Sensible running order, short comments | notebooks run top to bottom; the walkthrough's last section lists the full pipeline commands in order; `README.md` "Reproducing the results" | ✅ |
| Code contains the experiments and evaluation behind the report | `harness/evaluate.py` (runs), `report/analysis.py` (every number), `report/make_figures.py` (every figure), stored outputs in `results/runs/` | ✅ |
| Relative paths only, no Google Drive or local paths | checked: no absolute paths in code or notebooks; both notebooks run from the unzipped copy (`--run-notebooks`) | ✅ |

### 2. Data

| Guide requirement | Where | Status |
|---|---|---|
| Original data used by the code | `data/original/bird/` (5 BIRD dev databases + their questions) | ✅ when BIRD is on disk |
| Cleaned / processed / split versions | `data/processed/` (test and tuning question files, splits); verified training pairs in `code/training_data*/` (kept there because the code reads them) | ✅ |
| Fine-tuning: training, validation **and test** files | train/val: `data/processed/lora_training/<adapter>/` (exact `make_sft.py` output, 3,273 / 258 examples); test: `data/processed/test_questions.jsonl` | ✅ (noschema adapter only if `data/synth/synth.jsonl` exists) |
| Prompt engineering: development and test inputs + expected outputs | dev: `tune_questions.jsonl`; test: `test_questions.jsonl`; expected output = gold SQL, checked by execution | ✅ |
| Source, licence, attribution | `data/source_and_licence.txt` (BIRD, CC BY-SA 4.0, which allows redistribution with attribution) | ✅ |
| Files that cannot be shared: link + instructions/script | BIRD train (several GB): URL + `build_training_data.py` command in `source_and_licence.txt` | ✅ |

### 3. Report

| Guide requirement | Where | Status |
|---|---|---|
| Final report PDF | `report.pdf` | ✅ outline version (decided) |
| LaTeX source: .tex, figures, bibliography | `report-source/` | ✅ every `\input`, figure and `.bib` it references is included; compile once to confirm (check 4) |
| Max 10 pages | both PDFs are 11 pages = title page + 10, confirmed OK with the module on 2026-11-26 | ✅ |
| Module LaTeX template | yes (`preamble.tex`, `ce893.bst`, logo, Archivo font) | ✅ |
| Self-contained without the notebook | problem, data, method, experiment, results, conclusions all in the report | ✅ |

### Final checks before uploading (guide section 3)

1. [ ] Run the code from a fresh environment: `python scripts/make_submission.py --run-notebooks` runs both notebooks inside the unzipped copy. Tested in the cloud: both ran without errors.
2. [ ] Every file the code loads is in the ZIP or can be obtained: notebooks read only `code/` files; the pipeline reads `data/bird` (copy from `data/original/bird`, see the ZIP README) and BIRD train (download instructions).
3. [ ] Saved outputs agree with the reported numbers: the notebooks recompute every table from `results/runs/`; compare with the PDF's tables.
4. [ ] Compile `report-source/` (`latexmk -xelatex main.tex`, or upload the folder to Overleaf with the XeLaTeX compiler) and check figures, tables, citations and references.
5. [ ] Page count and readable figure text (the script warns above 11 pages, title page + 10).
6. [ ] Open the ZIP and check it contains code, data and the full report source (the script also checks the key files).

## Left out of the ZIP, and why

| Item | Why | How a reader gets it |
|---|---|---|
| Base models (Qwen2.5-Coder 7B / 14B) | third-party weights, 15 to 30 GB | Hugging Face / Ollama, named in the README |
| Merged / GGUF / Ollama exports of the adapters | tens of GB, and rebuilt exactly from the included adapters | `export_ollama.py` (command in `models/README.md`) |
| BIRD train databases (`data/bird_train`) | several GB; only needed to rebuild the training pairs, which are included | URL + command in `source_and_licence.txt` |
| The other 6 BIRD dev databases | not used | `scripts/download_bird.py` |
| `data/bird.duckdb`, `node_modules/`, raw `runs/` | generated / reinstallable; gzipped copies of all reported runs are in `results/runs/` | commands in the README |
| `proposal/` | not one of the three deliverables | GitHub repository |
| `report/main.pdf` + `report.tex` (outline submission) or the outline files (full submission) | only the submitted version's source goes in `report-source/` | GitHub repository |

## Open questions for Shamir

1. ~~Which report goes in?~~ Decided 2026-11-26: the outline version (`main_outline.pdf`, your own text), the
   script's default.
2. ~~Page limit~~ Confirmed 2026-11-26: 11 pages (title page + 10) is fine. The script warns only above 11.
3. ~~LoRA adapters~~ Decided 2026-11-26: include them (`models/lora/`). The base models and the merged/Ollama
   exports stay out (15 to 30 GB each, rebuildable). Your PC has 11 adapters, 2.4 GB in total (7B: 170 MB,
   14B: 278 MB each; 7 in `msc-ce893/lora/`, the 4 lean/noschema ones in `msc-ce893-schema/lora/`); the script
   picks up both folders. If FASER refuses a 2.4 GB file, `--weights-zip` puts them in a second ZIP.
4. **noschema adapter training file** (`data/synth/synth.jsonl`) only exists on your PC, so build the final ZIP
   there to include it.
5. **FASER upload limit.** The guide gives none. Without weights the ZIP is about 39 MB (14 MB BIRD databases,
   13 MB LoRA training files); the adapters add 2.4 GB (safetensors barely compress).
