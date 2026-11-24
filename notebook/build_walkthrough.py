"""Writes notebook/CE893_walkthrough.ipynb, the guided notebook for readers new to the project.

    python notebook/build_walkthrough.py
    jupyter nbconvert --to notebook --execute --inplace notebook/CE893_walkthrough.ipynb

It reads only the stored results in results/ (no GPU, Cube or Ollama) and shows the report's figures from
report/figures/ without redrawing them. notebook/CE893_evaluation.ipynb (build_notebook.py) is the compact
reproduction notebook and is left unchanged.
"""
from pathlib import Path

import nbformat as nbf

nb = nbf.v4.new_notebook()
C = []
md = lambda s: C.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: C.append(nbf.v4.new_code_cell(s.strip()))

# ---------------------------------------------------------------- 0. intro
md(r"""
# CE893 Generative AI project: a guided walkthrough

**Question.** A small open-weight LLM (Qwen2.5-Coder, 7B or 14B, on one consumer GPU) answers plain-English
questions about a database by writing queries for **Cube**, a semantic layer, inside an agent built with
Google's **Agent Development Kit (ADK)**. With the same agent, tools and prompt, how much does a **LoRA adapter**
add over **prompt engineering alone** (the "harness")? And can the adapter take over part of the prompt, so each
question costs fewer tokens?

**What this notebook does.** It walks through the project in the order of the report and recomputes every
number shown from the stored test runs in `results/runs/` (one JSON line per question per system, including
the model's full tool trace). Each section says what it shows, what the code does, and what to read from the output.

**How to run it.** No GPU, Cube or model server is needed; *Run All* takes about a minute.

```bash
pip install pandas sqlglot scipy matplotlib jupyter
jupyter notebook notebook/CE893_walkthrough.ipynb
```

| Section | Report |
|---|---|
| 1. Setup | |
| 2. The task, with one real example | Sections 1–2, Fig. 1–2 |
| 3. The data | Section 3, Table 2, Fig. 3 |
| 4. The compared systems | Section 5.1 |
| 5. How answers are scored | Table 4 |
| 6. Main result: does the LoRA help? | Table 5, Fig. 7 |
| 7. Where the gain comes from | Fig. 7b, Section 6 |
| 8. Moving the prompt into the weights | Fig. 8 |
| 9. How the models use the tools | Table 6, Fig. 9 |
| 10. Training curves | Fig. 6 |
| 11. What goes wrong | Fig. 10 |
| Appendix A. How the harness was developed | Fig. 4 |
| Appendix B. Commands that produced the stored runs | Table 3 |

All results are on the 332 test questions **without BIRD's hint** (see section 3). **Strict accuracy is the
headline metric.**
""")

# ---------------------------------------------------------------- 1. setup
md(r"""
## 1. Setup

All numbers come from `report/analysis.py`, the same module that produces the report's tables and figures, so the
notebook and the report cannot disagree. The cell below finds the repository root, imports that module and lists
the stored inputs. `show_figure` displays a figure of the report as it was printed (`report/figures/*.png`).
""")
code(r"""
import sys, json
from pathlib import Path
import pandas as pd
from IPython.display import Image, Markdown, display

ROOT = Path.cwd() if (Path.cwd() / "results").exists() else Path.cwd().parent
sys.path.insert(0, str(ROOT / "report"))
import analysis as A

pd.set_option("display.width", 160)
pd.set_option("display.max_colwidth", 90)

def show_figure(name, width=720):
    display(Image(filename=str(ROOT / "report" / "figures" / f"{name}.png"), width=width))

print(len(list(A.RUNS.glob("test_nohint_*.jsonl.gz"))), "stored test runs without the hint,",
      len(list(A.TRAINING.glob("*.json"))), "training logs")
""")

# ---------------------------------------------------------------- 2. task
md(r"""
## 2. The task, with one real example

The model gets a question and the **Cube schema** of the database (cube names, their dimensions and measures, with
BIRD's column descriptions). It answers by calling three tools in a loop run by the ADK agent:

* `search_values(dimension, text)` finds exact stored values, e.g. the spelling of a driver's name;
* `run_query(query)` runs a draft Cube query and returns the first rows or Cube's error;
* `submit_answer(query)` hands in the final query.

The submitted query is run on Cube, and the rows are compared with the rows of BIRD's gold SQL on the original
database. Below is the system diagram from the report, then one test question traced through two systems: the
base 7B model with the harness only, and the same model with the LoRA adapter. Lines starting with `>` are the
model's tool calls, lines starting with `<` are the harness's replies.
""")
code(r"""
show_figure("fig02_pipeline")
""")
code(r"""
h, l = A.test_run("harness_7b"), A.test_run("lora_7b")
qid = 923
print("QUESTION:", h.loc[qid, "question"])
print("GOLD SQL:", h.loc[qid, "gold_sql"], "\n")
for name, df in [("harness only", h), ("LoRA, full prompt", l)]:
    row = df.loc[qid]
    print(f"--- 7B {name}: {'CORRECT' if row.correct else 'wrong'}, {row.steps} model calls")
    print(A.render_trace(row, 160), "\n")
""")
md(r"""
A Cube query is JSON: `measures` (aggregates such as counts and sums), `dimensions` (fields to group by or return),
`filters`, `order` and `limit`. Cube writes the SQL, including the joins, from the schema. This is why the
schema in the prompt matters so much: the model can only use member names that exist there.
""")

# ---------------------------------------------------------------- 3. data
md(r"""
## 3. The data

Everything comes from **BIRD** (Li et al., NeurIPS 2023; CC BY-SA 4.0), a text-to-SQL benchmark with real
databases. No database is in more than one split:

* **Test**: all 332 BIRD dev questions on `formula_1` and `student_club`, frozen before any experiment.
* **Harness tuning**: BIRD dev questions on three other databases, used to develop the prompt (Appendix A).
* **LoRA training**: BIRD *train* questions on 58 other databases. Their gold SQL was translated into Cube queries
  by a rule-based translator (`harness/sql2cube.py`) and a pair was kept **only if the Cube result equals the SQL
  result**. Each kept pair was then written out as a full agent conversation in the harness's exact format.

BIRD adds an *evidence hint* to each question (e.g. "full name refers to forename, surname"). A real user would not
type that, so every reported run hides it from the model (`--no-evidence`).

The cell counts the test questions per database and difficulty, and the training pairs.
""")
code(r"""
test = A.test_run("harness_7b")
display(pd.crosstab(test.db_id, test.difficulty, margins=True, margins_name="total"))

train = pd.read_json(ROOT / "training_data" / "verified.jsonl", lines=True)
def cubes_used(q):
    members = q.get("measures", []) + q.get("dimensions", []) + [f["member"] for f in q.get("filters", []) if "member" in f] + list(q.get("order", {}))
    return {m.split(".")[0] for m in members}

print(f"{test.has_join.mean():.0%} of the test questions need a join in the gold SQL")
print(f"{len(train):,} verified training pairs on {train.db_id.nunique()} BIRD train databases; "
      f"{(train.cube_query.apply(cubes_used).str.len() > 1).mean():.0%} of their Cube queries use more than one cube (a join)")
""")
md("One training pair: the question, BIRD's SQL, and the Cube query that gives the same rows.")
code(r"""
ex = train.iloc[0]
print("QUESTION:", ex.question)
print("SQL:     ", ex.sql)
print("CUBE:    ", json.dumps(ex.cube_query))
print("ROWS:    ", ex.preview)
""")
md("And the start of the schema text the model sees for that training database (the same format is used for the test databases):")
code(r"""
print((ROOT / "training_data" / "schemas" / f"{ex.db_id}.txt").read_text()[:900], "...")
""")
md("The report's Fig. 3 shows how the 8,003 training questions became 3,531 verified pairs:")
code(r"""
show_figure("fig03_data")
""")

# ---------------------------------------------------------------- 4. systems
md(r"""
## 4. The compared systems

All systems use the **same harness** (agent, tools and checks; version v4.2, frozen before fine-tuning) and the
same Qwen2.5-Coder model file, 4-bit in Ollama. The names give the adapter and the prompt it was trained and run with.

| System | Adapter | Prompt |
|---|---|---|
| **harness only** (baseline) | none | full: query guide + tool schemas + full database schema |
| **LoRA, full prompt** (proposed) | trained on the 3,531 transcripts | full |
| **LoRA, lean prompt** | trained with the lean prompt | lean: no query guide, no tool schemas, compact schema |
| **noschema LoRA, no schema** | also trained on synthetic questions about the two test schemas | database name only |
| **noschema LoRA, lean prompt** | same adapter | lean |
| **lookup LoRA, full / lean prompt** (extra arms) | retrained to look filter values up (more lookup examples, trained `<tool_call>` tokens) | full / lean |

Each system was run at 7B and 14B. The cell lists the stored run behind each name.
""")
code(r"""
REPORTED = ["harness_7b", "lora_7b", "lean_7b", "noschema_7b", "noschema_lean_7b", "sv3_7b", "lean_sv3_7b",
            "harness_14b", "lora_14b", "lean_14b", "noschema_14b", "noschema_lean_14b", "sv3_14b", "lean_sv3_14b"]
pd.DataFrame([{"system": A.TEST_RUNS[k][1], "stored run": f"results/runs/{A.TEST_RUNS[k][0]}.jsonl.gz"} for k in REPORTED])
""")

# ---------------------------------------------------------------- 5. scoring
md(r"""
## 5. How answers are scored

**Execution accuracy**: the submitted Cube query is run and its rows are compared with the rows of the gold SQL
(as sets, column order ignored).

* **Strict** (headline): same rows and the same number of columns.
* **Relaxed**: also accepts extra columns. Cube must return every dimension it sorts by, while SQL can sort by a
  column it does not return, so some correct answers carry one extra column.

The cell shows a question that is correct under relaxed but not strict scoring.
""")
code(r"""
x = l[l.correct_relaxed & ~l.correct].iloc[0]
print("QUESTION:", x.question)
print("GOLD SQL:", x.gold_sql)
print("PREDICTED CUBE QUERY:", json.dumps(x.pred_query))
""")
md("The predicted query returns the lap time as well as the name, because it orders by it; the gold SQL orders by the time without returning it.")

# ---------------------------------------------------------------- 6. main result
md(r"""
## 6. Main result: does the LoRA help?

The cell rebuilds the report's Table 5: strict and relaxed accuracy, prompt tokens per question (summed over all
model calls), model calls ("steps"), and model latency per question.

The *p* column is an **exact McNemar test** on strict accuracy against the harness-only system of the same size.
Both systems answer the same 332 questions, so the test only looks at the questions where exactly one of the two is
right (`only_a` vs `only_b`) and asks whether that split could be a coin toss.
""")
code(r"""
ROWS = ["harness_7b", "lora_7b", "lean_7b", "harness_14b", "lora_14b", "lean_14b",
        "sv3_7b", "lean_sv3_7b", "sv3_14b", "lean_sv3_14b"]
t = A.summary_table(ROWS)
t["p vs harness only"] = [None if k.startswith("harness") else A.mcnemar("harness_" + k.split("_")[-1], k)["p"] for k in ROWS]
table5 = t[["run", "strict", "relaxed", "prompt_tokens", "steps", "latency_s", "latency_median_s", "latency_p90_s", "p vs harness only"]]
table5.style.format({"strict": "{:.1f}", "relaxed": "{:.1f}", "prompt_tokens": "{:,.0f}", "steps": "{:.1f}",
                     "latency_s": "{:.1f}", "latency_median_s": "{:.1f}", "latency_p90_s": "{:.1f}",
                     "p vs harness only": lambda p: "" if pd.isna(p) else f"{p:.0e}"}).hide(axis="index")
""")
md("The paired counts behind the two headline comparisons:")
code(r"""
pd.DataFrame([A.mcnemar("harness_7b", "lora_7b"), A.mcnemar("harness_14b", "lora_14b"),
              A.mcnemar("harness_14b", "lora_7b")]).assign(p=lambda d: d.p.map("{:.1e}".format))
""")
md(r"""
**How to read it.** The LoRA adds about 15 points of strict accuracy at 7B (21.4 → 36.7) and 13 at 14B
(27.4 → 40.7), with *p* far below 0.05. The fine-tuned 7B even beats the 14B harness-only model (last row: 14B
harness only is right on fewer of the disputed questions). It also needs about half the model calls and half the
latency, since it gets the query right in fewer tries.
""")
code(r"""
show_figure("fig07_main")
""")

# ---------------------------------------------------------------- 7. breakdown
md(r"""
## 7. Where the gain comes from

Strict accuracy of the four main systems split by database, by whether the gold SQL needs a join, and by BIRD's
difficulty label.
""")
code(r"""
rows = []
for run in A.MAIN_ARMS:
    for by in ["db_id", "has_join", "difficulty"]:
        for k, v in A.breakdown(run, by).items():
            rows.append({"system": A.TEST_RUNS[run][1], "group": f"{by} = {k}", "strict": v})
pd.DataFrame(rows).pivot(index="group", columns="system", values="strict")[[A.TEST_RUNS[r][1] for r in A.MAIN_ARMS]].round(1)
""")
md(r"""
The gain holds on both databases. At 7B it holds with and without a join; at 14B it comes entirely from questions
that need a join (22.9 → 40.7), where the adapter has learned which cube a field lives in and lets Cube build the
join. Most training queries use more than one cube (section 3).
""")

# ---------------------------------------------------------------- 8. offloading
md(r"""
## 8. Moving the prompt into the weights

The full prompt is long (about 8–16k tokens per question over all calls). Can the adapter learn what the prompt
spells out, so the prompt can be shorter? The cell compares the full-prompt LoRA with:

* the **LoRA, lean prompt** (trained and run without the query guide and tool schemas, compact schema);
* the **noschema LoRA** run with no schema at all, and with the lean prompt;
* a **control**: the full-prompt LoRA given the lean prompt *without* retraining (7B).
""")
code(r"""
OFF = ["lora_7b", "lean_7b", "noschema_7b", "noschema_lean_7b", "lora_on_lean_7b",
       "lora_14b", "lean_14b", "noschema_14b", "noschema_lean_14b"]
o = A.summary_table(OFF)
o["p vs LoRA, full prompt"] = [None if k.startswith("lora_") and "lean" not in k else A.mcnemar("lora_" + k.split("_")[-1], k)["p"] for k in OFF]
o[["run", "strict", "relaxed", "prompt_tokens", "p vs LoRA, full prompt"]].style.format(
    {"strict": "{:.1f}", "relaxed": "{:.1f}", "prompt_tokens": "{:,.0f}",
     "p vs LoRA, full prompt": lambda p: "" if pd.isna(p) else f"{p:.2g}"}).hide(axis="index")
""")
md(r"""
**How to read it.** The lean prompt costs about 40% of the full prompt's tokens and loses only about 2 points at
7B and less than 1 at 14B, neither significant. Without any schema, accuracy falls by 16–19 points: the model
invents field names (count below). With the short schema back, the noschema LoRA is level with the full-prompt
LoRA. The control scores 0%: an adapter only works with the prompt it was trained on.
""")
code(r"""
for k in ["noschema_14b", "lean_14b"]:
    df = A.test_run(k)
    n = df.trace.apply(lambda t: any("Unknown member" in (s.get("content") or "") for s in t or [])).sum()
    print(f"{A.TEST_RUNS[k][1]}: an 'unknown member' error in {n} of {len(df)} questions")
""")
code(r"""
show_figure("fig08_offload")
""")

# ---------------------------------------------------------------- 9. tool use
md(r"""
## 9. How the models use the tools

Per question: `search_values` calls (lookups), `run_query` calls (drafts), calls the harness or Cube turned back
(rejected), the share of tool calls the model wrote as plain text instead of in `<tool_call>` tags (the harness
parses those with a text fallback), and the share of answers whose last draft returned no rows (empty answers).
This is the report's Table 6.
""")
code(r"""
TOOLS = ["harness_7b", "lora_7b", "harness_14b", "lora_14b", "sv3_7b", "lean_sv3_7b", "sv3_14b", "lean_sv3_14b"]
A.tool_table(TOOLS).set_index("run")[["search_values_per_q", "run_query_per_q", "rejected_per_q", "text_calls_pct", "empty_answer_pct"]].round(2)
""")
md(r"""
**How to read it.** The first adapters almost never look values up (0.01 and 0.00 lookups per question): only
22% of their training transcripts had a lookup. So when a filter value is spelled differently in the data, they
get zero rows and submit anyway (13% empty answers). The **lookup LoRAs** were retrained with a lookup before
every text filter and with the `<tool_call>` tokens trained (they are untrained in the base model). They look
values up about once per question, cut empty answers to 4–6%, and gain 5.2 points at 7B and 4.2 at 14B:
""")
code(r"""
pd.DataFrame([A.mcnemar("lora_7b", "sv3_7b"), A.mcnemar("lora_14b", "sv3_14b"),
              A.mcnemar("lean_14b", "lean_sv3_14b"), A.mcnemar("lean_7b", "lean_sv3_7b")]).assign(p=lambda d: d.p.map("{:.3f}".format))
""")
md(r"""
The 14B lookup LoRA with the lean prompt is the best system of the study (46.1% strict at 4.7k prompt tokens).
The 7B one with the lean prompt loops between lookups and drafts and does not improve; it is kept as a negative
result. Real traces from the report:
""")
code(r"""
show_figure("fig09_outputs")
""")

# ---------------------------------------------------------------- 10. training
md(r"""
## 10. Training curves

Each adapter is a QLoRA (rank 16) on Qwen2.5-Coder-Instruct, trained for 2 epochs on the transcripts, with the
loss only on the model's own turns. The cell lists each run's settings and its validation loss at the start and
every half epoch (from the stored Trainer logs). `train_examples` counts only the transcripts that fit in
`max_len` tokens; the 14B fits 16 GB of GPU memory only at 4,096 tokens, so it saw about 10% fewer examples. The
noschema adapters also see the synthetic questions about the test schemas.
""")
code(r"""
pd.DataFrame([{"run": n, **{k: A.training_log(n)["config"][k] for k in ["base", "max_len", "train_examples", "val_examples"]},
               "hours": round(A.training_log(n)["runtime_h"], 1),
               "validation loss": [round(x, 3) for x in A.training_log(n)["eval"]["eval_loss"]]}
              for n in A.TRAIN_LOGS]).set_index("run")
""")
md("The loss drops within the first quarter epoch and then flattens, so 2 epochs are enough:")
code(r"""
show_figure("fig06_training")
""")

# ---------------------------------------------------------------- 11. errors
md(r"""
## 11. What goes wrong

Each wrong answer gets one category from a set of rules (`analysis.error_category`, applied in order). The rules
compare the predicted Cube query with the gold SQL, so they describe *where* the answer differs, not *why* the model
erred. "Needs SQL Cube cannot express" marks questions that no Cube query can answer (nested queries, CASE
arithmetic, string slicing): about 30% of the test set, a ceiling for every system.
""")
code(r"""
pd.DataFrame({A.TEST_RUNS[r][1]: A.error_counts(r) for r in A.MAIN_ARMS})
""")
code(r"""
show_figure("fig10_errors")
""")
md("So that the report does not show only chosen cases, here are five LoRA errors drawn at random (fixed seed):")
code(r"""
for qid, row in l[~l.correct].sample(5, random_state=0).iterrows():
    print(f"Q{qid} [{A.error_category(row)}] {row.question}")
    print("   predicted:", json.dumps(row.pred_query)[:200])
    print("   gold SQL: ", row.gold_sql[:200], "\n")
""")

# ---------------------------------------------------------------- appendix A
md(r"""
## Appendix A. How the harness was developed

The harness-only baseline is not a weak straw man: its prompt and tools went through five versions, tuned on
three *other* databases. These development runs **kept BIRD's hint** (the setting used while building the harness),
so they are not comparable with the results above; they show only how far prompt engineering got before the
harness was frozen (v4.2).
""")
code(r"""
dev = {**{f"test, {k.split('_')[2]}": A.summary(A.test_run(k)) for k in ["hint_harness_v1_7b", "hint_harness_v2_7b"]},
       "test, v4.2": A.summary(A.test_run("hint_harness_7b")),
       **{f"tuning (120 q), {k}": A.summary(A.load_run(v)) for k, v in A.TUNE_RUNS.items()},
       **{f"tuning (363 q), {k}": A.summary(A.load_run(v)) for k, v in A.TUNE363_RUNS.items()}}
pd.DataFrame(dev).T[["n", "strict", "relaxed", "steps"]].round(1)
""")
code(r"""
show_figure("fig04_harness_versions")
""")

# ---------------------------------------------------------------- appendix B
md(r"""
## Appendix B. Commands that produced the stored runs

Not run here: they need BIRD downloaded, Cube on port 4000, Ollama, and a 16 GB GPU for training (one evaluation
takes 10–25 minutes; training about 5 h at 7B and 9 h at 14B on an RTX 5070 Ti). The README has the full setup.
""")
code(r"""
STEPS = '''
# data: BIRD -> DuckDB -> Cube models; training pairs
python scripts/download_bird.py
python scripts/build_duckdb.py --bird-dir data/bird --out data/bird.duckdb --db formula_1 --db student_club
python scripts/gen_cube_models.py --bird-dir data/bird --out cube/model --db formula_1 --db student_club
python scripts/make_sft.py --data training_data --out data/sft

# harness only (7B; 14B: qwen2.5-coder:14b)
python -m harness.evaluate --db formula_1 --db student_club --ids-file splits/test_ids.json --no-evidence --model ollama_chat/qwen2.5-coder:7b --setup harness-v4.2-7b-nohint --out runs/test_nohint_harness_7b.jsonl

# LoRA, full prompt (14B: Qwen2.5-Coder-14B-Instruct, --max-len 4096)
python scripts/train_lora.py --base Qwen/Qwen2.5-Coder-7B-Instruct --out lora/qwen7b
python scripts/export_ollama.py --base Qwen/Qwen2.5-Coder-7B-Instruct --adapter lora/qwen7b/final --ollama-base qwen2.5-coder:7b --name qwen7b-cube-lora
python -m harness.evaluate --db formula_1 --db student_club --ids-file splits/test_ids.json --no-evidence --model ollama_chat/qwen7b-cube-lora --setup lora-v4.2-7b-nohint --out runs/test_nohint_lora_7b.jsonl

# LoRA, lean prompt and noschema LoRA
python scripts/make_sft.py --data training_data --profile lean --out data/sft_lean
python scripts/synth_schema_qa.py --db formula_1 --n 800 --db student_club --n 500 --out data/synth/synth.jsonl --audit
python scripts/make_sft.py --data training_data --profile noschema --synthetic data/synth/synth.jsonl --out data/sft_noschema
python -m harness.evaluate ... --profile lean --model ollama_chat/qwen7b-cube-lean

# lookup LoRA (details: docs/README_technical.md)
python scripts/make_sft.py --data training_data_sv3 --out data/sft_sv3
python scripts/train_lora.py --base Qwen/Qwen2.5-Coder-7B-Instruct --data data/sft_sv3 --out lora/qwen7b-sv3 --train-tokens "<tool_call>,</tool_call>"
'''
print(STEPS.strip())
""")
md("After a new run, gzip its output into `results/runs/` under the file name used in `report/analysis.py` and run this notebook again.")

nb["cells"] = C
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
out = Path(__file__).parent / "CE893_walkthrough.ipynb"
nbf.write(nb, out)
print("wrote", out)
