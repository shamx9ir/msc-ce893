"""Build the FASER submission ZIP (code + data + report) from this repository.

Layout follows the module's "Project Deliverables" guide (see SUBMISSION.md):

    <REG>-project/
    |-- README.md                  what is where, how to run it (from submission/README.md)
    |-- report.pdf                 the final report (outline version = Shamir's text, or the full draft)
    |-- report-source/             LaTeX source that compiles report.pdf (main.tex, references.bib, figures/ ...)
    |-- code/                      snapshot of the repository's tracked code, notebooks and stored results
    |-- models/lora/<adapter>/     trained LoRA adapters (lora/*/final), unless --no-weights
    `-- data/
        |-- original/bird/         BIRD dev files for the 5 databases used (questions, tables, SQLite + descriptions)
        |-- processed/             test/tuning question files, splits, LoRA train/validation files per adapter
        `-- source_and_licence.txt

Works on Windows, macOS and Linux (standard library only; pypdf is used for the page check if installed).
Run it from anywhere; paths are resolved from the repository root:

    python scripts/make_submission.py                       # outline version, BIRD from data/bird
    python scripts/make_submission.py --report full         # submit the full draft instead
    python scripts/make_submission.py --download-bird       # fetch BIRD dev first if data/bird is missing
    python scripts/make_submission.py --run-notebooks       # also execute both notebooks inside the staged copy
    python scripts/make_submission.py --weights-zip         # adapters in a second ZIP if one file is too big

The ZIP is written to dist/<REG>-project.zip (dist/ is git-ignored).
"""
import argparse
import gzip
import json
import shutil
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TEST_DBS = ["formula_1", "student_club"]
TUNE_DBS = ["superhero", "toxicology", "california_schools"]
USED_DEV_DBS = TEST_DBS + TUNE_DBS
PAGE_LIMIT = 10

# Report files per version: (file in report/, name inside report-source/)
REPORT_COMMON = ["preamble.tex", "references.bib", "ce893.bst", "essex_logo.png"]
REPORT_VERSIONS = {
    "outline": {"pdf": "main_outline.pdf", "tex": [("main_outline.tex", "main.tex"), ("report_outline.tex", "report_outline.tex")]},
    "full": {"pdf": "main.pdf", "tex": [("main.tex", "main.tex"), ("report.tex", "report.tex")]},
}
# The notebooks import report/analysis.py and draw into report/figures, so code/ keeps these parts of report/.
CODE_KEEP_FROM_REPORT = ("report/analysis.py", "report/make_figures.py", "report/figures/")
# Tracked paths left out of code/ (the report has its own folder; the proposal is not a deliverable).
CODE_EXCLUDE_PREFIXES = ("report/", "proposal/")

# LoRA training/validation files, one set per adapter family in the report (make_sft.py is deterministic, seed 13).
SFT_SETS = [
    ("lora_full", ["--data", "training_data"]),
    ("lora_lean", ["--data", "training_data", "--profile", "lean"]),
    ("lookup_lora_full", ["--data", "training_data_sv3"]),
    ("lookup_lora_lean", ["--data", "training_data_sv3", "--profile", "lean"]),
]
SYNTH = ROOT / "data" / "synth" / "synth.jsonl"  # synthetic questions for the noschema adapter (needs Cube to rebuild)

QUESTION_FIELDS = ["question_id", "db_id", "difficulty", "question", "evidence", "SQL"]


def log(msg: str) -> None:
    print(msg, flush=True)


def git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True, text=True,
                              encoding="utf-8").stdout
    except (OSError, subprocess.CalledProcessError) as e:
        sys.exit(f"git {' '.join(args)} failed ({e}). Run this from a git checkout with git on PATH.")


def tracked_files() -> list[str]:
    return [p for p in git("ls-files", "-z").split("\0") if p]


def copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def pdf_pages(path: Path) -> int | None:
    try:
        from pypdf import PdfReader
    except ImportError:
        return None
    return len(PdfReader(str(path)).pages)


# ---------------------------------------------------------------- report

def build_report(stage: Path, version: str, files: list[str]) -> list[str]:
    warnings = []
    spec = REPORT_VERSIONS[version]
    pdf = ROOT / "report" / spec["pdf"]
    if not pdf.exists():
        sys.exit(f"{pdf} is missing: build it first (see report/README.md).")
    copy(pdf, stage / "report.pdf")
    pages = pdf_pages(pdf)
    if pages is None:
        warnings.append("pypdf not installed: page count of report.pdf not checked (pip install pypdf).")
    elif pages > PAGE_LIMIT:
        warnings.append(f"report.pdf has {pages} pages; the guide says no more than {PAGE_LIMIT} "
                        f"(title page included unless the module says otherwise).")
    log(f"report.pdf <- report/{spec['pdf']} ({pages if pages is not None else '?'} pages)")

    src = stage / "report-source"
    tex_text = ""
    for name, target in spec["tex"]:
        copy(ROOT / "report" / name, src / target)
        tex_text += (ROOT / "report" / name).read_text(encoding="utf-8")
    for name in REPORT_COMMON:
        copy(ROOT / "report" / name, src / name)
    for f in files:  # template font
        if f.startswith("report/Archivio/"):
            copy(ROOT / f, src / f[len("report/"):])
    # only the figures this version uses (both .pdf and .png of each)
    used = set()
    for f in files:
        if f.startswith("report/figures/"):
            stem = Path(f).stem
            if f"figures/{stem}." in tex_text or f"figures/{stem}}}" in tex_text:
                copy(ROOT / f, src / "figures" / Path(f).name)
                used.add(stem)
    log(f"report-source/: {len(spec['tex'])} tex files + preamble, bibliography, {len(used)} figures")
    (src / "BUILD.txt").write_text(
        "Compile with XeLaTeX (the template's Archivo font is in Archivio/):\n\n"
        "    latexmk -xelatex main.tex\n\n"
        "or: xelatex main && bibtex main && xelatex main && xelatex main\n"
        "The figures were drawn by code/report/make_figures.py from code/results/runs/.\n",
        encoding="utf-8")
    return warnings


# ---------------------------------------------------------------- code

def build_code(stage: Path, files: list[str]) -> int:
    n = 0
    for f in files:
        if f.startswith(CODE_EXCLUDE_PREFIXES) and not f.startswith(CODE_KEEP_FROM_REPORT):
            continue
        p = ROOT / f
        if p.is_file():
            copy(p, stage / "code" / f)
            n += 1
    log(f"code/: {n} tracked files")
    return n


# ---------------------------------------------------------------- data

def questions_from_bird(bird: Path, ids: list[int]) -> list[dict]:
    by_id = {q["question_id"]: q for q in json.load(open(bird / "dev.json", encoding="utf-8"))}
    return [{k: by_id[i].get(k) for k in QUESTION_FIELDS} for i in ids]


def questions_from_runs(ids: list[int] | None, pattern: str) -> list[dict]:
    """Fallback when BIRD is not on disk: the stored runs carry each question, hint and gold SQL.
    ids=None returns every question found, in BIRD order."""
    found = {}
    for run in sorted((ROOT / "results" / "runs").glob(pattern)):
        with gzip.open(run, "rt", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                qid = int(r["question_id"])
                found.setdefault(qid, {"question_id": qid, "db_id": r["db_id"], "difficulty": r.get("difficulty"),
                                       "question": r["question"], "evidence": r.get("evidence"), "SQL": r["gold_sql"]})
    if ids is None:
        return [found[i] for i in sorted(found)]
    missing = [i for i in ids if i not in found]
    if missing:
        sys.exit(f"{len(missing)} question ids not found in stored runs ({pattern}); provide BIRD with --bird-dir")
    return [found[i] for i in ids]


def build_data(stage: Path, bird: Path | None, make_sft: bool) -> list[str]:
    warnings = []
    data = stage / "data"
    test_ids = json.loads((ROOT / "splits" / "test_ids.json").read_text(encoding="utf-8"))

    # original: BIRD dev, only the five databases the project uses
    if bird:
        out = data / "original" / "bird"
        dev = [q for q in json.load(open(bird / "dev.json", encoding="utf-8")) if q["db_id"] in USED_DEV_DBS]
        (out).mkdir(parents=True, exist_ok=True)
        (out / "dev.json").write_text(json.dumps(dev, indent=2, ensure_ascii=False), encoding="utf-8")
        tables = [t for t in json.load(open(bird / "dev_tables.json", encoding="utf-8")) if t["db_id"] in USED_DEV_DBS]
        (out / "dev_tables.json").write_text(json.dumps(tables, indent=2, ensure_ascii=False), encoding="utf-8")
        for db in USED_DEV_DBS:
            src = bird / "dev_databases" / db
            if not src.is_dir():
                sys.exit(f"{src} missing: is --bird-dir the unpacked BIRD dev folder?")
            shutil.copytree(src, out / "dev_databases" / db, ignore=shutil.ignore_patterns(".*", "__MACOSX"))
        log(f"data/original/bird/: dev.json ({len(dev)} questions), dev_tables.json, {len(USED_DEV_DBS)} databases")
        test = questions_from_bird(bird, test_ids)
        tune = [{k: q.get(k) for k in QUESTION_FIELDS} for q in dev if q["db_id"] in TUNE_DBS]
    else:
        warnings.append("BIRD dev not found: data/original/ has no BIRD files (only the download instructions). "
                        "Re-run with --bird-dir or --download-bird before submitting.")
        test = questions_from_runs(test_ids, "test_nohint_harness_7b.jsonl.gz")
        tune = questions_from_runs(None, "tune363_*.jsonl.gz")

    proc = data / "processed"
    write_jsonl(proc / "test_questions.jsonl", test)
    write_jsonl(proc / "tune_questions.jsonl", tune)
    for f in ("test_ids.json", "tune_ids.json"):
        copy(ROOT / "splits" / f, proc / "splits" / f)
    log(f"data/processed/: test_questions.jsonl ({len(test)}), tune_questions.jsonl ({len(tune)}), splits/")

    if make_sft:
        sets = list(SFT_SETS)
        if SYNTH.exists():
            copy(SYNTH, proc / "synthetic" / "synth.jsonl")
            sets.append(("noschema_lora", ["--data", "training_data", "--profile", "noschema", "--synthetic", str(SYNTH)]))
        else:
            warnings.append(f"{SYNTH.relative_to(ROOT)} not found: the noschema adapter's training files are not included "
                            "(run on the experiment PC, or rebuild with scripts/synth_schema_qa.py + Cube).")
        for name, args in sets:
            out = proc / "lora_training" / name
            cmd = [sys.executable, str(ROOT / "scripts" / "make_sft.py"), *args, "--out", str(out)]
            r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, encoding="utf-8")
            if r.returncode:
                sys.exit(f"make_sft.py failed for {name}:\n{r.stderr[-2000:]}\n(pip install -r requirements.txt)")
            for f in out.glob("sft_single_*.jsonl"):  # single-turn mode was not used by any reported adapter
                f.unlink()
            log(f"data/processed/lora_training/{name}/: {r.stdout.strip().splitlines()[-1]}")
    else:
        warnings.append("--no-sft: LoRA training/validation files not included.")

    copy(ROOT / "submission" / "source_and_licence.txt", data / "source_and_licence.txt")
    return warnings


# ---------------------------------------------------------------- model weights

# Known adapter folders (lora/<name>/final, written by scripts/train_lora.py) and their names in the report.
ADAPTER_NAMES = {
    "qwen7b": "LoRA, full prompt (7B)", "qwen14b": "LoRA, full prompt (14B)",
    "qwen7b-lean": "LoRA, lean prompt (7B)", "qwen14b-lean": "LoRA, lean prompt (14B)",
    "qwen7b-noschema": "noschema LoRA (7B)", "qwen14b-noschema": "noschema LoRA (14B)",
    "qwen7b-sv3": "lookup LoRA, full prompt (7B)", "qwen14b-sv3": "lookup LoRA, full prompt (14B)",
    "qwen7b-lean-sv3": "lookup LoRA, lean prompt (7B)", "qwen14b-lean-sv3": "lookup LoRA, lean prompt (14B)",
    "qwen7b-sv2": "search_values v2 retrain (7B, development only, not in the report)",
}


def build_weights(dest: Path, lora_dirs: list[Path]) -> list[str]:
    """Copy every trained LoRA adapter (<lora dir>/<name>/final + train_info.json); checkpoints are left out."""
    finals = {}
    for lora_dir in lora_dirs:
        for p in sorted(lora_dir.glob("*/final")) if lora_dir.is_dir() else []:
            if p.is_dir():
                finals.setdefault(p.parent.name, p)  # first --lora-dir wins on a name clash
    finals = [finals[k] for k in sorted(finals)]
    if not finals:
        return [f"no LoRA adapters found in {', '.join(map(str, lora_dirs))} (expected <name>/final from "
                "train_lora.py): build the ZIP on the training PC or pass --lora-dir."]
    rows, total = [], 0
    for final in finals:
        name = final.parent.name
        out = dest / "lora" / name
        shutil.copytree(final, out)
        if (final.parent / "train_info.json").exists():
            copy(final.parent / "train_info.json", out / "train_info.json")
        size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
        total += size
        rows.append(f"| `lora/{name}/` | {ADAPTER_NAMES.get(name, 'see code/docs/README_technical.md')} | {size / 1e6:.0f} MB |")
        log(f"models/lora/{name}/: {size / 1e6:.0f} MB")
    base = "Qwen/Qwen2.5-Coder-14B-Instruct or Qwen/Qwen2.5-Coder-7B-Instruct (matching the adapter's size)"
    (dest / "README.md").write_text(
        "# Trained LoRA adapters\n\n"
        "PEFT adapters (rank 16, all attention and MLP projections) written by `code/scripts/train_lora.py`, one per\n"
        f"system in the report. Each folder holds the adapter, the tokenizer and `train_info.json` (settings + loss log).\n"
        "Adapters trained with `--train-tokens` also hold the learned `<tool_call>` token rows used by\n"
        "`export_ollama.py`. The base models are not included (15 to 30 GB, Apache 2.0, on Hugging Face).\n\n"
        "| Folder | System | Size |\n|---|---|---|\n" + "\n".join(rows) + "\n\n"
        "Use with Hugging Face PEFT:\n\n```python\n"
        "from transformers import AutoModelForCausalLM, AutoTokenizer\nfrom peft import PeftModel\n"
        "base = AutoModelForCausalLM.from_pretrained(\"Qwen/Qwen2.5-Coder-7B-Instruct\", device_map=\"auto\")\n"
        "model = PeftModel.from_pretrained(base, \"models/lora/qwen7b\")\n"
        "tok = AutoTokenizer.from_pretrained(\"models/lora/qwen7b\")\n```\n\n"
        "Or build the Ollama model the experiments used (merge, GGUF Q4_K_M):\n\n"
        "```\npython code/scripts/export_ollama.py --base Qwen/Qwen2.5-Coder-7B-Instruct --adapter models/lora/qwen7b \\\n"
        "    --llama-cpp ../llama.cpp --quantize ../llama.cpp/build/bin/llama-quantize --ollama-base qwen2.5-coder:7b --name qwen7b-cube-lora\n```\n"
        f"\nBase model for each adapter: {base}.\n", encoding="utf-8")
    log(f"models/: {len(finals)} adapters, {total / 1e9:.2f} GB")
    return []


# ---------------------------------------------------------------- checks

def run_notebooks(stage: Path) -> list[str]:
    warnings = []
    nb_dir = stage / "code" / "notebook"
    for nb in ("CE893_walkthrough.ipynb", "CE893_evaluation.ipynb"):
        cmd = [sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute", "--inplace",
               "--ExecutePreprocessor.timeout=900", nb]
        log(f"executing {nb} in the staged copy ...")
        r = subprocess.run(cmd, cwd=nb_dir, capture_output=True, text=True, encoding="utf-8")
        if r.returncode:
            warnings.append(f"{nb} failed to run from the staged copy:\n{r.stderr[-1500:]}")
        else:
            log(f"  {nb}: ran without errors (outputs saved in the ZIP)")
    return warnings


def verify_zip(zpath: Path, top: str) -> list[str]:
    with zipfile.ZipFile(zpath) as z:
        names = set(z.namelist())
        bad = z.testzip()
    need = ["README.md", "report.pdf", "report-source/main.tex", "report-source/references.bib",
            "code/notebook/CE893_walkthrough.ipynb", "code/notebook/CE893_evaluation.ipynb", "code/harness/agent.py",
            "data/source_and_licence.txt", "data/processed/test_questions.jsonl"]
    problems = [f"missing in ZIP: {n}" for n in need if f"{top}/{n}" not in names]
    if bad:
        problems.append(f"corrupt entry in ZIP: {bad}")
    return problems


def write_zip(zpath: Path, folder: Path, prefix: Path) -> None:
    if zpath.exists():
        zpath.unlink()
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True) as z:
        for p in sorted(folder.rglob("*")):
            if p.is_file():
                # safetensors barely compress; storing them is much faster
                kind = zipfile.ZIP_STORED if p.suffix == ".safetensors" else zipfile.ZIP_DEFLATED
                z.write(p, prefix / p.relative_to(folder), compress_type=kind)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reg", default="DIMAA21403", help="registration number used in the folder and ZIP name")
    ap.add_argument("--report", choices=sorted(REPORT_VERSIONS), default="outline",
                    help="outline = main_outline.pdf (Shamir's own text, default); full = main.pdf (full draft)")
    ap.add_argument("--bird-dir", type=Path, default=ROOT / "data" / "bird",
                    help="unpacked BIRD dev folder (has dev.json, dev_tables.json, dev_databases/)")
    ap.add_argument("--download-bird", action="store_true", help="run scripts/download_bird.py if --bird-dir is missing")
    ap.add_argument("--no-sft", action="store_true", help="skip generating the LoRA training/validation files")
    ap.add_argument("--no-weights", action="store_true", help="leave the trained LoRA adapters out")
    ap.add_argument("--lora-dir", type=Path, action="append",
                    help="folder with <name>/final adapters; repeat for several (default: lora/ and, if it exists, "
                         "../msc-ce893-schema/lora/, where the lean and noschema adapters were trained)")
    ap.add_argument("--weights-zip", action="store_true",
                    help="put the adapters in a second ZIP (<REG>-project-weights.zip) instead of the main one")
    ap.add_argument("--run-notebooks", action="store_true", help="execute both notebooks inside the staged copy")
    ap.add_argument("--out", type=Path, default=ROOT / "dist")
    args = ap.parse_args()
    if not args.lora_dir:
        args.lora_dir = [d for d in (ROOT / "lora", ROOT.parent / "msc-ce893-schema" / "lora") if d.is_dir()] or [ROOT / "lora"]

    top = f"{args.reg}-project"
    stage = args.out / top
    zpath = args.out / f"{top}.zip"
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)

    files = tracked_files()
    commit = git("rev-parse", "--short", "HEAD").strip()
    warnings = []
    dirty = [l for l in git("status", "--porcelain").splitlines() if l and not l[3:].startswith("dist/")]
    if dirty:
        warnings.append(f"{len(dirty)} uncommitted change(s) in the working tree are included as they are on disk "
                        f"(commit them so the ZIP matches GitHub {commit}).")

    bird = args.bird_dir if (args.bird_dir / "dev.json").exists() else None
    if bird is None and args.download_bird:
        subprocess.run([sys.executable, str(ROOT / "scripts" / "download_bird.py")], cwd=ROOT, check=True)
        bird = ROOT / "data" / "bird" if (ROOT / "data" / "bird" / "dev.json").exists() else None

    warnings += build_report(stage, args.report, files)
    build_code(stage, files)
    warnings += build_data(stage, bird, not args.no_sft)
    if args.no_weights:
        warnings.append("--no-weights: the trained LoRA adapters are not included.")
    elif args.weights_zip:
        wstage = args.out / f"{top}-weights"
        if wstage.exists():
            shutil.rmtree(wstage)
        warnings += build_weights(wstage / f"{top}" / "models", args.lora_dir)
    else:
        warnings += build_weights(stage / "models", args.lora_dir)

    readme = (ROOT / "submission" / "README.md").read_text(encoding="utf-8")
    readme = readme.replace("{REG}", args.reg).replace("{COMMIT}", commit).replace(
        "{REPORT}", "outline version (`main_outline.tex`, renamed `main.tex`)" if args.report == "outline"
        else "full version (`main.tex`)").replace(
        "{DATE}", datetime.now(timezone.utc).strftime("%Y-%m-%d"))
    (stage / "README.md").write_text(readme, encoding="utf-8")

    if args.run_notebooks:
        warnings += run_notebooks(stage)

    write_zip(zpath, stage, Path(top))
    warnings += verify_zip(zpath, top)
    if args.weights_zip and not args.no_weights and (args.out / f"{top}-weights").exists():
        wzip = args.out / f"{top}-weights.zip"
        write_zip(wzip, args.out / f"{top}-weights", Path())
        log(f"Wrote {wzip} ({wzip.stat().st_size / 1e6:.1f} MB): unzip it next to the main ZIP, it adds {top}/models/")

    size = zpath.stat().st_size / 1e6
    log(f"\nWrote {zpath} ({size:.1f} MB), staged copy in {stage}")
    if warnings:
        log("\nCHECK BEFORE UPLOADING:")
        for w in warnings:
            log(f"  - {w}")
    else:
        log("No warnings.")


if __name__ == "__main__":
    main()
