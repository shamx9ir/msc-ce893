"""Builds report_outline.tex from report.tex: same structure, figures and tables, but every
prose paragraph is replaced by bullet notes on what to write (for writing the text yourself).

    python report/make_outline.py && latexmk -xelatex main_outline.tex
"""
import re
from pathlib import Path

HERE = Path(__file__).parent

NOTES = {
    "TITLE": [
        "Summary (3 sentences): task (questions to Cube queries with an ADK agent), intervention (LoRA, harness fixed), headline numbers without BIRD's hint (7B 21.4 to 36.7, 14B 27.4 to 40.7 strict) and the prompt-offloading result (lean prompt matches it with 2.4x fewer tokens per call).",
    ],
    "Motivation and context": [
        "Who asks: analysts/managers who want a number or a chart, not SQL.",
        "Define semantic layer, cube, dimension, measure (Cube). Why Cube instead of SQL: smaller language, joins handled, rows go straight to a chart.",
        "What is hard: exact member names, dimension vs measure slot, exact stored values, tool order.",
        "The cost of prompting: guide, rules, examples and schema are re-sent on every model call.",
    ],
    "Related work": [
        "Benchmarks: Spider, then BIRD (real databases, evidence hint, execution accuracy).",
        "Prompt-only systems (DIN-SQL) and agent loops (ReAct): our harness is in this family.",
        "Fine-tuned text-to-SQL (CodeS) and API-call fine-tuning (Gorilla).",
        "LoRA and QLoRA make it fit one 16 GB GPU.",
        "Context distillation (Askell et al., Snell et al.): the idea behind moving the prompt into the weights.",
    ],
    "Research question and contribution": [
        "Contribution list: (i) Cube layer + ADK agent for BIRD, (ii) 3,531 execution-verified Cube transcripts from BIRD train, (iii) LoRA adapters for 7B and 14B under three prompt profiles, (iv) demo app (A2UI + Highcharts).",
        "Stress fairness: same Q4_K_M model file in Ollama, arms differ only in adapter and prompt.",
    ],
    "Problem definition": [
        "One sentence introducing Table 1.",
    ],
    "Proposed solution": [
        "Baseline = prompt-engineering harness: guide, rules, 4 examples, schema, 3 tools, pre-checks with 'did you mean' hints (Fig. 2).",
        "Proposed = same harness + LoRA trained on transcripts with exactly this prompt and these tools.",
        "Why fine-tuning fits: output is strict JSON with exact names; baseline errors were format/protocol errors.",
    ],
    "Expected improvement": [
        "Baseline errors on dev data: dimensions in measures, invented members, malformed tool calls.",
        "Expectations: (1) higher accuracy, (2) fewer steps, (3) guide and tool definitions can be dropped once learned.",
    ],
    "Source and preparation": [
        "BIRD fields: question, evidence, gold SQL, SQLite db. Copied to DuckDB; Cube models generated from schema, foreign keys and column descriptions.",
        "Rule-based SQL to Cube translator; keep a pair only if Cube rows = gold rows and non-empty (Fig. 3a funnel: 8,003 to 3,531).",
        "Chat transcript format, loss on assistant turns only; 769 transcripts include a search_values call.",
    ],
    "Data quality and limitations": [
        "Translator coverage: no subqueries, CASE arithmetic, string slicing; 30% of test needs such SQL (ceiling 231/332).",
        "BIRD gold SQL has some errors and ties; scoring against gold rows means any correct query counts.",
        "Ten largest training dbs skipped (over 600 MB).",
    ],
    "Base model and baseline": [
        "Qwen2.5-Coder 7B/14B Instruct, Ollama Q4_K_M, temperature 0, 16k context.",
        "Harness versions v1 to v4.2 developed on tuning dbs only (Fig. 4); list what each version added; v4.2 frozen before any LoRA run.",
    ],
    "Chosen method": [
        "QLoRA with PEFT: 4-bit NF4 base, rank-16 adapters on all attention + MLP projections (Fig. 5).",
        "Export: merge, GGUF, Q4_K_M, base chat template, so only the adapter differs.",
        "Prompt profiles: lean (no guide, no tool schemas, compact schema) and noschema (needs ~1,300 synthetic questions on the test schemas; no BIRD dev data used).",
        "Fig. 6 (training curves) can be described here or in Results.",
    ],
    "Implementation details": [
        "One sentence introducing Table 3.",
    ],
    "Compared systems": [
        "Baseline: base model + harness v4.2 (full prompt).",
        "Proposed: same model file + LoRA, same harness and prompt.",
        "No-hint runs: baseline and LoRA at both sizes with the hint hidden (adapters were trained with hints).",
        "Prompt-offloading arms: lean LoRA; noschema LoRA with no schema / with lean prompt; control = full LoRA given the lean prompt without retraining.",
    ],
    "Evaluation data and measures": [
        "332 test questions, 2 unseen dbs, difficulty mix 218/91/23, 77% need a join. Introduce Table 4.",
    ],
    "Experimental procedure": [
        "Order: freeze test split + scoring, tune harness on tuning dbs, freeze v4.2, train each adapter once (seed 7), run each arm once.",
        "List what is identical across arms within a size.",
    ],
    "Main findings": [
        "All test runs without BIRD's hint. Lead with Table 5: +15.4 points (7B, 70 fixed vs 19 broken) and +13.3 points (14B, 58 vs 14).",
        "7B LoRA beats 14B baseline by 9.3 points (p=0.0009); 14B LoRA vs 7B LoRA not significant (p=0.09).",
        "14B baseline relaxed >> strict (35.8 vs 27.4, extra columns); adapter mostly removes this.",
        "Fig. 8: lean = 1.5k tokens per call vs 3.6k; loses 2.1 (7B, p=0.30) and 0.6 points (14B, p=0.83), n.s. No schema: 18.1% / 24.7% (invented members, 114/332 unknown-member errors at 14B); noschema + short schema 37.0 / 41.0. Full LoRA given lean prompt without retraining: 0%.",
    ],
    "Tool use": [
        "Table 6: 7B baseline ~2 lookups, ~1 rejected call per q; adapters draft once and submit. But they never look values up: 13% of answers from an empty draft vs 2-10%. 14B models write most calls as text.",
    ],
    "Resource use": [
        "Training time 5.3 h (7B), 9.0 h (14B), lean 2-3 h; one 16 GB GPU; converged within half an epoch (Fig. 6).",
        "Steps 4.1 to 2.1, prompt tokens per question halve, model time 3.5 s to 1.3 s (7B), p90 6.8 s to 1.7 s.",
        "Output speed unchanged (~105-120 tok/s 7B, 60-67 14B), so the saving is fewer and shorter calls.",
        "Time to first token not measured: future work (would separate prompt processing from decoding).",
        "Caveat: part of the saving is because the adapter skips search_values (1% vs 92%).",
    ],
    "Interpretation and error analysis": [
        "Walk through Fig. 9 case A (extra grouping dimension) and case B (guessed value, empty result) and case C (invented member without schema).",
        "Fig. 10: failed runs 45 to 4 (7B); wrong filters 64 to 50 (7B), 48 to 39 (14B); ~55 per arm need SQL Cube cannot express (ceiling, not a model error).",
        "Explain the value-lookup weakness: only 22% of training transcripts had search_values.",
    ],
    "Limitations and responsible use": [
        "Two dbs, one run per arm: about +/-5 points 95% interval. With BIRD's hint (standard setting) all arms score higher (LoRA 46.4 / 49.7), so not comparable with leaderboards.",
        "Translator bias toward simple query shapes; noschema is database-specific.",
        "Adapters often write a stray token instead of <tool_call> (all 14B calls; also without quantisation); harness parses text calls.",
        "Data: public, no private personal data. Use: wrong filter gives a convincing wrong chart, so the app shows query and trace.",
    ],
    "Conclusion": [
        "Answer the RQ directly with the two headline numbers and the halved steps.",
        "Most of the prompt can move into the weights (lean within ~2 points at both sizes, n.s., 40% of tokens); the schema must stay for unseen dbs.",
        "Main limitation: adapter stopped looking up values.",
        "Future work: search_values retrain (7B) changed behaviour, not accuracy (36.1 vs 36.7, p=0.88): stray first token -> text fallback, malformed calls. Fix the first-token labels or accept text + JSON; retest; measure time to first token.",
    ],
}


def esc(t):
    return (t.replace("\\", "/").replace("_", "\\_").replace("%", "\\%").replace("<", "\\textless{}")
             .replace(">", "\\textgreater{}").replace("~", "\\textasciitilde{}").replace("+/-", "$\\pm$"))


def bullets(items):
    body = "\n".join(f"  \\item {esc(i)}" for i in items)
    return ("\\resultplaceholder{\\textbf{Write here:}\\begin{itemize}[leftmargin=1.2em,itemsep=0.05em]\n"
            + body + "\n\\end{itemize}}\n")


def main():
    src = (HERE / "report.tex").read_text()
    out, i = [], 0
    lines = src.splitlines(keepends=True)
    current, in_env, emitted = None, 0, set()
    for line in lines:
        s = line.strip()
        if re.match(r"\\begin\{(figure|table|tabularx|tabular|enumerate|minipage|center)\}", s):
            in_env += 1
        m = re.match(r"\\(sub)?section\{([^}]*)\}", s)
        if m:
            current = m.group(2)
            out.append(line)
            continue
        if in_env == 0 and current and s and not s.startswith(("\\", "%", "{", "}")) or (
                in_env == 0 and current and s.startswith("\\resultplaceholder")):
            # prose paragraph (or RQ box kept) -> replace with notes once per subsection
            if s.startswith("\\resultplaceholder"):
                out.append(line)
                continue
            if current in NOTES and current not in emitted:
                out.append(bullets(NOTES[current]))
                emitted.add(current)
            continue
        if re.match(r"\\end\{(figure|table|tabularx|tabular|enumerate|minipage|center)\}", s):
            in_env -= 1
        out.append(line)
    text = "".join(out)
    text = re.sub(r"\\begin\{enumerate\}.*?\\end\{enumerate\}\n", "", text, flags=re.S)
    # title page summary -> notes
    text = re.sub(r"\\textbf\{Summary\.\}.*?\n", "\\\\textbf{Summary (to write).} " + esc(NOTES["TITLE"][0]).replace("\\", "\\\\") + "\n", text, count=1)
    text = text.replace("{\\footnotesize\n\\bibliographystyle", "{\\footnotesize\n\\nocite{*}\n\\bibliographystyle")
    text = text.replace("% Full draft.", "% OUTLINE version, generated by make_outline.py from report.tex.")
    # sections whose only prose came after the floats still need notes
    for sec, items in NOTES.items():
        if sec != "TITLE" and sec not in emitted:
            text = re.sub(r"(\\(?:sub)?section\{" + re.escape(sec) + r"\}[^\n]*\n)", lambda m: m.group(1) + bullets(items), text, count=1)
    (HERE / "report_outline.tex").write_text(text)
    (HERE / "main_outline.tex").write_text((HERE / "main.tex").read_text().replace("\\input{report}", "\\input{report_outline}"))


if __name__ == "__main__":
    main()
