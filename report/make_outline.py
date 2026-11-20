"""Builds report_outline.tex from report.tex: same structure, figures and tables, but every
prose paragraph is replaced by bullet notes on what to write (for writing the text yourself).

    python report/make_outline.py && latexmk -xelatex main_outline.tex
"""
import re
from pathlib import Path

HERE = Path(__file__).parent

NOTES = {
    "TITLE": [
        "Summary (3 sentences): task (questions to Cube queries with an ADK agent), intervention (LoRA, harness fixed), headline numbers (7B 27.7 to 46.4, 14B 28.9 to 49.7 strict) and the prompt-offloading result (lean prompt, 2.4x fewer tokens per call).",
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
        "Lead with Table 5: +18.7 points (7B, 78 fixed vs 16 broken) and +20.8 points (14B, 77 vs 8).",
        "7B LoRA beats 14B baseline by 17.5 points; 14B LoRA vs 7B LoRA not significant (p=0.12).",
        "14B baseline relaxed >> strict (extra columns); adapter mostly removes this.",
        "Training curves (Fig. 6): converged within half an epoch, val loss 1.00 to 0.27 (7B), no overfitting.",
        "Fig. 8: lean = 1.5k tokens per call vs 3.6-3.9k; loses 2.1 points at 14B (n.s.) and 4.5 at 7B (p=0.008). No schema: 38.3% (invented members). Full LoRA given lean prompt without retraining: 0%.",
    ],
    "Without the hint": [
        "Table 6: hiding the hint costs the adapters -9.6 (7B) and -9.0 (14B) points strict vs -6.3 and -1.5 (n.s.) for the baselines.",
        "Why: adapters were trained with hints and never call search_values, so they guess values; empty answers 7.5% to 13.0% (7B). Baselines look up more (1.6 to 2.0 per q at 7B).",
        "LoRA still leads by 15.3 (7B) and 13.3 (14B) points, both p<1e-6.",
        "Tool use: 7B baseline has 0.8-1.1 rejected calls per q, adapters about 0.1; 14B models write most calls as text (71% / 100%).",
    ],
    "Resource use": [
        "Training time 5.3 h (7B), 9.0 h (14B), lean 2-3 h; one 16 GB GPU.",
        "Steps 4.0 to 2.1, prompt tokens per question halve, model time 3.2 s to 1.3 s (7B).",
        "Latency columns in Table 5: p90 5.9 s to 1.7 s (7B); output speed unchanged (~110 tok/s 7B, 60-65 14B), so the saving is fewer and shorter calls.",
        "Time to first token not measured: future work (would separate prompt processing from decoding).",
        "Caveat: part of the saving is because the adapter skips search_values (0.3% vs 92%).",
    ],
    "Interpretation and error analysis": [
        "Walk through Fig. 9 case A (extra grouping dimension) and case B (guessed value, empty result) and case C (invented member without schema).",
        "Fig. 10: failed runs 38 to 5 (7B); wrong filters 27 to 15 (14B); ~50 per arm need SQL Cube cannot express (ceiling, not a model error).",
        "Explain the value-lookup weakness: only 22% of training transcripts had search_values.",
    ],
    "Limitations and responsible use": [
        "Two dbs, one run per arm: about +/-5 points 95% interval.",
        "Translator bias toward simple query shapes; noschema is database-specific.",
        "14B quantised adapter writes a stray token instead of <tool_call>; harness parses text calls.",
        "Data: public, no private personal data. Use: wrong filter gives a convincing wrong chart, so the app shows query and trace.",
    ],
    "Conclusion": [
        "Answer the RQ directly with the two headline numbers and the halved steps.",
        "Most of the prompt can move into the weights (lean within 2 points at 14B, 40% of tokens); the schema must stay for unseen dbs.",
        "No hint: adapters lose about 9 points (more than baselines) but still lead by 13-15.",
        "Main limitation: adapter stopped looking up values (worst without the hint). Next steps: retrain with search_values in every text-filter transcript and some hints removed; measure time to first token.",
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
