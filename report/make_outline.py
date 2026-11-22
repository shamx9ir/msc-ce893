"""Builds report_outline.tex from report.tex: same structure, figures and tables, but every
prose paragraph is replaced by bullet notes on what to write (for writing the text yourself).

    python report/make_outline.py && latexmk -xelatex main_outline.tex
"""
import re
from pathlib import Path

HERE = Path(__file__).parent

NOTES = {
    "TITLE": [
        "Summary (3 sentences): task (questions to Cube queries with an ADK agent), intervention (LoRA, harness fixed), headline numbers (7B 21.4 to 36.7, 14B 27.4 to 40.7 strict) and the prompt-offloading result (lean prompt matches it with 2.4x fewer tokens per call).",
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
        "Baseline = prompt-engineering harness: guide, rules, 4 examples, schema with question-linked fields, 3 tools (Fig. 2).",
        "Proposed = same harness + LoRA trained on transcripts with exactly this prompt and these tools; also adapters with shorter prompts (lean, no schema) to test how much of the prompt the adapter absorbs.",
        "Why fine-tuning fits: output is strict JSON with exact names; baseline errors were format/protocol errors.",
    ],
    "Expected improvement": [
        "Baseline errors on dev data: dimensions in measures, invented members, malformed tool calls.",
        "Expectations: (1) higher accuracy, (2) fewer steps, (3) guide and tool definitions can be dropped once learned.",
    ],
    "Source and preparation": [
        "BIRD fields: question, gold SQL, SQLite db. Copied to DuckDB; Cube models generated from schema, foreign keys and column descriptions.",
        "Rule-based SQL to Cube translator; keep a pair only if Cube rows = gold rows and non-empty (Fig. 3a funnel: 8,003 to 3,531).",
        "Chat transcript format, loss on assistant turns only; 769 transcripts include a search_values call.",
    ],
    "Data quality and limitations": [
        "Translator: no subqueries, CASE, string slicing; 30% of test needs such SQL (ceiling 231/332).",
    ],
    "Base model and baseline": [
        "Qwen2.5-Coder 7B/14B Instruct, Ollama Q4_K_M, temperature 0, 16k context.",
        "Harness versions v1 to v4.2 developed on tuning dbs only (Fig. 4); list what each version added; v4.2 frozen before any LoRA run.",
    ],
    "Chosen method": [
        "QLoRA with PEFT: 4-bit NF4 base, rank-16 adapters on all attention + MLP projections (Fig. 5).",
        "Export: merge, GGUF, Q4_K_M, base chat template.",
        "Prompt profiles: lean and noschema (needs ~1,300 synthetic questions on the test schemas).",
    ],
    "Implementation details": [
        "One sentence introducing Table 3.",
    ],
    "Compared systems": [
        "Baseline: base model + harness v4.2 (full prompt).",
        "Proposed: same model file + LoRA, same harness and prompt.",
        "Prompt-offloading arms: lean LoRA; noschema LoRA with no schema / with lean prompt; control = full LoRA given the lean prompt without retraining.",
        "Extra arm: LoRA retrained to look up filter values (Section 6.2).",
    ],
    "Evaluation data and measures": [
        "332 test questions, 2 unseen dbs, difficulty mix 218/91/23, 77% need a join. Introduce Table 4.",
    ],
    "Experimental procedure": [
        "Order: freeze test split, tune harness, freeze v4.2, train each adapter once, run each arm once.",
    ],
    "Main findings": [
        "Lead with Table 5: +15.4 points (7B, 70 fixed vs 19 broken) and +13.3 points (14B, 58 vs 14).",
        "7B LoRA beats 14B baseline by 9.3 points (p=0.0009); 14B LoRA vs 7B LoRA not significant (p=0.09).",
        "Fig. 8: lean = 1.5k vs 3.6k tokens per call, loses 2.1 / 0.6 points (n.s.). No schema: 18.1 / 24.7%; noschema + short schema 37.0 / 41.0. Full LoRA on lean prompt: 0%.",
    ],
    "Tool use": [
        "Table 6: baseline ~2 lookups, ~1 rejected call per q; adapters draft once, never look values up (13% empty answers).", "Extra arm (retrained for lookups; untrained <tool_call> tokens fixed): 7B 41.9% (+5.1, p=0.019), 14B 44.9% (+4.2, p=0.020); lookups ~70%, empty answers 4-5%.",
    ],
    "Resource use": [
        "Training 5.3 h (7B), 9.0 h (14B) on one 16 GB GPU (Fig. 6). Steps 4.1 to 2.1, tokens halve, time 3.5 s to 1.3 s (7B).",
        "Output speed unchanged, so the saving is fewer, shorter calls (partly from skipping search_values).",
    ],
    "Interpretation and error analysis": [
        "Walk through Fig. 9: A extra grouping, B guessed value, C invented member.",
        "Fig. 10: failed runs 45 to 4 (7B); wrong filters 64 to 50; ~55 per arm need SQL Cube cannot express.",
    ],
    "Limitations and responsible use": [
        "Two dbs, one run per arm (about +/-5 points). Setting differs from BIRD's standard one: not comparable with its leaderboard.",
        "Stray token instead of <tool_call> (untrained); retrain fixes it. Public data; app shows the query.",
    ],
    "Conclusion": [
        "Answer the RQ with the headline numbers; most of the prompt can move into the weights, the schema must stay.",
        "Retrain: 41.9 / 44.9%. Future: retrain lean adapters; tool tokens for baseline; newer open-weight families (not only Qwen); more dbs; time to first token.",
    ],
}


# Shamir's research question, used in the outline's RQ box.
OWN_RQ = r"\resultplaceholder{With the agent harness held constant, to what extent can training a LoRA adapter on verified tool-use transcripts boost execution accuracy on unfamiliar databases? How much of the prompt, comprising the query guide, tool definitions, and database schema, can be offloaded into the LoRA adapter without compromising performance in terms of accuracy, saved tokens, and generation speed?}"


# Text Shamir has written for the outline version: used as-is instead of the notes for that section.
OWN_TEXT = {
    "Motivation and context": r"""As open-weights models continue to gain capabilities, local Large Language Models (LLMs) are set to become more and more significant. Due to their smaller footprint, open-weights models are highly accessible for local fine-tuning, offering greater customisation potential despite being less powerful than large-scale foundation models.

Current industry practice is to build a harness around the model. LLM harness is the surrounding support components to turn a model to a reliable, goal oriented agent. Typically we would customise a local LLM to a specific use case with a specific workflow and deploy it within a constrained hardware in the most efficient way.

To demonstrate this, we focus on a common business scenario involving data analysis on domain-specific datasets. Analysts submit questions to the LLM, which then query the underlying data source. Our implementation leverages a semantic layer powered by Cube~\cite{cube} to serve as the backend language in contrast to direct SQL. To guide the LLM in answering data-related queries, we will construct a prompt equipped with tools.
""",
    "Research question and contribution": r"""A key feature of this experiment is that the evaluation harness remains strictly fixed while only model weights are varied. Every test arm utilises the identical 4-bit model file hosted in Ollama, alongside identical prompts and available tools, ensuring that any performance variations stem directly from the LoRA adapter. Departing from traditional text-to-SQL paradigms, the model generates queries targeted at a semantic-layer API. Furthermore, we evaluate the extent to which the adapter can absorb components of the harness prompt, including the query guide, tool definitions, and schema details. All assessments are conducted across two unseen databases using realistic, analyst-style questions.
""",
    "Problem definition": r"""As outlined in Table~\ref{tab:task}, the objective is to take an analyst's natural-language query regarding an unfamiliar database and translate it into a single Cube query yielding output rows identical to BIRD's gold SQL reference, all executed using a local model on a single 16\,GB GPU.
""",
    "Proposed solution": r"""The baseline architecture relies on a prompt-engineering framework consisting of a system prompt with a Cube query guide, operational rules, four worked examples, a question-linked database schema, and three dedicated tools (Fig.~\ref{fig:pipeline}). The proposed approach integrates a LoRA adapter fine-tuned on verified interaction transcripts using the identical prompt and toolset. Fine-tuning is particularly effective here because the task requires producing strict JSON with precise field names, whereas baseline failures primarily stem from formatting and protocol errors rather than missing domain knowledge. By training the model to internalise both the query language and tool protocol directly from examples, portions of the prompt can be systematically removed. To evaluate the adapter's capacity to absorb these instructions, we also train variants using abbreviated prompts: a ``lean'' configuration that omits the query guide and tool definitions, and a ``no schema'' setup that provides only the database name.
""",
    "Expected improvement": r"""On the tuning databases, most baseline failures were not missing knowledge but protocol errors: dimensions placed in ``measures'', invented member names, malformed tool calls, and extra grouping columns. Because the adapter is trained on transcripts in which every query was executed and checked, we expect four improvements. (1) Higher execution accuracy on the unseen test databases, mainly from fewer format and naming errors; we expect the largest gains on questions that need joins, where the schema is hardest to follow from the prompt alone. (2) Fewer steps and tokens per question, because the adapter should produce a valid query in fewer attempts, needing fewer rejected calls and retries. (3) Lower latency per question as a result of fewer and shorter model calls; output speed itself should not change, since the merged adapter adds no inference cost. (4) A shorter prompt without loss of accuracy: once the query guide and tool definitions have been learned, they should be removable, but the database schema will likely still be needed on unseen databases.
""",
    "Source and preparation": r"""Each question in the BIRD dataset includes an English prompt, a corresponding SQLite database, and a gold SQL query. After migrating every database into DuckDB, we automatically generated Cube models based on their schema, foreign key relationships, and column descriptions provided by BIRD. To construct our training set, a rule-based translation engine converted each gold SQL query into a Cube equivalent. We retained only the query pairs whose execution outputs matched identically without producing empty results (Fig.~\ref{fig:data}a). These validated pairs were then compiled into full conversation transcripts featuring the exact system prompt of the harness, the initial user prompt, and sequential tool executions: \texttt{run\_query} followed by \texttt{submit\_answer}. Additionally, 769 of these transcripts incorporated an initial \texttt{search\_values} call to retrieve filtered text values. Hints from BIRD were preserved in the user messages during training, whereas they were omitted during evaluation. Finally, loss computation was restricted strictly to the assistant's turns.
""",
    "Data quality and limitations": r"""The translator is restricted to SQL constructs supported by Cube, meaning the training corpus excludes subqueries, CASE expressions, arithmetic, and string slicing. Consequently, the 30\% of test questions requiring these features fall outside the capability of any Cube query, placing an upper performance limit of 231/332 on the test set for this framework. Additionally, discrepancies and tied results exist within the BIRD benchmark's reference SQL. Evaluation relies on matching the expected output rows, allowing any functionally accurate Cube query to be marked correct. Finally, the ten largest databases in the training set (exceeding 600\,MB) were excluded.
""",
    "Base model and baseline": r"""Both experimental setups employ the Qwen2.5-Coder-7B-Instruct and -14B-Instruct architectures, served as Q4\_K\_M GGUF format through Ollama with zero temperature and a context window of 16,384 tokens. Serving as the baseline, the agent harness was refined across five iterations using strictly the tuning databases (Fig.~\ref{fig:versions}). Version 2 introduced four worked examples alongside rules targeted at common error patterns; versions 3 through 4.2 integrated numerical and date field value lookups, schema linking (wherein matching field names are provided in the user prompt), repeated-call prevention, and pre-submission column verification. Version 4.2 was finalised and frozen prior to initiating any LoRA fine-tuning runs.
""",
    "Chosen method": r"""Fine-tuning is conducted via QLoRA~\cite{dettmers2023qlora} utilising PEFT~\cite{peft}, where rank-16 adapters are attached to all attention and MLP layers while the underlying base model remains frozen in 4-bit NF4 (Fig.~\ref{fig:method}). Upon completion of training, each adapter is merged with the base model, converted into GGUF format via llama.cpp~\cite{llamacpp}, and quantised to Q4\_K\_M to match the base setup. By preserving the base model's original chat template, the evaluation strictly isolates the influence of the adapter. To evaluate how effectively model weights can internalise prompt instructions, two additional adapter variants were trained per model size using reduced prompts: the lean variant strips out the query guide, worked examples, and tool JSON schemas, while offering an abbreviated schema; the noschema variant omits schema definitions entirely. Because operating without a schema requires prior exposure to target structures, the noschema training set was augmented with roughly 1,300 synthetic template questions covering the two evaluation databases, explicitly avoiding any BIRD dev questions or SQL.
""",
    "Related work": r"""The evaluation of text-to-SQL tasks has progressed from Spider~\cite{yu2018spider} to BIRD~\cite{li2023bird}, a benchmark that incorporates substantial real-world databases, dirty values, and question-level ``evidence'' annotations to assess execution accuracy. In prompt-driven approaches, frameworks like DIN-SQL~\cite{pourreza2023din} implement task decomposition and self-correction, whereas ReAct~\cite{yao2023react} style agent loops enable models to evaluate queries dynamically prior to final response generation, an approach adopted by our harness. For open-weights architectures, specialised fine-tuned models such as CodeS~\cite{li2024codes} achieve competitive performance on BIRD even at smaller parameter scales, while Gorilla~\cite{patil2023gorilla} demonstrates that targeted fine-tuning outperforms pure prompting for accurate API invocation. Parameter-efficient methods like LoRA~\cite{hu2021lora} and QLoRA~\cite{dettmers2023qlora} facilitate this fine-tuning process on single consumer-grade GPUs. Lastly, context distillation~\cite{askell2021general,snell2022distilling} techniques enable models to emulate the behaviour of extensive system prompts, providing the foundational concept behind moving the harness prompt into the LoRA weights.
""",
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
        if in_env == 0 and current and s and (not s.startswith(("\\", "%", "{", "}")) or s.startswith("\\textbf{")) or (
                in_env == 0 and current and s.startswith("\\resultplaceholder")):
            # prose paragraph (or RQ box kept) -> replace with notes once per subsection
            if s.startswith("\\resultplaceholder"):
                out.append(OWN_RQ + "\n" if "textbf{RQ.}" in s else line)
                continue
            if current in OWN_TEXT and current not in emitted:
                out.append(OWN_TEXT[current] + "\n")
                emitted.add(current)
            elif current in NOTES and current not in emitted:
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
    # The outline's notes run longer than the prose in places, so a few figures are smaller here to keep 10 pages.
    for fig, (w_old, w_new) in {"fig01_glance": ("0.7", "0.62"), "fig03_data": ("0.7", "0.64"), "fig05_method": ("0.7", "0.6"), "fig04_harness_versions": ("0.7", "0.62"), "fig06_training": ("0.7", "0.62"), "fig07_main": ("0.62", "0.56"), "fig08_offload": ("0.68", "0.6"), "fig09_outputs": ("0.6", "0.7"), "fig10_errors": ("0.52", "0.44")}.items():
        text = text.replace(f"width={w_old}\\linewidth]{{figures/{fig}", f"width={w_new}\\linewidth]{{figures/{fig}")
    # Shamir: Fig. 9 in the outline uses the wider two-column layout (shorter traces, larger text).
    text = text.replace("{figures/fig09_outputs.pdf}", "{figures/fig09_outputs_wide.pdf}", 1)
    # Shamir: Fig. 9 caption beside the figure (outline only), to avoid white space at the sides.
    text = re.sub(r"\\centering\n\\includegraphics\[width=[0-9.]+\\linewidth\]\{figures/fig09_outputs_wide\.pdf\}\n(\\caption\{.*?\})\n(\\label\{fig:outputs\})",
                  lambda m: "\\begin{minipage}[c]{0.68\\linewidth}\\includegraphics[width=\\linewidth]{figures/fig09_outputs_wide.pdf}\\end{minipage}\\hfill\n"
                            "\\begin{minipage}[c]{0.3\\linewidth}\\small\\raggedright " + m.group(1) + m.group(2) + "\\end{minipage}", text, count=1, flags=re.S)
    # Shamir: explain why the funnel's 59 databases become 58 in the splits (outline only).
    text = text.replace("790 gave different or empty results.", "790 gave different or empty results; one database (craftbeer) yielded no verified pairs, so the splits cover 58.", 1)
    # Outline only: Fig. 7 placed before 5.3 so it can fill the end of page 7 (keeps 10 pages).
    m5 = re.search(r"\\begin\{figure\}\[!htbp\]\n(?:(?!\\end\{figure\}).)*fig07_main.*?\\end\{figure\}\n\n?", text, flags=re.S)
    if m5:
        text = text.replace(m5.group(0), "", 1).replace("\\subsection{Experimental procedure}", m5.group(0) + "\\subsection{Experimental procedure}", 1)
    # Shamir: in the outline, Fig. 2 sits before the Proposed solution subsection.
    m = re.search(r"\\begin\{figure\}\[!htbp\]\n(?:(?!\\end\{figure\}).)*fig02_pipeline.*?\\end\{figure\}\n\n?", text, flags=re.S)
    if m:
        fig = m.group(0)
        text = text.replace(fig, "", 1).replace("\\subsection{Proposed solution}", fig + "\\subsection{Proposed solution}", 1)
    text = text.replace("% Full draft.", "% OUTLINE version, generated by make_outline.py from report.tex.")
    # sections whose only prose came after the floats still need notes
    for sec, own in OWN_TEXT.items():
        if sec not in emitted:
            text = re.sub(r"(\\(?:sub)?section\{" + re.escape(sec) + r"\}[^\n]*\n)", lambda m: m.group(1) + own + "\n", text, count=1)
            emitted.add(sec)
    for sec, items in NOTES.items():
        if sec != "TITLE" and sec not in emitted:
            text = re.sub(r"(\\(?:sub)?section\{" + re.escape(sec) + r"\}[^\n]*\n)", lambda m: m.group(1) + bullets(items), text, count=1)
    (HERE / "report_outline.tex").write_text(text)
    (HERE / "main_outline.tex").write_text((HERE / "main.tex").read_text().replace("\\input{report}", "\\input{report_outline}"))


if __name__ == "__main__":
    main()
