"""Draws every report figure into report/figures/ from the stored runs in results/.

    python report/make_figures.py

Needs matplotlib, pandas, sqlglot and scipy. Figures are drawn at their printed size
(7 in wide = the report's text width), so the font sizes below are the sizes on paper.
"""
import json
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch

import analysis as A

OUT = Path(__file__).parent / "figures"
OUT.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.size": 8, "font.family": "DejaVu Sans", "axes.spines.top": False,
    "axes.spines.right": False, "axes.titlesize": 8.5, "axes.titleweight": "bold",
    "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
    "axes.edgecolor": "#8a8a85", "axes.linewidth": 0.6, "xtick.color": "#52514e",
    "ytick.color": "#52514e", "axes.grid": False, "pdf.fonttype": 42,
})
INK, MUTED, GRID = "#1f1f1e", "#5f5e5a", "#e4e3df"
# categorical slots in fixed order (validated default palette): harness, LoRA, lean, noschema
C_H, C_L, C_LEAN, C_NS = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
C_OK, C_BAD = "#1baf7a", "#e34948"
W = 7.0  # text width in inches


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUT / f"{name}.png", dpi=200, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def box(ax, x, y, w, h, title, body="", color=C_H, fill="#eef4fb", fs=7.2, tfs=8.2, mono=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.04",
                                fc=fill, ec=color, lw=1.1))
    ax.text(x + w / 2, y + h - 0.06, title, ha="center", va="top", fontsize=tfs,
            fontweight="bold", color=INK)
    if body:
        ax.text(x + w / 2, y + h - 0.29, body, ha="center", va="top", fontsize=fs, color=MUTED,
                linespacing=1.3, family="DejaVu Sans Mono" if mono else None)


def arrow(ax, a, b, text="", color=INK, rad=0.0, dy=0.05, fs=6.8, ha="center"):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=9, lw=0.9,
                                 color=color, connectionstyle=f"arc3,rad={rad}"))
    if text:
        ax.text((a[0] + b[0]) / 2, (a[1] + b[1]) / 2 + dy, text, ha=ha, va="bottom",
                fontsize=fs, color=color)


def canvas(w, h):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.axis("off")
    return fig, ax


def bar14(ax, pos, vals, width, run, horizontal=False):
    """Bars for one run: colour = arm (blue harness, orange LoRA), hatch = 14B."""
    c = C_L if "lora" in run else C_H
    big = run.endswith("14b")
    kw = dict(color="white" if big else c, edgecolor=c, hatch="//////" if big else None, lw=0.8,
              label=A.TEST_RUNS[run][1])
    return ax.barh(pos, vals, width, **kw) if horizontal else ax.bar(pos, vals, width, **kw)


def ygrid(ax):
    ax.yaxis.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)


# ------------------------------------------------------------------ Fig 1: at a glance
def fig_glance():
    fig, ax = canvas(W, 1.75)
    box(ax, 0.02, 0.15, 1.45, 1.5, "Question",
        "\"How many Formula 1\nraces took place on\nthe circuits in Italy?\"", fs=6.8)
    box(ax, 1.75, 0.15, 1.75, 1.5, "Agent (ADK)",
        "Qwen2.5-Coder 7B/14B\nprompt + 3 Cube tools\n\nA: base model\nB: base + LoRA adapter",
        color=C_L, fill="#fdf0ea", fs=6.8)
    box(ax, 3.78, 0.15, 1.6, 1.5, "Cube query",
        '{"measures":\n ["races.count"],\n "filters": [{\n  "member":\n  "circuits.country",\n  "values":["Italy"]}]}',
        fs=6.0, mono=True)
    box(ax, 5.66, 0.15, 1.32, 1.5, "Answer", "", color=C_OK, fill="#e9f7f1")
    ax.text(6.32, 0.95, "95", ha="center", va="center", fontsize=20, fontweight="bold", color=INK)
    ax.text(6.32, 0.55, "races (chart / table\nin the demo app)", ha="center", va="center", fontsize=6.3,
            color=MUTED)
    for x0, x1 in [(1.47, 1.75), (3.5, 3.78), (5.38, 5.66)]:
        arrow(ax, (x0 + 0.01, 0.9), (x1 - 0.01, 0.9))
    save(fig, "fig01_glance")


# ------------------------------------------------------------------ Fig 2: pipeline
def fig_pipeline():
    fig, ax = canvas(W, 3.05)
    box(ax, 0.02, 1.45, 1.3, 1.25, "1 Input", "question\n+ database id", fs=6.8)
    box(ax, 1.6, 1.15, 1.95, 1.85, "2 ADK agent loop",
        "system prompt: query guide,\nCube schema, tool list\nuser msg: question,\nlinked fields\n\n"
        "LiteLLM -> Ollama\nQwen2.5-Coder Q4_K_M\n(base, or + LoRA)", color=C_L, fill="#fdf0ea", fs=6.6)
    box(ax, 3.85, 1.15, 1.45, 1.85, "3 Tools",
        "search_values\nrun_query\nsubmit_answer\n\nharness checks:\nunknown member,\ndim vs measure,\n"
        "repeat-call guard", fs=6.6)
    box(ax, 5.6, 1.45, 1.38, 1.25, "4 Cube", "JSON query\n-> SQL on\nDuckDB copy\nof BIRD db", fs=6.8)
    arrow(ax, (1.33, 2.07), (1.6, 2.07))
    arrow(ax, (3.56, 2.3), (3.85, 2.3), "call", dy=0.03)
    arrow(ax, (3.85, 1.85), (3.56, 1.85), "result", dy=-0.16, fs=6.2)
    arrow(ax, (5.31, 2.07), (5.6, 2.07))
    ax.text(2.57, 0.95, "max 8 model calls per question", ha="center", fontsize=6.4, color=MUTED,
            style="italic")
    box(ax, 2.35, 0.03, 2.3, 0.78, "5a Evaluation",
        "result rows vs BIRD gold SQL rows\non original SQLite (strict, relaxed)", color=C_OK, fill="#e9f7f1",
        fs=6.4, tfs=7.6)
    box(ax, 4.95, 0.03, 2.03, 0.78, "5b Demo app",
        "A2UI surface: Highcharts\nchart, table, query, trace", color=C_OK, fill="#e9f7f1", fs=6.4, tfs=7.6)
    arrow(ax, (4.57, 1.15), (4.0, 0.82), "submitted query", dy=-0.02, fs=6.2, ha="left")
    arrow(ax, (6.29, 1.45), (6.0, 0.82), "rows", fs=6.2, ha="left")
    save(fig, "fig02_pipeline")


# ------------------------------------------------------------------ Fig 3: data prep
def fig_data():
    fig = plt.figure(figsize=(W, 3.25))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.55], wspace=0.08)
    ax = fig.add_subplot(gs[0])
    stages = [("BIRD train questions\n(59 of 69 dbs)", 8003), ("rule translator\nproduced a query", 8003 - 2782),
              ("Cube ran it\nwithout error", 8003 - 2782 - 900), ("result = gold\n(non-empty)", 3531),
              ("training split\n(55 dbs)", 3273), ("validation split\n(3 held-out dbs)", 258)]
    y = np.arange(len(stages))[::-1]
    ax.barh(y, [s[1] for s in stages], color=["#86b6ef"] * 4 + [C_L, C_LEAN], height=0.62)
    for yi, (lab, v) in zip(y, stages):
        ax.text(v + 120, yi, f"{v:,}", va="center", fontsize=7, color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels([s[0] for s in stages], fontsize=6.8)
    ax.set_xlim(0, 9800)
    ax.set_xticks([0, 4000, 8000])
    ax.set_xlabel("questions")
    ax.set_title("a) Building the fine-tuning set", loc="left")
    ax2 = fig.add_subplot(gs[1])
    ax2.axis("off")
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.set_title("b) One record, from BIRD to a training example (car_retails db)", loc="left")
    blocks = [
        ("BIRD record", "#f3f2ee",
         "Q: Where's Foon Yue Tseng's office located at? Give the\n   detailed address.\n"
         "Hint: Detailed address comprises addressLine1 and addressLine2\n"
         "SQL: SELECT T1.addressLine1, T1.addressLine2 FROM offices T1\n"
         "     JOIN employees T2 ON T1.officeCode = T2.officeCode\n"
         "     WHERE T2.firstName = 'Foon Yue' AND T2.lastName = 'Tseng'"),
        ("translated Cube query (kept: same rows as the SQL)", "#eef4fb",
         '{"dimensions": ["offices.addressLine1", "offices.addressLine2"],\n'
         ' "filters": [{"member": "employees.firstName", "operator": "equals",\n'
         '              "values": ["Foon Yue"]}, {... lastName = "Tseng"}]}'),
        ("chat example (loss only on the assistant turns)", "#fdf0ea",
         "system: harness prompt + car_retails Cube schema\nuser:   Question ... Hint ... linked fields\n"
         "assistant: run_query({...})      tool: {\"row_count\": 1, ...}\n"
         "assistant: submit_answer({...})  tool: Accepted."),
    ]
    top = 0.97
    for title, fc, body in blocks:
        n = body.count("\n") + 1
        h = 0.08 + n * 0.05
        ax2.add_patch(FancyBboxPatch((0.0, top - h), 1.0, h, boxstyle="round,pad=0.005,rounding_size=0.01",
                                     fc=fc, ec="#c9c8c2", lw=0.6, transform=ax2.transAxes))
        ax2.text(0.015, top - 0.015, title, fontsize=6.9, fontweight="bold", va="top", color=INK)
        ax2.text(0.015, top - 0.07, body, fontsize=5.9, family="DejaVu Sans Mono", va="top", color=INK,
                 linespacing=1.25)
        top -= h + 0.035
    save(fig, "fig03_data")


# ------------------------------------------------------------------ Fig 4: harness versions
def fig_harness_versions():
    fig, axes = plt.subplots(1, 2, figsize=(W, 2.05), gridspec_kw={"width_ratios": [1.1, 1]})
    ax = axes[0]
    names = ["hint_harness_v1_7b", "hint_harness_v2_7b", "hint_harness_7b"]
    labels = ["v1\nfirst prompt", "v2\n+ rules, 4 examples,\nmeasure/dim check", "v4.2 (frozen)\n+ value lookup,\nlinking, guards"]
    s = [A.summary(A.test_run(n)) for n in names]
    x = np.arange(3)
    ax.bar(x - 0.19, [v["strict"] for v in s], 0.36, color=C_H, label="strict")
    ax.bar(x + 0.19, [v["relaxed"] for v in s], 0.36, color="#86b6ef", label="relaxed")
    for i, v in enumerate(s):
        ax.text(i - 0.19, v["strict"] + 0.8, f"{v['strict']:.1f}", ha="center", fontsize=6.8)
        ax.text(i + 0.19, v["relaxed"] + 0.8, f"{v['relaxed']:.1f}", ha="center", fontsize=6.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=6.6)
    ax.set_ylabel("accuracy on 332 test q (%)")
    ax.set_ylim(0, 42)
    ygrid(ax)
    ax.legend(frameon=False, loc="upper left")
    ax.set_title("a) Harness versions, 7B, test set", loc="left")
    ax = axes[1]
    tune = [(k, A.summary(A.load_run(v))) for k, v in A.TUNE_RUNS.items()]
    ax.plot(range(4), [t[1]["strict"] for t in tune], "-o", color=C_H, ms=4, lw=1.6, label="120-q tuning subset")
    t363 = {k: A.summary(A.load_run(v))["strict"] for k, v in A.TUNE363_RUNS.items()}
    ax.plot([1, 4], [t363["v3 7B"], t363["v4.2 7B"]], "s--", color=C_H, ms=4, lw=1.0, mfc="white",
            label="all 363 tuning q")
    ax.plot([4], [t363["v4.2 14B"]], "D", color=INK, ms=4.5, label="v4.2 with 14B")
    for xx, yy in [(1, t363["v3 7B"]), (4, t363["v4.2 7B"]), (4, t363["v4.2 14B"])]:
        ax.text(xx + 0.12, yy, f"{yy:.1f}", va="center", fontsize=6.6)
    for i, t in enumerate(tune):
        ax.text(i, t[1]["strict"] - 2.4, f"{t[1]['strict']:.1f}", ha="center", fontsize=6.6)
    ax.set_xticks(range(5))
    ax.set_xticklabels(["v2", "v3", "v4", "v4.1", "v4.2"])
    ax.set_xlim(-0.4, 4.9)
    ax.set_ylim(15, 42)
    ax.set_ylabel("strict accuracy (%)")
    ygrid(ax)
    ax.legend(frameon=False, loc="upper left", fontsize=6.6)
    ax.set_title("b) Tuning dbs (test set not used)", loc="left")
    fig.tight_layout(w_pad=1.5)
    save(fig, "fig04_harness_versions")


# ------------------------------------------------------------------ Fig 5: fine-tuning method
def fig_method():
    fig, ax = canvas(W, 2.35)
    box(ax, 0.02, 0.95, 1.35, 1.25, "SFT data", "2,929 / 2,621 chats\n(7B / 14B length cap)\n258 / 139 val", fs=6.6)
    box(ax, 1.62, 0.95, 1.7, 1.25, "QLoRA (PEFT)",
        "base frozen in 4-bit NF4\nLoRA r=16, alpha=32\nall attention + MLP proj.\n2 epochs, lr 2e-4", color=C_L,
        fill="#fdf0ea", fs=6.6)
    box(ax, 3.57, 0.95, 1.5, 1.25, "Export", "merge adapter\n-> GGUF f16\n-> Q4_K_M\n(same as base)", fs=6.6)
    box(ax, 5.32, 0.95, 1.66, 1.25, "Ollama model", "base Modelfile and\nchat template;\nsame harness,\nsame tools", fs=6.6)
    for a, b in [(1.37, 1.62), (3.32, 3.57), (5.07, 5.32)]:
        arrow(ax, (a, 1.57), (b, 1.57))
    # prompt profiles strip
    ax.text(0.02, 0.72, "Prompt profiles (schema-offloading arms): what each LoRA is trained and tested with",
            fontsize=7, fontweight="bold", color=INK)
    rows = [("full", "query guide + examples | tool JSON schemas | full Cube schema | linked fields", C_L),
            ("lean", "one line naming the tools | compact schema (no measure lines) | linked fields", C_LEAN),
            ("noschema", "database name only (schema learned from 1,300 synthetic q on the test dbs)", C_NS)]
    for i, (n, d, c) in enumerate(rows):
        yy = 0.47 - i * 0.2
        ax.add_patch(FancyBboxPatch((0.02, yy), 0.75, 0.15, boxstyle="round,pad=0.005,rounding_size=0.03",
                                    fc=c, ec="none"))
        ax.text(0.395, yy + 0.075, n, ha="center", va="center", fontsize=6.8, color="white", fontweight="bold")
        ax.text(0.9, yy + 0.075, d, va="center", fontsize=6.6, color=INK)
    save(fig, "fig05_method")


# ------------------------------------------------------------------ Fig 6: training curves
def fig_training():
    fig, axes = plt.subplots(1, 2, figsize=(W, 2.1), sharey=True)
    specs = [("lora_full_7b", "7B full", C_L, "-"), ("lora_full_14b", "14B full", C_L, "--"),
             ("lora_lean_7b", "7B lean", C_LEAN, "-"), ("lora_lean_14b", "14B lean", C_LEAN, "--"),
             ("lora_noschema_7b", "7B noschema", C_NS, "-"), ("lora_noschema_14b", "14B noschema", C_NS, "--")]
    for name, lab, c, ls in specs:
        d = A.training_log(name)
        axes[0].plot(d["train"]["epoch"], d["train"]["loss"], ls, color=c, lw=1.2, label=lab)
        axes[1].plot(d["eval"]["epoch"], d["eval"]["eval_loss"], ls, marker="o", ms=3, color=c, lw=1.2, label=lab)
    axes[0].set_title("a) Training loss (every 10 optimiser steps)", loc="left")
    axes[1].set_title("b) Validation loss on held-out dbs", loc="left")
    for ax in axes:
        ax.set_xlabel("epoch")
        ax.set_yscale("log")
        ax.set_yticks([0.25, 0.3, 0.5, 0.7, 1.0, 1.5])
        ax.set_yticklabels(["0.25", "0.3", "0.5", "0.7", "1.0", "1.5"])
        ax.minorticks_off()
        ygrid(ax)
    axes[0].set_ylabel("cross-entropy (log scale)")
    axes[1].legend(frameon=False, ncol=2, fontsize=6.6, loc="upper right")
    axes[1].annotate("epoch 0 = before training", (0.0, 1.0), (0.35, 0.62), fontsize=6.4, color=MUTED,
                     arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6))
    fig.tight_layout(w_pad=1.2)
    save(fig, "fig06_training")


# ------------------------------------------------------------------ Fig 7: main results
def fig_main():
    fig, axes = plt.subplots(1, 2, figsize=(W, 2.3), gridspec_kw={"width_ratios": [0.8, 1.6]})
    ax = axes[0]
    pairs = [("7B", "harness_7b", "lora_7b"), ("14B", "harness_14b", "lora_14b")]
    x = np.arange(2)
    for j, (col, lab, alpha) in enumerate([("strict", "strict", 1.0), ("relaxed", "relaxed", 0.45)]):
        for k, (arm, c) in enumerate([(1, C_H), (2, C_L)]):
            vals = [A.summary(A.test_run(p[arm]))[col] for p in pairs]
            pos = x + (k - 0.5) * 0.4 + (j - 0.5) * 0.19
            ax.bar(pos, vals, 0.18, color=c, alpha=alpha)
            for p_, v in zip(pos, vals):
                ax.text(p_, v + 0.8, f"{v:.0f}", ha="center", fontsize=6.4)
    ax.set_xticks(x)
    ax.set_xticklabels(["7B", "14B"])
    ax.set_ylim(0, 78)
    ax.set_ylabel("execution accuracy (%)")
    ygrid(ax)
    ax.legend(handles=[Patch(color=C_H, label="harness only"), Patch(color=C_L, label="LoRA, full prompt"),
                       Patch(color="#9a9994", label="dark = strict"),
                       Patch(color="#9a9994", alpha=0.45, label="light = relaxed")],
              frameon=False, fontsize=6.5, loc="upper left", ncol=2)
    ax.set_title("a) Before / after LoRA (332 test q)", loc="left")
    ax = axes[1]
    groups = [("F1\n(174)", "db_id", "formula_1"), ("club\n(158)", "db_id", "student_club"),
              ("no join\n(74)", "has_join", False), ("join\n(258)", "has_join", True),
              ("simple\n(218)", "difficulty", "simple"), ("moder.\n(91)", "difficulty", "moderate"),
              ("hard\n(23)", "difficulty", "challenging")]
    xs = np.arange(len(groups))
    for k, r in enumerate(["harness_7b", "lora_7b", "harness_14b", "lora_14b"]):
        vals = [A.breakdown(r, g[1]).get(g[2], np.nan) for g in groups]
        bar14(ax, xs + (k - 1.5) * 0.2, vals, 0.19, r, horizontal=False)
    ax.set_xticks(xs)
    ax.set_xticklabels([g[0] for g in groups], fontsize=6.4)
    ax.set_ylabel("strict accuracy (%)")
    ax.set_ylim(0, 80)
    ygrid(ax)
    ax.legend(frameon=False, fontsize=6.4, ncol=2, loc="upper right")
    ax.set_title("b) Strict accuracy by database, joins and BIRD difficulty", loc="left")
    fig.tight_layout(w_pad=1.0)
    save(fig, "fig07_main")


# ------------------------------------------------------------------ Fig 8: tokens vs accuracy
def fig_offload():
    fig, axes = plt.subplots(1, 2, figsize=(W, 2.5), sharey=True)
    arms = [("harness", "harness only", C_H, "o"), ("lora", "LoRA, full prompt", C_L, "o"),
            ("lean", "LoRA, lean prompt", C_LEAN, "s"), ("noschema_lean", "noschema LoRA, lean prompt", C_NS, "D"),
            ("noschema", "noschema LoRA, no schema", C_NS, "^"),
            ("sv3", "lookup LoRA, full prompt", C_L, "P"), ("lean_sv3", "lookup LoRA, lean prompt", C_LEAN, "P")]
    for ax, size in zip(axes, ["7b", "14b"]):
        for key, lab, c, m in arms:
            s = A.summary(A.test_run(f"{key}_{size}"))
            ax.scatter(s["prompt_tokens"] / 1000, s["strict"], s=22 if key == "lean" else 34, color=c, marker=m,
                       zorder=4 if key == "lean" else 3,
                       edgecolor="white", lw=0.8, label=lab)
            off = (5, -9) if key == "lean" else (5, 3)
            ax.annotate(f"{s['strict']:.1f}", (s["prompt_tokens"] / 1000, s["strict"]), off,
                        textcoords="offset points", fontsize=6.4)
        ax.set_xlabel("prompt tokens per question (thousands, all model calls)")
        ax.set_xlim(0, 17)
        ygrid(ax)
        ax.set_title(f"{'a' if size == '7b' else 'b'}) {size.upper()}: accuracy vs prompt cost", loc="left")
    axes[0].set_ylabel("strict accuracy (%)")
    axes[0].set_ylim(10, 50)
    axes[1].set_ylim(10, 50)
    fig.tight_layout(w_pad=1.0, rect=(0, 0.13, 1, 1))
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, fontsize=6.6, loc="lower center", ncol=4, bbox_to_anchor=(0.5, 0.0))
    save(fig, "fig08_offload")


# ------------------------------------------------------------------ Fig 9: real outputs
def _trace_lines(row, width=60, max_rows=7):
    out = []
    for s in row["trace"] or []:
        if s.get("role") == "assistant" and s.get("tool"):
            q = s.get("args", {})
            q = q.get("query", q)
            out.append(("call", f"{s['tool']}({json.dumps(q, separators=(',', ':'), ensure_ascii=False)})"))
        elif s.get("role") == "tool":
            out.append(("res", s.get("content", "").split("\n")[0]))
        elif s.get("content"):  # tool calls the model wrote as plain text (parsed by the harness)
            for ln in s["content"].splitlines():
                ln = ln.strip()
                if ln.startswith('{"name"'):
                    try:
                        c = json.loads(ln)
                    except ValueError:
                        continue
                    q = c.get("arguments", {})
                    q = q.get("query", q)
                    out.append(("call", f"{c['name']}({json.dumps(q, separators=(',', ':'), ensure_ascii=False)}) [as text]"))
    lines = []
    for kind, t in out[:max_rows]:
        wrapped = textwrap.wrap(t, width) or [""]
        if len(wrapped) > 3:
            wrapped = wrapped[:3]
            wrapped[-1] = wrapped[-1][: width - 4] + " ..."
        for i, w in enumerate(wrapped):
            lines.append((kind, ("> " if kind == "call" else "< ") + w if i == 0 else "  " + w))
    return lines


def fig_outputs(wide=False):
    cases = [
        (923, "harness_7b", "lora_7b", "A. Fixed by the LoRA (7B)"),
        (1314, "harness_7b", "lora_7b", "B. Broken by the LoRA (7B): no value lookup"),
        (1316, "lean_14b", "noschema_14b", "C. No schema in the prompt (14B)"),
    ]
    fig = plt.figure(figsize=(W, 5.2 if wide else 6.6))
    gs = fig.add_gridspec(len(cases), 2, hspace=0.42 if wide else 0.36, wspace=0.03, left=0.005, right=0.995, top=0.96, bottom=0.01)
    for i, (qid, a, b, title) in enumerate(cases):
        for j, run in enumerate([a, b]):
            ax = fig.add_subplot(gs[i, j])
            ax.axis("off")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            row = A.test_run(run).loc[qid]
            ok = bool(row["correct"])
            ec = C_OK if ok else C_BAD
            ax.add_patch(FancyBboxPatch((0, 0), 1, 0.86, boxstyle="round,pad=0.004,rounding_size=0.015",
                                        fc="#fafaf8", ec=ec, lw=1.1, transform=ax.transAxes))
            ax.text(0.02, 0.82, f"{A.TEST_RUNS[run][1]}:  " + ("correct" if ok else "wrong"),
                    fontsize=7, fontweight="bold", color=ec, va="top")
            y = 0.74
            step = 0.092 if wide else 0.071
            for kind, t in _trace_lines(row, width=55, max_rows=5 if wide else 7):
                ax.text(0.02, y, t, fontsize=6.0, family="DejaVu Sans Mono", va="top",
                        color=INK if kind == "call" else "#5f5e5a")
                y -= step
                if y < 0.06:
                    ax.text(0.02, y, "  ...", fontsize=5.5, family="DejaVu Sans Mono", va="top", color=MUTED)
                    break
            if j == 0:
                q = textwrap.shorten(row["question"], 125, placeholder=" ...")
                ax.text(0.0, 1.08, title, fontsize=7.4, fontweight="bold", va="top",
                        color=INK, transform=ax.transAxes, clip_on=False)
                ax.text(0.0, 0.98, f"Q{qid}: \"{q}\"", fontsize=6.8, va="top",
                        color=INK, transform=ax.transAxes, clip_on=False)
    save(fig, "fig09_outputs_wide" if wide else "fig09_outputs")


# ------------------------------------------------------------------ Fig 10: errors
def fig_errors():
    runs = ["harness_7b", "lora_7b", "harness_14b", "lora_14b"]
    cats = [c for c in A.ERROR_ORDER]
    data = np.array([A.error_counts(r).reindex(cats).values for r in runs])
    fig, ax = plt.subplots(figsize=(W, 2.6))
    y = np.arange(len(cats))[::-1]
    for k, r in enumerate(runs):
        b = bar14(ax, y + (1.5 - k) * 0.2, data[k], 0.19, r, horizontal=True)
        b.set_label(f"{A.TEST_RUNS[r][1]} ({332 - int(A.test_run(r)['correct'].sum())} wrong)")
        for yy, v in zip(y + (1.5 - k) * 0.2, data[k]):
            ax.text(v + 0.6, yy, str(int(v)), va="center", fontsize=5.6, color=MUTED)
    ax.set_yticks(y)
    ax.set_yticklabels(cats, fontsize=7)
    ax.set_xlabel("wrong answers out of 332 test questions (fewer is better)")
    ax.xaxis.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=6.6, loc="lower center", bbox_to_anchor=(0.4, 1.0), ncol=4)
    save(fig, "fig10_errors")


if __name__ == "__main__":
    for f in [fig_glance, fig_pipeline, fig_data, fig_harness_versions, fig_method, fig_training,
              fig_main, fig_offload, fig_outputs, fig_errors]:
        f()
        print("drew", f.__name__)
