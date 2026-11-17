"""Draws the proposal figures into proposal/figures/.

    python proposal/make_figures.py

Needs only matplotlib. Pilot numbers come from proposal/pilot_results.json, which was
summarised from runs/test_*.jsonl and the LoRA trainer_state.json files.
"""
import json
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

HERE = Path(__file__).parent
OUT = HERE / "figures"
OUT.mkdir(exist_ok=True)
RES = json.loads((HERE / "pilot_results.json").read_text())

plt.rcParams.update({"font.size": 11, "font.family": "DejaVu Sans", "axes.spines.top": False,
                     "axes.spines.right": False})
INK, MUTED = "#1f2933", "#52606d"
BLUE, ORANGE, GREEN, GREY, RED = "#2f6fb0", "#e07b28", "#2e8b57", "#9aa5b1", "#c0392b"


def box(ax, x, y, w, h, title, body="", color=BLUE, fill="#eef4fb"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.05",
                                fc=fill, ec=color, lw=1.6))
    ax.text(x + w / 2, y + h - 0.13, title, ha="center", va="top", fontsize=11.5,
            fontweight="bold", color=INK)
    if body:
        ax.text(x + w / 2, y + h - 0.45, body, ha="center", va="top", fontsize=9.5,
                color=MUTED, linespacing=1.35)


def arrow(ax, a, b, text="", color=INK, rad=0.0, dy=0.12):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=14, lw=1.4,
                                 color=color, connectionstyle=f"arc3,rad={rad}"))
    if text:
        ax.text((a[0] + b[0]) / 2, (a[1] + b[1]) / 2 + dy, text, ha="center", va="bottom",
                fontsize=9, color=color)


def fig_pipeline():
    fig, ax = plt.subplots(figsize=(13, 5.6))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 5.6)
    ax.axis("off")

    box(ax, 0.1, 2.35, 2.3, 2.0, "1  Question",
        "English question\n+ BIRD hint (evidence)\n\n\"How many drivers\ndon't have a code?\"")
    box(ax, 3.0, 2.0, 3.0, 2.7, "2  ADK agent (harness)",
        "system prompt: Cube query guide\n+ cubes, dimensions, measures\n+ fields linked to the question\n\n"
        "model via LiteLLM / Ollama:\nQwen2.5-Coder 7B or 14B\n(base, or base + LoRA)", color=ORANGE, fill="#fdf1e7")
    box(ax, 6.7, 2.0, 2.6, 2.7, "3  Tools",
        "search_values(dim, text)\nrun_query(query)\nsubmit_answer(query)\n\nmember-name check\n+ \"did you mean ...\" hints")
    box(ax, 10.0, 2.35, 2.9, 2.0, "4  Cube semantic layer",
        "JSON query -> SQL\non DuckDB copy of\nthe BIRD database")
    box(ax, 6.2, 0.1, 3.4, 1.45, "5a  Evaluation",
        "rows == rows of BIRD gold SQL\non original SQLite (strict, relaxed)", color=GREEN, fill="#eaf6ef")
    box(ax, 10.0, 0.1, 2.9, 1.45, "5b  Demo app",
        "A2UI surface with\nHighcharts chart + table", color=GREEN, fill="#eaf6ef")

    arrow(ax, (2.4, 3.35), (3.0, 3.35))
    arrow(ax, (6.0, 3.55), (6.7, 3.55), "tool call")
    arrow(ax, (6.7, 3.0), (6.0, 3.0), "rows / error", dy=-0.38)
    arrow(ax, (9.3, 3.35), (10.0, 3.35))
    arrow(ax, (11.45, 2.35), (11.45, 1.55), "result rows", dy=0)
    arrow(ax, (8.0, 2.0), (8.0, 1.55))
    ax.text(8.1, 1.72, "submitted query", fontsize=9, color=INK, va="center")
    # LoRA branch
    ax.add_patch(FancyBboxPatch((2.7, 4.85), 3.6, 0.6, boxstyle="round,pad=0.02", fc="white",
                                ec=ORANGE, lw=1.3, ls="--"))
    ax.text(4.5, 5.15, "intervention: LoRA adapter only", ha="center", va="center", fontsize=10,
            color=ORANGE, fontweight="bold")
    ax.text(0.1, 0.9, "Same prompt, tools and\nCube models in every arm;\nonly the model weights change.",
            fontsize=10, color=MUTED, va="center")
    fig.savefig(OUT / "fig1_pipeline.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_data():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw={"width_ratios": [1.05, 1]})
    cats = ["verified (kept)", "not expressible\nby translator", "Cube error", "result differed",
            "empty gold result"]
    vals = [3531, 2782, 900, 682, 108]
    cols = [GREEN, GREY, RED, ORANGE, GREY]
    y = range(len(cats))[::-1]
    a1.barh(list(y), vals, color=cols)
    for yi, v in zip(y, vals):
        a1.text(v + 60, yi, f"{v:,}", va="center", fontsize=10.5)
    a1.set_yticks(list(y), cats)
    a1.set_xlim(0, 4300)
    a1.set_xlabel("BIRD train questions (59 databases, 8,003 in total)")
    a1.set_title("a) What happened to each training question", loc="left", fontsize=12)

    a2.axis("off")
    a2.set_title("b) One verified pair (address database)", loc="left", fontsize=12)
    q = "Q: What is the total number of households in Arecibo county?"
    h = "Hint: \"ARECIBO\" is the county; total number of households refer to sum(households)"
    sql = ("Gold SQL: SELECT SUM(T1.households) FROM zip_data AS T1\n"
           "  INNER JOIN country AS T2 ON T1.zip_code = T2.zip_code\n  WHERE T2.county = 'ARECIBO'")
    cube = ('Cube query: {"measures": ["zip_data.sum_households"],\n'
            '  "filters": [{"member": "country.county",\n'
            '               "operator": "equals", "values": ["ARECIBO"]}]}')
    chk = "Check: both return 36526  ->  keep"
    yy = 0.95
    for txt, col, fam in [(textwrap.fill(q, 62), INK, "DejaVu Sans"), (textwrap.fill(h, 66), MUTED, "DejaVu Sans"),
                          (sql, INK, "DejaVu Sans Mono"), (cube, BLUE, "DejaVu Sans Mono"), (chk, GREEN, "DejaVu Sans")]:
        a2.text(0.0, yy, txt, va="top", fontsize=9.8, color=col, family=fam, transform=a2.transAxes)
        yy -= 0.07 * (txt.count("\n") + 1) + 0.06
    fig.tight_layout()
    fig.savefig(OUT / "fig2_data_prep.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_pilot():
    acc = RES["test_accuracy"]
    arms = [("harness_7b", "7B\nharness"), ("lora_7b", "7B\nLoRA + harness"),
            ("harness_14b", "14B\nharness"), ("lora_14b", "14B\nLoRA + harness")]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.6), gridspec_kw={"width_ratios": [1.2, 1]})
    xs = range(len(arms))
    s = [acc[k]["all"]["strict"] for k, _ in arms]
    r = [acc[k]["all"]["relaxed"] for k, _ in arms]
    cols = [GREY, BLUE, GREY, BLUE]
    a1.bar([x - 0.2 for x in xs], s, 0.4, color=cols)
    a1.bar([x + 0.2 for x in xs], r, 0.4, color=cols, alpha=0.45)
    for x, v in zip(xs, s):
        a1.text(x - 0.2, v + 1, f"{v:.1f}", ha="center", fontsize=10, fontweight="bold")
    for x, v in zip(xs, r):
        a1.text(x + 0.2, v + 1, f"{v:.1f}", ha="center", fontsize=9.5, color=MUTED)
    a1.axhline(70, ls=":", color=MUTED)
    a1.text(3.45, 71, "rule-translator ceiling 70% (relaxed)", ha="right", fontsize=9, color=MUTED)
    a1.set_xticks(list(xs), [l for _, l in arms])
    a1.set_ylabel("execution accuracy (%)")
    a1.set_ylim(0, 80)
    a1.set_title("a) Pilot accuracy on 332 test questions\n    dark = strict (primary), light = relaxed", loc="left", fontsize=12)

    for key, label, col in [("qwen7b", "7B", BLUE), ("qwen14b", "14B", ORANGE)]:
        t = RES["training"][key]
        a2.plot([p[0] for p in t["train"]], [p[1] for p in t["train"]], color=col, lw=1.4,
                label=f"{label} train ({t['train_examples']:,} ex.)")
        a2.plot([p[0] for p in t["val"]], [p[1] for p in t["val"]], "o--", color=col, ms=6,
                label=f"{label} validation")
    a2.set_xlabel("optimizer step (16 examples each)")
    a2.set_ylabel("loss (assistant tokens only)")
    a2.set_ylim(0.2, 0.75)
    a2.legend(frameon=False, fontsize=9.5)
    a2.set_title("b) LoRA training progress (2 epochs)", loc="left", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "fig3_pilot.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    fig_pipeline()
    fig_data()
    fig_pilot()
    print("figures written to", OUT)
