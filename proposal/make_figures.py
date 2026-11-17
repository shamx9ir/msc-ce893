"""Draws the proposal figures into proposal/figures/.

    python proposal/make_figures.py

Needs only matplotlib.
"""
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

HERE = Path(__file__).parent
OUT = HERE / "figures"
OUT.mkdir(exist_ok=True)

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
    fig = plt.figure(figsize=(13, 5.4))
    a1 = fig.add_axes([0, 0.5, 1, 0.5])
    a1.set_xlim(0, 13)
    a1.set_ylim(0, 2.0)
    a1.axis("off")
    a1.text(0.05, 1.95, "a) Planned preparation of the LoRA training data (BIRD train split)", fontsize=12,
            va="top")
    steps = [("1  Load", "BIRD train:\nquestion, hint,\ngold SQL, SQLite DB"),
             ("2  Cube models", "generate cubes from\nschema + column\ndescriptions"),
             ("3  Translate", "rule-based\nSQL -> Cube JSON;\ndrop what Cube\ncannot express"),
             ("4  Verify", "run both; keep pair\nonly if results match;\ndrop empty results"),
             ("5  Format", "chat examples with\nthe harness prompt\nand tool calls;\nhold out 3 DBs")]
    for i, (t, b) in enumerate(steps):
        x = 0.1 + i * 2.6
        col, fill = (GREEN, "#eaf6ef") if i == 3 else (BLUE, "#eef4fb")
        box(a1, x, 0.05, 2.2, 1.6, t, b, color=col, fill=fill)
        if i < 4:
            arrow(a1, (x + 2.2, 0.85), (x + 2.6, 0.85))

    a2 = fig.add_axes([0, 0, 1, 0.46])
    a2.axis("off")
    a2.text(0.005, 0.97, "b) One prepared pair (address database)", fontsize=12, va="top",
            transform=a2.transAxes)
    left = [("Q: What is the total number of households in Arecibo county?", INK, "DejaVu Sans"),
            ('Hint: "ARECIBO" is the county; total number of households refer to sum(households)', MUTED,
             "DejaVu Sans"),
            ("Gold SQL: SELECT SUM(T1.households) FROM zip_data AS T1\n"
             "  INNER JOIN country AS T2 ON T1.zip_code = T2.zip_code\n  WHERE T2.county = 'ARECIBO'", INK,
             "DejaVu Sans Mono")]
    right = [('Cube query (training target):\n{"measures": ["zip_data.sum_households"],\n'
              ' "filters": [{"member": "country.county",\n'
              '              "operator": "equals", "values": ["ARECIBO"]}]}', BLUE, "DejaVu Sans Mono"),
             ("Verification: keep only if the Cube result\nequals the gold SQL result on the SQLite file", GREEN,
              "DejaVu Sans")]
    for col_x, items in [(0.005, left), (0.55, right)]:
        yy = 0.82
        for txt, c, fam in items:
            a2.text(col_x, yy, txt, va="top", fontsize=10, color=c, family=fam, transform=a2.transAxes)
            yy -= 0.1 * (txt.count("\n") + 1) + 0.07
    fig.savefig(OUT / "fig2_data_prep.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_eval():
    fig, ax = plt.subplots(figsize=(15, 4.6))
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 4.6)
    ax.axis("off")
    ax.text(0.05, 4.5, "a) Data splits: no database appears in more than one split", fontsize=12, va="top")
    splits = [("LoRA training", "BIRD train\n~9,400 questions\n69 databases", BLUE, "#eef4fb"),
              ("LoRA validation", "3 whole train\ndatabases held out", BLUE, "#eef4fb"),
              ("Harness tuning", "BIRD dev:\nsuperhero, toxicology,\ncalifornia_schools\n= 363 questions", ORANGE,
               "#fdf1e7"),
              ("Test (used once)", "BIRD dev:\nformula_1 (174)\n+ student_club (158)\n= 332 questions", GREEN,
               "#eaf6ef")]
    for i, (t, b, c, f) in enumerate(splits):
        box(ax, 0.1 + i * 2.35, 1.75, 2.2, 2.35, t, b, color=c, fill=f)
    ax.text(9.7, 4.5, "b) Four conditions, compared in pairs", fontsize=12, va="top")
    cells = [("A", "7B base", "harness only"), ("B", "7B + LoRA", "same harness"),
             ("C", "14B base", "harness only"), ("D", "14B + LoRA", "same harness")]
    for i, (k, m, h) in enumerate(cells):
        x = 9.8 + (i % 2) * 2.7
        y = 2.95 - (i // 2) * 1.25
        c, f = (BLUE, "#eef4fb") if i % 2 else (GREY, "#f2f4f6")
        ax.add_patch(FancyBboxPatch((x, y), 2.3, 0.95, boxstyle="round,pad=0.02", fc=f, ec=c, lw=1.5))
        ax.text(x + 0.15, y + 0.62, f"{k}  {m}", fontsize=11, fontweight="bold", color=INK, va="center")
        ax.text(x + 0.15, y + 0.25, h, fontsize=9.5, color=MUTED, va="center")
    arrow(ax, (12.1, 3.42), (12.5, 3.42))
    arrow(ax, (12.1, 2.17), (12.5, 2.17))
    ax.text(9.8, 1.45, "Arrow = effect of adding LoRA (A vs B, C vs D).\n"
            "Primary metric: strict execution accuracy on\nthe 332 test questions; paired McNemar tests.", fontsize=10, color=INK, va="top")
    ax.text(0.1, 1.45, "Qwen2.5-Coder-Instruct 7B and 14B; LoRA via Hugging Face PEFT (4-bit QLoRA).",
            fontsize=10, color=MUTED, va="top")
    fig.savefig(OUT / "fig3_evaluation.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    fig_pipeline()
    fig_data()
    fig_eval()
    print("figures written to", OUT)
