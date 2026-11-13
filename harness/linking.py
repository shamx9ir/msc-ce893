"""Schema linking: point the model at the fields the question and hint are about.

Works on the plain-text schema (CubeClient.describe), so the harness and the LoRA training
prompts (make_sft.py, which only has the saved schema text) build exactly the same prompt.
"""
import re
from dataclasses import dataclass, field

STOP = set("""a an the of in on at to for from by with and or is are was were be been what which who whom
whose how many much list name names give show tell please state among all each that this these those
there their its it as than then do does did has have had not no yes refers refer mean means
where when about into per most least more less highest lowest top id""".split())

SYNONYMS = {"color": "colour", "colours": "colour", "favorite": "favourite", "id": "id"}


def tokens(text: str) -> set[str]:
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text or "")
    out = set()
    for w in re.findall(r"[A-Za-z0-9]+", text.lower()):
        if w in STOP or len(w) < 2:
            continue
        w = w[:-1] if len(w) > 3 and w.endswith("s") and not w.endswith("ss") else w
        out.add(SYNONYMS.get(w, w))
    return out


@dataclass
class Member:
    name: str
    kind: str          # dim | meas
    line: str
    name_tokens: set = field(default_factory=set)
    desc_tokens: set = field(default_factory=set)


@dataclass
class CubeText:
    name: str
    header: str
    members: list


def parse_schema(text: str) -> list[CubeText]:
    cubes = []
    for line in text.splitlines():
        if line.startswith("cube "):
            cubes.append(CubeText(line[5:].strip(), line, []))
            continue
        m = re.match(r"\s+(dim|meas)\s+(\S+)", line)
        if m and cubes:
            name = m.group(2)
            desc = line.split(" -- ", 1)[1] if " -- " in line else ""
            short = name.split(".", 1)[1] if "." in name else name
            for agg in ("sum_", "avg_", "min_", "max_"):
                short = short[len(agg):] if short.startswith(agg) else short
            cubes[-1].members.append(Member(name, m.group(1), line, tokens(short), tokens(desc)))
    return cubes


def linked_fields(schema_text: str, question: str, evidence: str | None = None,
                  max_fields: int = 8) -> list[str]:
    """Dimensions whose names the question or hint mention, best first.

    A name the hint spells out (e.g. `bond_type = '-'`) ranks highest; words that only repeat
    the cube's own name count for little, so `superhero.id` does not match "superheroes"."""
    q = tokens(question) | tokens(evidence or "")
    hint_t = tokens(evidence or "")
    hint = (evidence or "").lower()

    def score(m: Member) -> float:
        own = m.name_tokens - tokens(m.name.split(".", 1)[0])
        s = 2.0 * len(own & q) + 1.0 * len(own & hint_t) + 0.3 * len(m.desc_tokens & q)
        s += 0.5 * len((m.name_tokens - own) & q)
        bare = m.name.split(".", 1)[-1].lower()
        if len(bare) > 3 and re.search(r"(?<![a-z0-9_])" + re.escape(bare) + r"(?![a-z0-9_])", hint):
            s += 5
        return s

    scored = [(score(m), i, m) for i, m in enumerate(
        m for c in parse_schema(schema_text) for m in c.members if m.kind == "dim")]
    return [m.name for s, _, m in sorted(scored, key=lambda x: (-x[0], x[1])) if s >= 1.5][:max_fields]
