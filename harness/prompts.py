"""Prompt text shared by the harness and (later) the LoRA training data, so both see identical inputs."""

QUERY_GUIDE = """\
A Cube query is a JSON object with these optional keys:
  "measures":   ["cube.measure", ...]            aggregated values (count, sum_x, avg_x, min_x, max_x)
  "dimensions": ["cube.dimension", ...]          columns to group by / return
  "filters":    [{"member": "cube.member", "operator": OP, "values": ["v1", ...]}]
                OP is one of: equals, notEquals, contains, notContains, startsWith, endsWith,
                gt, gte, lt, lte, set, notSet, inDateRange, notInDateRange, beforeDate, afterDate.
                Values are always strings. Filters on a measure act like SQL HAVING.
                Combine with {"or": [filter, ...]} or {"and": [filter, ...]}.
  "timeDimensions": [{"dimension": "cube.time_dim", "dateRange": ["2019-01-01", "2019-12-31"],
                      "granularity": "year"}]     granularity is optional (day, week, month, quarter, year)
  "order":      {"cube.member": "asc" | "desc"}
  "limit":      N
To sort by a dimension it must also be listed in "dimensions" (Cube ignores ordering by a
dimension that is not returned). Measures can be used in "order" and "filters" without being returned.
Cube joins cubes automatically, so members from several cubes can be mixed in one query.
A query with only dimensions returns the distinct combinations of those dimensions.
Return only the columns the question asks for, in the order it asks for them.

Rules that are easy to get wrong:
  - "measures" may only contain names listed as `meas` below (count, sum_..., avg_..., min_..., max_...).
    A column such as a name or a year is a `dim` and goes in "dimensions" or "filters".
  - "How many X" is X's count measure with filters, not a list of rows.
  - "Which X has the most/highest Y" = dimension X, order by the measure for Y desc, limit 1.
  - Use search_values to get the exact stored spelling before filtering on a name or category.

Examples (for an imaginary shop database with cubes orders, customers, products):
  Q: How many orders did customers from Berlin place?
  {"measures": ["orders.count"], "filters": [{"member": "customers.city", "operator": "equals", "values": ["Berlin"]}]}
  Q: Which product category earned the most revenue in 2021?
  {"dimensions": ["products.category"], "order": {"orders.sum_amount": "desc"}, "limit": 1,
   "filters": [{"member": "orders.order_date", "operator": "inDateRange", "values": ["2021-01-01", "2021-12-31"]}]}
  Q: List the names of customers older than 60, oldest first.
  {"dimensions": ["customers.name", "customers.age"], "filters": [{"member": "customers.age", "operator": "gt", "values": ["60"]}],
   "order": {"customers.age": "desc"}}
  Q: What is the average price of products in the "Toys" or "Games" category?
  {"measures": ["products.avg_price"], "filters": [{"member": "products.category", "operator": "equals", "values": ["Toys", "Games"]}]}"""

SYSTEM_TOOLS = """\
You answer questions about the "{db}" database by writing a query for the Cube semantic layer.

{guide}

The semantic layer for "{db}":
{schema}

You have tools:
  search_values(dimension, text) - find stored values of a dimension that contain text (use it to get exact spellings)
  run_query(query)               - run a Cube query and see the first rows or the error
  submit_answer(query)           - submit your final Cube query (ends the task)
Check exact values and test your query before you submit it. Always finish by calling submit_answer."""

SYSTEM_SINGLE = """\
You answer questions about the "{db}" database by writing a query for the Cube semantic layer.

{guide}

The semantic layer for "{db}":
{schema}

Reply with the Cube query as a single JSON object and nothing else."""


# ---- prompt profiles for the schema-offloading experiment ----------------------------------
# "full"     harness v4.2: query guide + full schema; tool schemas sent through the API.
# "lean"     no query guide, no tool schemas (the LoRA learned both), compact schema.
# "noschema" no guide, no tool schemas, no schema: only the database name. Needs a LoRA that has
#            been trained on that database's schema (scripts/synth_schema_qa.py).
PROFILES = ("full", "lean", "noschema")

SYSTEM_LEAN = """\
You answer questions about the "{db}" database by writing Cube queries.
Tools: search_values(dimension, text), run_query(query), submit_answer(query).

The semantic layer for "{db}" (members are cube.column; every cube has a count measure;
columns marked number+ also have sum_, avg_, min_ and max_ measures, e.g. cube.avg_column):
{schema}"""

SYSTEM_NOSCHEMA = """\
You answer questions about the "{db}" database by writing Cube queries.
Tools: search_values(dimension, text), run_query(query), submit_answer(query)."""


def compact_schema(schema: str) -> str:
    """Shorten CubeClient.describe() text: no measure lines (count and the sum_/avg_/min_/max_
    pattern are implied, the numeric columns that have them are marked number+) and no repeated
    cube prefix on each column."""
    lines = schema.splitlines()
    measures = {l.split()[1] for l in lines if l.startswith("  meas ")}
    out = []
    for l in lines:
        if l.startswith("cube "):
            out.append(l)
        elif l.startswith("  dim "):
            head, _, desc = l[6:].strip().partition(" -- ")
            name, kind = head.split(" ", 1)
            kind = kind.strip("()")
            cube, col = name.split(".", 1)
            if kind == "number" and f"{cube}.sum_{col}" in measures:
                kind = "number+"
            out.append(f"  {col} {kind}" + (f" -- {desc}" if desc else ""))
    return "\n".join(out)


def render_profile(profile: str, db: str, schema: str) -> str:
    """The instruction (system prompt before ADK's agent line) for a tools-mode profile."""
    if profile == "full":
        return SYSTEM_TOOLS.format(db=db, guide=QUERY_GUIDE, schema=schema)
    if profile == "lean":
        return SYSTEM_LEAN.format(db=db, schema=compact_schema(schema))
    if profile == "noschema":
        return SYSTEM_NOSCHEMA.format(db=db)
    raise ValueError(profile)


def user_prompt(question: str, evidence: str | None, schema: str | None = None) -> str:
    """The per-question message. With the schema text, it also lists the fields whose names
    the question or hint mention (harness v4 schema linking)."""
    text = f"Question: {question}"
    if evidence:
        text += f"\nHint: {evidence}"
    if schema:
        from .linking import linked_fields
        fields = linked_fields(schema, question, evidence)
        if fields:
            text += ("\nPossibly relevant fields (they share words with the question or hint; this is not a "
                     "list of columns to return): " + ", ".join(fields))
    return text

