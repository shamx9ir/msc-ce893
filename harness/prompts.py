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
Cube joins cubes automatically, so members from several cubes can be mixed in one query.
A query with only dimensions returns the distinct combinations of those dimensions.
Return only the columns the question asks for, in the order it asks for them."""

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


def user_prompt(question: str, evidence: str | None) -> str:
    text = f"Question: {question}"
    if evidence:
        text += f"\nHint: {evidence}"
    return text


TOOLS = [
    {"type": "function", "function": {
        "name": "search_values",
        "description": "List stored values of a dimension that contain the given text (case-insensitive).",
        "parameters": {"type": "object", "properties": {
            "dimension": {"type": "string", "description": "e.g. drivers.surname"},
            "text": {"type": "string"}}, "required": ["dimension", "text"]}}},
    {"type": "function", "function": {
        "name": "run_query",
        "description": "Run a Cube query and return up to 10 rows, the row count, or the error.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "object", "description": "Cube query JSON"}}, "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "submit_answer",
        "description": "Submit the final Cube query.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "object", "description": "Cube query JSON"}}, "required": ["query"]}}},
]
