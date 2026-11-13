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
            text += "\nFields that match words in the question or hint: " + ", ".join(fields)
    return text

