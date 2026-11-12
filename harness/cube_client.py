"""Thin client for Cube's REST API plus helpers that render the model for prompts."""
import json
import time

import requests


class CubeError(Exception):
    pass


class CubeClient:
    def __init__(self, base_url: str = "http://localhost:4000/cubejs-api/v1",
                 token: str | None = None, timeout: float = 60):
        self.base = base_url.rstrip("/")
        self.headers = {"Authorization": token} if token else {}
        self.timeout = timeout
        self._meta = None

    # ---- raw API -------------------------------------------------------
    def meta(self) -> dict:
        if self._meta is None:
            r = requests.get(f"{self.base}/meta", headers=self.headers, timeout=self.timeout)
            r.raise_for_status()
            self._meta = r.json()
        return self._meta

    def load(self, query: dict) -> list[dict]:
        """Run a Cube query and return its rows. Raises CubeError with Cube's message."""
        deadline = time.time() + self.timeout
        while True:
            r = requests.post(f"{self.base}/load", json={"query": query},
                              headers=self.headers, timeout=self.timeout)
            body = r.json() if r.content else {}
            if body.get("error") == "Continue wait" and time.time() < deadline:
                time.sleep(0.5)
                continue
            if r.status_code != 200 or "error" in body:
                raise CubeError(body.get("error") or f"HTTP {r.status_code}")
            return body.get("data", [])

    def sql(self, query: dict) -> str:
        r = requests.post(f"{self.base}/sql", json={"query": query},
                          headers=self.headers, timeout=self.timeout)
        body = r.json()
        if "error" in body:
            raise CubeError(body["error"])
        return body["sql"]["sql"][0]

    # ---- helpers -------------------------------------------------------
    def cubes_for(self, db: str) -> list[dict]:
        return [c for c in self.meta()["cubes"] if (c.get("meta") or {}).get("db") == db]

    def member_names(self, db: str) -> set[str]:
        names = set()
        for c in self.cubes_for(db):
            for kind in ("measures", "dimensions", "segments"):
                names.update(m["name"] for m in c.get(kind, []))
        return names

    def describe(self, db: str, with_descriptions: bool = True) -> str:
        """Compact, LLM-friendly rendering of the semantic layer for one database."""
        out = []
        for c in self.cubes_for(db):
            out.append(f"cube {c['name']}")
            for d in c["dimensions"]:
                line = f"  dim  {d['name']} ({d['type']})"
                if with_descriptions and d.get("description"):
                    line += f" -- {d['description']}"
                out.append(line)
            for m in c["measures"]:
                out.append(f"  meas {m['name']} ({m.get('aggType', m['type'])})")
        return "\n".join(out)

    def dimension_values(self, dimension: str, search: str | None = None,
                         limit: int = 20) -> list:
        q = {"dimensions": [dimension], "limit": limit}
        if search:
            q["filters"] = [{"member": dimension, "operator": "contains", "values": [search]}]
        return [row[dimension] for row in self.load(q)]


def dumps(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, default=str)
