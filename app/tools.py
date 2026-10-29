"""Tools the LLM can call: their JSON schemas, and the code that runs them.

The LLM only ever sees the schemas below. When it asks for a tool, ToolExecutor
validates the arguments, runs the matching database query (with caching), and
returns a JSON-serialisable result. Errors are returned to the LLM as data
instead of crashing the request, so it can correct itself.
"""

import json
import logging
import time
from pathlib import Path

from app import db
from app.cache import TTLCache

logger = logging.getLogger(__name__)

# OpenAI-style function definitions, which Groq uses for tool calling.
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "search_employees",
            "description": "Find employees by (part of) their name and/or department. "
                           "Returns id, name, email, department, title, location and hire date.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Full or partial name, e.g. 'Priya'."},
                    "department": {"type": "string", "description": "e.g. 'Engineering', 'Sales'."},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_employee",
            "description": "Get one employee's full profile, including their manager and direct reports. "
                           "Use search_employees first if you only know the name.",
            "parameters": {
                "type": "object",
                "properties": {"employee_id": {"type": "integer"}},
                "required": ["employee_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_leave_balance",
            "description": "Get an employee's remaining annual leave and sick days.",
            "parameters": {
                "type": "object",
                "properties": {"employee_id": {"type": "integer"}},
                "required": ["employee_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_policies",
            "description": "Search company HR policies (leave, remote work, expenses, parental leave, "
                           "learning budget) by keywords.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "Keywords, e.g. 'remote work'."}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_departments",
            "description": "List all departments with their headcount.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]


class ToolError(Exception):
    """Bad arguments or unknown tool; the message is shown to the LLM."""


class ToolExecutor:
    def __init__(self, db_path: Path, cache_ttl: float = 300):
        self.db_path = db_path
        self.cache = TTLCache(cache_ttl)

    def run(self, name: str, arguments: dict) -> dict:
        """Run a tool and always return a dict (results or {"error": ...})."""
        cache_key = f"{name}:{json.dumps(arguments, sort_keys=True)}"
        cached = self.cache.get(cache_key)
        if cached is not None:
            logger.info("tool=%s args=%s cache=hit", name, arguments)
            return cached

        start = time.perf_counter()
        try:
            result = self._dispatch(name, arguments)
        except ToolError as e:
            logger.warning("tool=%s args=%s error=%s", name, arguments, e)
            return {"error": str(e)}
        except Exception:
            logger.exception("tool=%s args=%s crashed", name, arguments)
            return {"error": "Internal error while running the tool."}

        logger.info("tool=%s args=%s cache=miss ms=%.1f", name, arguments,
                    (time.perf_counter() - start) * 1000)
        self.cache.set(cache_key, result)
        return result

    def _dispatch(self, name: str, args: dict) -> dict:
        if name == "search_employees":
            name_q, dept = _optional_str(args, "name"), _optional_str(args, "department")
            if not name_q and not dept:
                raise ToolError("Give at least one of 'name' or 'department'.")
            matches = db.search_employees(self.db_path, name=name_q, department=dept)
            return {"count": len(matches), "employees": matches}

        if name == "get_employee":
            employee = db.get_employee(self.db_path, _required_int(args, "employee_id"))
            return {"employee": employee} if employee else {"error": "No employee with that id."}

        if name == "get_leave_balance":
            balance = db.get_leave_balance(self.db_path, _required_int(args, "employee_id"))
            return {"leave_balance": balance} if balance else {"error": "No leave record for that id."}

        if name == "search_policies":
            query = _optional_str(args, "query")
            if not query:
                raise ToolError("'query' is required.")
            policies = db.search_policies(self.db_path, query)
            return {"count": len(policies), "policies": policies}

        if name == "list_departments":
            return {"departments": db.list_departments(self.db_path)}

        raise ToolError(f"Unknown tool '{name}'.")


def _optional_str(args: dict, key: str) -> str | None:
    value = args.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ToolError(f"'{key}' must be a string.")
    return value.strip() or None


def _required_int(args: dict, key: str) -> int:
    value = args.get(key)
    # LLMs sometimes send numbers as strings ("3"), so accept those too.
    if isinstance(value, str) and value.strip().isdigit():
        value = int(value)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ToolError(f"'{key}' must be an integer.")
    return value
