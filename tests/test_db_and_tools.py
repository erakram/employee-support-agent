from app import db
from app.cache import TTLCache
from app.tools import TOOL_SCHEMAS, ToolExecutor


def test_init_db_seeds_once(db_path):
    db.init_db(db_path)  # second call must not duplicate rows
    assert len(db.search_employees(db_path, department="", limit=25)) == 12


def test_search_employees_by_name_and_department(db_path):
    assert [e["name"] for e in db.search_employees(db_path, name="priya")] == ["Priya Nair"]
    assert {e["name"] for e in db.search_employees(db_path, department="Sales")} == {
        "Fatima Zahra", "James O'Connor", "Hana Sato"}


def test_salary_is_never_returned(db_path):
    employee = db.get_employee(db_path, 3)
    assert "salary" not in employee
    assert all("salary" not in e for e in db.search_employees(db_path, department="Engineering"))


def test_get_employee_includes_manager_and_reports(db_path):
    omar = db.get_employee(db_path, 2)
    assert omar["manager_name"] == "Sarah Khan"
    assert omar["direct_reports"] == ["Aisha Rahman", "Daniel Reyes", "Priya Nair"]
    assert db.get_employee(db_path, 999) is None


def test_search_policies_matches_keywords(db_path):
    topics = [p["topic"] for p in db.search_policies(db_path, "remote work")]
    assert "remote-work" in topics


def test_sql_injection_is_treated_as_text(db_path):
    assert db.search_employees(db_path, name="' OR 1=1 --") == []


def test_tool_schemas_match_executor(db_path):
    tools = ToolExecutor(db_path)
    for schema in TOOL_SCHEMAS:
        result = tools.run(schema["function"]["name"], {"employee_id": 1, "name": "Sarah", "query": "leave"})
        assert "Unknown tool" not in result.get("error", "")


def test_tool_errors_are_returned_not_raised(db_path):
    tools = ToolExecutor(db_path)
    assert "Unknown tool" in tools.run("delete_everything", {})["error"]
    assert "integer" in tools.run("get_employee", {"employee_id": "abc"})["error"]
    assert "at least one" in tools.run("search_employees", {})["error"]
    assert tools.run("get_employee", {"employee_id": 999}) == {"error": "No employee with that id."}


def test_tool_accepts_numeric_strings(db_path):
    assert ToolExecutor(db_path).run("get_leave_balance", {"employee_id": "4"})["leave_balance"]["annual_days"] == 9


def test_tool_results_are_cached(db_path):
    tools = ToolExecutor(db_path)
    tools.run("list_departments", {})
    tools.run("list_departments", {})
    assert tools.cache.hits == 1


def test_ttl_cache_expires_and_evicts(monkeypatch):
    now = [100.0]
    monkeypatch.setattr("app.cache.time.monotonic", lambda: now[0])
    cache = TTLCache(ttl_seconds=10, max_size=2)
    cache.set("a", 1)
    assert cache.get("a") == 1
    now[0] += 11
    assert cache.get("a") is None  # expired
    cache.set("b", 2)
    cache.set("c", 3)
    cache.set("d", 4)
    assert cache.get("b") is None  # evicted: oldest of three in a size-2 cache
    assert cache.get("d") == 4
