"""SQLite data layer: schema, sample data, and the read-only queries the agent's tools use.

Salaries are stored in the database but no query here ever returns them, so the
LLM cannot leak data it never sees.
"""

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

# Columns that are safe to show to any employee. Salary is deliberately excluded.
PUBLIC_COLUMNS = "e.id, e.name, e.email, e.department, e.title, e.location, e.hire_date"

SCHEMA = """
CREATE TABLE IF NOT EXISTS employees (
    id          INTEGER PRIMARY KEY,
    name        TEXT NOT NULL,
    email       TEXT NOT NULL UNIQUE,
    department  TEXT NOT NULL,
    title       TEXT NOT NULL,
    manager_id  INTEGER REFERENCES employees(id),
    location    TEXT NOT NULL,
    hire_date   TEXT NOT NULL,
    salary      INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS leave_balances (
    employee_id INTEGER PRIMARY KEY REFERENCES employees(id),
    annual_days REAL NOT NULL,
    sick_days   REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS policies (
    topic TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    body  TEXT NOT NULL
);
"""

# Fictional sample company ("Northwind Labs").
EMPLOYEES = [
    # id, name, email, department, title, manager_id, location, hire_date, salary
    (1, "Sarah Khan", "sarah.khan@northwind.example", "Leadership", "CEO", None, "Dubai", "2018-03-01", 420000),
    (2, "Omar Haddad", "omar.haddad@northwind.example", "Engineering", "VP of Engineering", 1, "Dubai", "2019-01-15", 310000),
    (3, "Priya Nair", "priya.nair@northwind.example", "Engineering", "Senior Backend Engineer", 2, "Bengaluru", "2020-06-01", 180000),
    (4, "Daniel Reyes", "daniel.reyes@northwind.example", "Engineering", "Frontend Engineer", 2, "Remote", "2022-02-14", 140000),
    (5, "Aisha Rahman", "aisha.rahman@northwind.example", "Engineering", "ML Engineer", 2, "Dubai", "2023-09-04", 165000),
    (6, "Lena Fischer", "lena.fischer@northwind.example", "People", "Head of People", 1, "Berlin", "2019-05-20", 210000),
    (7, "Yusuf Ali", "yusuf.ali@northwind.example", "People", "HR Specialist", 6, "Dubai", "2021-11-01", 95000),
    (8, "Mei Chen", "mei.chen@northwind.example", "Finance", "Finance Manager", 1, "Singapore", "2020-01-06", 190000),
    (9, "Carlos Mendes", "carlos.mendes@northwind.example", "Finance", "Accountant", 8, "Remote", "2023-03-13", 98000),
    (10, "Fatima Zahra", "fatima.zahra@northwind.example", "Sales", "Head of Sales", 1, "Dubai", "2019-08-26", 230000),
    (11, "James O'Connor", "james.oconnor@northwind.example", "Sales", "Account Executive", 10, "London", "2022-07-18", 120000),
    (12, "Hana Sato", "hana.sato@northwind.example", "Sales", "Sales Development Rep", 10, "Remote", "2024-04-08", 82000),
]

LEAVE_BALANCES = [
    (1, 18, 10), (2, 22, 8), (3, 14.5, 6), (4, 9, 10), (5, 21, 9), (6, 25, 10),
    (7, 11, 7), (8, 16, 10), (9, 20, 10), (10, 7.5, 4), (11, 13, 10), (12, 22, 10),
]

POLICIES = [
    ("annual-leave", "Annual Leave",
     "Full-time employees get 25 days of paid annual leave per year, accrued monthly. "
     "Up to 5 unused days carry over to the next year and must be used by 31 March. "
     "Request leave in the HR portal at least 2 weeks ahead; your manager approves it."),
    ("sick-leave", "Sick Leave",
     "Employees get 10 paid sick days per year. For absences longer than 2 consecutive days, "
     "upload a doctor's note to the HR portal. Sick days do not carry over."),
    ("remote-work", "Remote Work",
     "Employees may work remotely up to 3 days a week with manager approval. Fully remote roles "
     "are marked 'Remote' as their location. Working from another country for more than 30 days "
     "a year needs approval from the People team."),
    ("expenses", "Expense Reimbursement",
     "Submit business expenses with receipts within 30 days through the finance portal. "
     "Meals while travelling are covered up to USD 60 per day. Expenses over USD 500 need "
     "pre-approval from your manager. Reimbursements are paid with the next payroll."),
    ("parental-leave", "Parental Leave",
     "Primary caregivers get 16 weeks of fully paid parental leave; secondary caregivers get "
     "6 weeks. Notify the People team at least 8 weeks before the expected start date."),
    ("learning-budget", "Learning Budget",
     "Each employee has a USD 1,500 yearly learning budget for courses, books and conferences. "
     "Request it through the People team; unused budget does not carry over."),
]


@contextmanager
def connect(db_path: Path) -> Iterator[sqlite3.Connection]:
    """Open a connection, commit on success, and always close it."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        with conn:
            yield conn
    finally:
        conn.close()


def init_db(db_path: Path) -> None:
    """Create the tables and load sample data if the database is empty."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with connect(db_path) as conn:
        conn.executescript(SCHEMA)
        if conn.execute("SELECT COUNT(*) FROM employees").fetchone()[0] == 0:
            conn.executemany("INSERT INTO employees VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", EMPLOYEES)
            conn.executemany("INSERT INTO leave_balances VALUES (?, ?, ?)", LEAVE_BALANCES)
            conn.executemany("INSERT INTO policies VALUES (?, ?, ?)", POLICIES)


# --- Queries used by the agent's tools. All use parameters, never string formatting. ---


def search_employees(db_path: Path, name: str | None = None, department: str | None = None,
                     limit: int = 10) -> list[dict]:
    sql = f"SELECT {PUBLIC_COLUMNS} FROM employees e WHERE 1=1"
    params: list = []
    if name:
        sql += " AND e.name LIKE ?"
        params.append(f"%{name}%")
    if department:
        sql += " AND e.department LIKE ?"
        params.append(f"%{department}%")
    sql += " ORDER BY e.name LIMIT ?"
    params.append(max(1, min(limit, 25)))
    with connect(db_path) as conn:
        return [dict(row) for row in conn.execute(sql, params)]


def get_employee(db_path: Path, employee_id: int) -> dict | None:
    sql = f"""
        SELECT {PUBLIC_COLUMNS}, m.name AS manager_name, m.email AS manager_email
        FROM employees e LEFT JOIN employees m ON e.manager_id = m.id
        WHERE e.id = ?
    """
    with connect(db_path) as conn:
        row = conn.execute(sql, (employee_id,)).fetchone()
        if row is None:
            return None
        employee = dict(row)
        employee["direct_reports"] = [
            r["name"] for r in conn.execute(
                "SELECT name FROM employees WHERE manager_id = ? ORDER BY name", (employee_id,))
        ]
        return employee


def get_leave_balance(db_path: Path, employee_id: int) -> dict | None:
    sql = """
        SELECT e.id AS employee_id, e.name, l.annual_days, l.sick_days
        FROM leave_balances l JOIN employees e ON e.id = l.employee_id
        WHERE e.id = ?
    """
    with connect(db_path) as conn:
        row = conn.execute(sql, (employee_id,)).fetchone()
        return dict(row) if row else None


def search_policies(db_path: Path, query: str) -> list[dict]:
    """Match each word of the query against policy topics, titles and text."""
    words = [w for w in query.lower().split() if len(w) > 2] or [query.lower()]
    clauses = " OR ".join(["(topic || ' ' || title || ' ' || body) LIKE ?"] * len(words))
    with connect(db_path) as conn:
        rows = conn.execute(f"SELECT topic, title, body FROM policies WHERE {clauses}",
                            [f"%{w}%" for w in words])
        return [dict(row) for row in rows]


def list_departments(db_path: Path) -> list[dict]:
    with connect(db_path) as conn:
        rows = conn.execute(
            "SELECT department, COUNT(*) AS headcount FROM employees GROUP BY department ORDER BY department")
        return [dict(row) for row in rows]
