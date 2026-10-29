# Employee Support Agent

![tests](https://github.com/erakram/employee-support-agent/actions/workflows/tests.yml/badge.svg)

An AI agent that answers employee questions — "Who is Priya's manager?", "How much leave do I have left?", "What's the remote work policy?" — by calling tools that query a company database, instead of guessing.

Built with **Groq** (LLM function calling), **SQLite**, and **FastAPI**, with caching, retries, timeouts, structured logging, and tests.

<!-- Add a screenshot or demo GIF here: ![demo](docs/demo.gif) -->

## How it works

```
Browser chat UI ──► FastAPI  /api/chat ──► Agent loop ──► Groq LLM
                                              │   ▲
                                   tool calls │   │ tool results (JSON)
                                              ▼   │
                                      ToolExecutor (validation + TTL cache)
                                              │
                                              ▼
                                       SQLite (employees, leave, policies)
```

1. The user's message, the chat history and the **tool schemas** go to the LLM.
2. If the LLM asks for tools (e.g. `search_employees`, then `get_leave_balance`), the app validates the arguments, runs parameterized SQL, and sends the results back.
3. This loops until the LLM replies with a plain-text answer, with a step limit to stop runaway loops.
4. The UI shows which tools were called for each answer, so every answer can be traced to data.

### Tools

| Tool | What it does |
|---|---|
| `search_employees` | Find people by name and/or department |
| `get_employee` | Full profile with manager and direct reports |
| `get_leave_balance` | Remaining annual and sick days |
| `search_policies` | Keyword search over HR policies |
| `list_departments` | Departments with headcount |

## Engineering details

- **Data privacy by design** — salaries are in the database, but no tool query selects them, so the LLM can't leak data it never sees.
- **Safe SQL** — every query is parameterized; a test checks that SQL injection input is treated as plain text.
- **Robust tool calls** — bad JSON, wrong types, or unknown tools are returned to the LLM as error messages so it can correct itself, instead of crashing the request.
- **Caching** — tool results are cached in memory with a TTL and LRU eviction.
- **Retries and timeouts** — the Groq client retries rate limits, timeouts and 5xx errors with exponential backoff; each request has a timeout.
- **Friendly errors** — rate limits, timeouts and outages map to clear HTTP 429/503/504 messages.
- **Observability** — each request gets an ID; logs record latency per request, per LLM step and per tool call, plus cache hits.
- **Input limits** — request validation caps message length and history size, and the history can't include `system` messages.
- **Tests** — 26 pytest tests for the database, tools, cache, agent loop (with a fake LLM, so tests need no API key) and API.

## Run it locally

Requires Python 3.11+ and a free Groq API key from <https://console.groq.com/keys>.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then paste your key into .env
uvicorn app.main:app --reload
```

Open <http://localhost:8000>. The SQLite database is created with sample data on first start.

Run the tests:

```bash
pytest -q
```

## API

`POST /api/chat`

```json
{ "message": "Who is Priya Nair's manager?", "history": [] }
```

Response:

```json
{
  "answer": "Priya Nair's manager is Omar Haddad (VP of Engineering).",
  "tool_calls": [
    { "name": "search_employees", "arguments": { "name": "Priya Nair" } },
    { "name": "get_employee", "arguments": { "employee_id": 3 } }
  ],
  "latency_ms": 1240
}
```

`GET /api/health` returns status and whether the LLM is configured. Interactive API docs are at `/docs`.

## Deploy (free)

1. Push this folder to a GitHub repository.
2. On [Render](https://render.com), choose **New → Blueprint** and select the repo (it uses `render.yaml` and the `Dockerfile`).
3. Enter your `GROQ_API_KEY` when prompted.

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `GROQ_API_KEY` | — | Required for chat |
| `GROQ_MODEL` | `openai/gpt-oss-120b` | Any Groq model with tool calling |
| `REQUEST_TIMEOUT` | `30` | Seconds per LLM request |
| `MAX_RETRIES` | `3` | Automatic retries on transient errors |
| `MAX_AGENT_STEPS` | `6` | Max LLM calls per question |
| `CACHE_TTL` | `300` | Seconds tool results stay cached |

## Project structure

```
app/
  main.py      FastAPI app, request logging, error handling
  agent.py     Agent loop and system prompt
  tools.py     Tool schemas, argument validation, caching
  db.py        SQLite schema, sample data, queries
  cache.py     TTL + LRU cache
  config.py    Settings from environment variables
  static/      Chat UI (plain HTML/JS)
tests/         pytest suite (no API key needed)
```

All people and data are fictional sample data.
