"""FastAPI app: the /api/chat endpoint, a health check, and the chat web page."""

import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Literal

import groq
from fastapi import FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from app import db
from app.agent import AgentStepLimitError, build_agent
from app.config import PROJECT_ROOT, Settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("app")

STATIC_DIR = PROJECT_ROOT / "app" / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = Settings.from_env()
    db.init_db(settings.db_path)
    app.state.settings = settings
    # Without a key the UI still loads; /api/chat explains what's missing.
    app.state.agent = build_agent(settings) if settings.groq_api_key else None
    if app.state.agent is None:
        logger.warning("GROQ_API_KEY is not set; /api/chat will return 503.")
    yield


app = FastAPI(title="Employee Support Agent", lifespan=lifespan)


class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=4000)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    history: list[ChatTurn] = Field(default_factory=list, max_length=20)


class ToolCall(BaseModel):
    name: str
    arguments: dict


class ChatResponse(BaseModel):
    answer: str
    tool_calls: list[ToolCall]
    latency_ms: int


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Give each request an id and log its status and latency."""
    request_id = request.headers.get("x-request-id") or uuid.uuid4().hex[:8]
    start = time.perf_counter()
    response = await call_next(request)
    response.headers["x-request-id"] = request_id
    logger.info("request id=%s %s %s status=%d ms=%.0f", request_id, request.method,
                request.url.path, response.status_code, (time.perf_counter() - start) * 1000)
    return response


@app.post("/api/chat", response_model=ChatResponse)
async def chat(body: ChatRequest, request: Request) -> ChatResponse:
    agent = request.app.state.agent
    if agent is None:
        raise HTTPException(503, "The server has no GROQ_API_KEY configured.")

    start = time.perf_counter()
    history = [turn.model_dump() for turn in body.history]
    try:
        # The Groq client is synchronous, so run it in a worker thread.
        reply = await run_in_threadpool(agent.chat, body.message, history)
    except groq.RateLimitError:
        raise HTTPException(429, "The AI service is busy. Please try again in a minute.")
    except groq.APITimeoutError:
        raise HTTPException(504, "The AI service took too long to respond. Please try again.")
    except groq.APIConnectionError:
        raise HTTPException(503, "Could not reach the AI service. Please try again.")
    except groq.AuthenticationError:
        logger.error("Groq rejected the API key.")
        raise HTTPException(503, "The server's AI credentials are invalid.")
    except AgentStepLimitError:
        raise HTTPException(502, "I couldn't work that out. Try rephrasing your question.")
    except groq.APIStatusError as e:
        logger.error("Groq error status=%s: %s", e.status_code, e.message)
        raise HTTPException(502, "The AI service returned an error. Please try again.")

    return ChatResponse(answer=reply.answer, tool_calls=reply.tool_calls,
                        latency_ms=round((time.perf_counter() - start) * 1000))


@app.get("/api/health")
def health(request: Request) -> dict:
    return {"status": "ok", "llm_configured": request.app.state.agent is not None,
            "model": request.app.state.settings.model}


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")
