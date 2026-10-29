"""Settings, read from environment variables (or a local .env file)."""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Settings:
    groq_api_key: str | None
    model: str
    db_path: Path
    request_timeout: float  # seconds per LLM request
    max_retries: int  # automatic retries on rate limits, timeouts and 5xx errors
    max_agent_steps: int  # max LLM calls per user message (stops runaway tool loops)
    cache_ttl: float  # seconds a tool result stays cached

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            groq_api_key=os.getenv("GROQ_API_KEY") or None,
            model=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"),
            db_path=Path(os.getenv("DB_PATH", PROJECT_ROOT / "data" / "company.db")),
            request_timeout=float(os.getenv("REQUEST_TIMEOUT", "30")),
            max_retries=int(os.getenv("MAX_RETRIES", "3")),
            max_agent_steps=int(os.getenv("MAX_AGENT_STEPS", "6")),
            cache_ttl=float(os.getenv("CACHE_TTL", "300")),
        )
