import json
from types import SimpleNamespace

import pytest

from app import db


@pytest.fixture
def db_path(tmp_path):
    path = tmp_path / "test.db"
    db.init_db(path)
    return path


def tool_call(call_id: str, name: str, arguments: dict | str):
    args = arguments if isinstance(arguments, str) else json.dumps(arguments)
    return SimpleNamespace(id=call_id, function=SimpleNamespace(name=name, arguments=args))


def llm_message(content: str | None = None, tool_calls=None):
    """Shape of a Groq chat completion response, with just the fields the agent reads."""
    message = SimpleNamespace(content=content, tool_calls=tool_calls)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class FakeGroq:
    """Stands in for the Groq client: returns scripted responses and records each request."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        # Copy the messages, since the agent keeps appending to the same list.
        self.requests.append({**kwargs, "messages": list(kwargs["messages"])})
        return self.responses.pop(0)
