import json

import pytest

from app.agent import AgentStepLimitError, EmployeeSupportAgent
from app.tools import ToolExecutor
from tests.conftest import FakeGroq, llm_message, tool_call


def make_agent(db_path, responses, max_steps=6):
    client = FakeGroq(responses)
    return EmployeeSupportAgent(client, ToolExecutor(db_path), "test-model", max_steps), client


def test_answers_directly_without_tools(db_path):
    agent, client = make_agent(db_path, [llm_message("Hello!")])
    reply = agent.chat("hi")
    assert reply.answer == "Hello!"
    assert reply.tool_calls == []
    assert len(client.requests) == 1


def test_runs_tools_then_answers(db_path):
    agent, client = make_agent(db_path, [
        llm_message(tool_calls=[tool_call("c1", "search_employees", {"name": "Daniel"})]),
        llm_message(tool_calls=[tool_call("c2", "get_leave_balance", {"employee_id": 4})]),
        llm_message("Daniel has 9 annual leave days left."),
    ])
    reply = agent.chat("How much leave does Daniel have?")

    assert reply.answer == "Daniel has 9 annual leave days left."
    assert [c["name"] for c in reply.tool_calls] == ["search_employees", "get_leave_balance"]
    assert reply.steps == 3

    # The final request must contain the tool results, linked to their call ids.
    last_messages = client.requests[-1]["messages"]
    tool_msgs = [m for m in last_messages if m["role"] == "tool"]
    assert [m["tool_call_id"] for m in tool_msgs] == ["c1", "c2"]
    assert json.loads(tool_msgs[1]["content"])["leave_balance"]["annual_days"] == 9


def test_multiple_tool_calls_in_one_step(db_path):
    agent, client = make_agent(db_path, [
        llm_message(tool_calls=[tool_call("a", "list_departments", {}),
                                tool_call("b", "search_policies", {"query": "expenses"})]),
        llm_message("Done."),
    ])
    reply = agent.chat("Departments and expense policy?")
    assert len(reply.tool_calls) == 2
    assert sum(m["role"] == "tool" for m in client.requests[-1]["messages"]) == 2


def test_invalid_tool_arguments_are_sent_back_as_errors(db_path):
    agent, client = make_agent(db_path, [
        llm_message(tool_calls=[tool_call("x", "get_employee", "{not json")]),
        llm_message("Sorry, let me try again."),
    ])
    agent.chat("Who is employee 1?")
    tool_msg = next(m for m in client.requests[-1]["messages"] if m["role"] == "tool")
    assert "not valid JSON" in tool_msg["content"]


def test_history_is_included(db_path):
    agent, client = make_agent(db_path, [llm_message("Sure.")])
    history = [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello!"}]
    agent.chat("Thanks", history)
    roles = [m["role"] for m in client.requests[0]["messages"]]
    assert roles == ["system", "user", "assistant", "user"]


def test_stops_after_max_steps(db_path):
    looping = [llm_message(tool_calls=[tool_call(f"c{i}", "list_departments", {})]) for i in range(3)]
    agent, _ = make_agent(db_path, looping, max_steps=3)
    with pytest.raises(AgentStepLimitError):
        agent.chat("loop forever")
