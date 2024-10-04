from collections.abc import Iterator
from typing import cast
from uuid import uuid4

from langchain_core.messages import AIMessageChunk
from langchain_core.messages import BaseMessage

from danswer.chat.models import StreamStopInfo
from danswer.chat.models import StreamStopReason
from danswer.llm.interfaces import LLM
from danswer.tools.force import ForceUseTool
from danswer.tools.tool import Tool
from danswer.tools.tool_runner import ToolRunner
from danswer.utils.logger import setup_logger

logger = setup_logger()


def find_tool_by_name(tool_name: str, tools: list[Tool]) -> Tool | None:
    for tool in tools:
        if tool.name == tool_name:
            return tool
    return None


def handle_force_tool_use(
    force_use_tool: ForceUseTool,
    tools: list[Tool],
    llm: LLM,
    prompt: list[BaseMessage],
) -> ToolRunner:
    """Given a tool and a prompt, force the LLM to use the tool."""
    if not force_use_tool.force_use:
        raise RuntimeError("Force tool use is not enabled")

    tool = find_tool_by_name(force_use_tool.tool_name, tools)
    if tool is None:
        raise RuntimeError(f"Tool '{force_use_tool.tool_name}' not found")

    logger.info(
        f"Forcefully using tool='{tool.name}'"
        + (
            f" with args='{force_use_tool.args}'"
            if force_use_tool.args is not None
            else ""
        )
    )

    if force_use_tool.args:
        tool_call_chunk = AIMessageChunk(
            content="",
        )
        tool_call_chunk.tool_calls = [
            {
                "name": force_use_tool.tool_name,
                "args": force_use_tool.args,
                "id": str(uuid4()),
            }
        ]

        return ToolRunner(tool, force_use_tool.args)

    # let the LLM fill in the args for the tool call
    tool_call_chunk: None | AIMessageChunk = None
    for message in llm.stream(
        prompt=prompt,
        tools=[tool],
        tool_choice="required",
    ):
        if tool_call_chunk is None:
            tool_call_chunk = message
        else:
            tool_call_chunk += message  # type: ignore

    tool_call_requests = cast(AIMessageChunk, tool_call_chunk).tool_calls
    if not tool_call_requests:
        raise RuntimeError(
            "Tool call request not found despite force use "
            "tool being set. This should never happen."
        )

    return ToolRunner(tool, tool_call_requests[0]["args"])


def handle_optional_tool_use(
    tools: list[Tool], prompt: list[BaseMessage], llm: LLM
) -> Iterator[BaseMessage | StreamStopInfo | ToolRunner]:
    tool_call_chunk: AIMessageChunk | None = None
    # if tool calling is supported, first try the raw message
    # to see if we don't need to use any tools
    final_tool_definitions = [tool.tool_definition() for tool in tools]

    output: list[BaseMessage | StreamStopInfo] = []
    for message in llm.stream(
        prompt=prompt,
        tools=final_tool_definitions if final_tool_definitions else None,
    ):
        output.append(message)
        yield message

        if message.additional_kwargs.get("usage_metadata", {}).get("stop") == "length":
            stop_info = StreamStopInfo(stop_reason=StreamStopReason.CONTEXT_LENGTH)
            output.append(stop_info)
            yield stop_info

    tool_call_chunk: AIMessageChunk | None = None
    for message in output:
        if isinstance(message, AIMessageChunk) and (
            message.tool_call_chunks or message.tool_calls
        ):
            if tool_call_chunk is None:
                tool_call_chunk = message
            else:
                tool_call_chunk += message  # type: ignore

    if not tool_call_chunk or not tool_call_chunk.tool_calls:
        return

    tool_call_requests = tool_call_chunk.tool_calls
    tool = find_tool_by_name(tool_call_requests[0]["name"], tools)
    if tool is None:
        raise RuntimeError(f"Tool '{tool_call_requests[0]['name']}' not found")

    yield ToolRunner(tool, tool_call_requests[0]["args"])
