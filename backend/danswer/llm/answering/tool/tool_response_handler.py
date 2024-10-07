from collections.abc import Generator

from langchain_core.messages import AIMessageChunk
from langchain_core.messages import BaseMessage
from langchain_core.messages import ToolCall

from danswer.llm.answering.llm_response_handler import LLMCall
from danswer.llm.answering.llm_response_handler import LLMResponseHandler
from danswer.llm.answering.llm_response_handler import ResponsePart
from danswer.tools.force import ForceUseTool
from danswer.tools.message import build_tool_message
from danswer.tools.message import ToolCallSummary
from danswer.tools.models import ToolCallFinalResult
from danswer.tools.models import ToolCallKickoff
from danswer.tools.models import ToolResponse
from danswer.tools.tool import Tool
from danswer.tools.tool_runner import ToolRunner
from danswer.utils.logger import setup_logger


logger = setup_logger()


class ToolResponseHandler(LLMResponseHandler):
    def __init__(self, tools: list[Tool]):
        self.tools = tools

        self.tool_call_chunk: AIMessageChunk | None = None
        self.tool_call_requests: list[ToolCall] = []

        self.tool_runner: ToolRunner | None = None
        self.tool_call_summary: ToolCallSummary | None = None

        self.tool_kickoff: ToolCallKickoff | None = None
        self.tool_responses: list[ToolResponse] = []
        self.tool_final_result: ToolCallFinalResult | None = None

    def _handle_tool_call(self) -> Generator[ResponsePart, None, None]:
        if not self.tool_call_chunk or not self.tool_call_chunk.tool_calls:
            return

        self.tool_call_requests = self.tool_call_chunk.tool_calls

        selected_tool: Tool | None = None
        selected_tool_call_request: ToolCall | None = None
        for tool_call_request in self.tool_call_requests:
            known_tools_by_name = [
                tool for tool in self.tools if tool.name == tool_call_request["name"]
            ]

            if not known_tools_by_name:
                logger.error(
                    "Tool call requested with unknown name field. \n"
                    f"self.tools: {self.tools}"
                    f"tool_call_request: {tool_call_request}"
                )
                continue
            else:
                selected_tool = known_tools_by_name[0]
                selected_tool_call_request = tool_call_request

            if selected_tool and selected_tool_call_request:
                break

        if not selected_tool or not selected_tool_call_request:
            return

        self.tool_runner = ToolRunner(selected_tool, selected_tool_call_request["args"])
        self.tool_call_summary = ToolCallSummary(
            tool_call_request=self.tool_call_chunk,
            tool_call_result=build_tool_message(
                tool_call_request, self.tool_runner.tool_message_content()
            ),
        )

        self.tool_kickoff = self.tool_runner.kickoff()
        yield self.tool_kickoff

        for response in self.tool_runner.tool_responses():
            self.tool_responses.append(response)
            yield response

        self.tool_final_result = self.tool_runner.tool_final_result()
        yield self.tool_final_result

    def handle_response_part(
        self,
        response_item: BaseMessage | None,
        previous_response_items: list[BaseMessage],
    ) -> Generator[ResponsePart, None, None]:
        if response_item is None:
            yield from self._handle_tool_call()

        if isinstance(response_item, AIMessageChunk) and (
            response_item.tool_call_chunks or response_item.tool_calls
        ):
            if self.tool_call_chunk is None:
                self.tool_call_chunk = response_item
            else:
                self.tool_call_chunk += response_item  # type: ignore

        return

    def finish(self, current_llm_call: LLMCall) -> LLMCall | None:
        if (
            self.tool_runner is None
            or self.tool_call_summary is None
            or self.tool_kickoff is None
            or self.tool_final_result is None
        ):
            return None

        tool_runner = self.tool_runner
        new_prompt_builder = tool_runner.tool.build_next_prompt(
            prompt_builder=current_llm_call.prompt_builder,
            tool_call_summary=self.tool_call_summary,
            tool_responses=self.tool_responses,
            using_tool_calling_llm=current_llm_call.using_tool_calling_llm,
        )
        return LLMCall(
            prompt_builder=new_prompt_builder,
            tools=[],  # for now, only allow one tool call per response
            force_use_tool=ForceUseTool(
                force_use=False,
                tool_name="",
                args=None,
            ),
            files=current_llm_call.files,
            using_tool_calling_llm=current_llm_call.using_tool_calling_llm,
            tool_call_info=[
                self.tool_kickoff,
                *self.tool_responses,
                self.tool_final_result,
            ],
        )
