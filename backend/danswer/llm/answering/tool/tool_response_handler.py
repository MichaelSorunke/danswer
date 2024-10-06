from langchain_core.messages import AIMessageChunk
from langchain_core.messages import BaseMessage
from langchain_core.messages import ToolCall

from danswer.llm.answering.llm_response_handler import LLMCall
from danswer.llm.answering.llm_response_handler import LLMResponseHandler
from danswer.llm.answering.llm_response_handler import ResponsePart
from danswer.tools.force import ForceUseTool
from danswer.tools.message import build_tool_message
from danswer.tools.message import ToolCallSummary
from danswer.tools.tool import Tool
from danswer.tools.tool_runner import ToolRunner
from danswer.utils.logger import setup_logger


logger = setup_logger()


class ToolResponseHandler(LLMResponseHandler):
    def __init__(self, tools: list[Tool]):
        self.tools = tools
        self.tool_call_chunk: AIMessageChunk | None = None
        self.tool_call_requests: list[ToolCall] = []

    def handle_response_part(
        self, response_item: BaseMessage, previous_response_items: list[BaseMessage]
    ) -> ResponsePart:
        if isinstance(response_item, AIMessageChunk) and (
            response_item.tool_call_chunks or response_item.tool_calls
        ):
            if self.tool_call_chunk is None:
                self.tool_call_chunk = response_item
            else:
                self.tool_call_chunk += response_item  # type: ignore

        if self.tool_call_chunk and self.tool_call_chunk.tool_calls:
            self.tool_call_requests = self.tool_call_chunk.tool_calls

        return None

    def finish(self, current_llm_call: LLMCall) -> LLMCall | None:
        if not self.tool_call_requests or not self.tool_call_chunk:
            return None

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
            return None

        tool_runner = ToolRunner(selected_tool, selected_tool_call_request["args"])
        tool_call_summary = ToolCallSummary(
            tool_call_request=self.tool_call_chunk,
            tool_call_result=build_tool_message(
                tool_call_request, tool_runner.tool_message_content()
            ),
        )

        # TODO: use prompt builder
        new_prompt = current_llm_call.prompt + [
            tool_call_summary.tool_call_request,
            tool_call_summary.tool_call_result,
        ]
        return LLMCall(
            prompt=new_prompt,
            tools=[],  # for now, only allow one tool call per response
            force_use_tool=ForceUseTool(
                force_use=False,
                tool_name="",
                args=None,
            ),
            files=current_llm_call.files,
            pre_call_yields=[
                tool_runner.kickoff(),
                *tool_runner.tool_responses(),
                tool_runner.tool_final_result(),
            ],
        )
