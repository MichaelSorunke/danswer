import abc
from collections.abc import Generator
from collections.abc import Iterator

from langchain_core.messages import BaseMessage
from pydantic.v1 import BaseModel as BaseModel__v1

from danswer.chat.models import CitationInfo
from danswer.chat.models import DanswerAnswerPiece
from danswer.chat.models import StreamStopInfo
from danswer.file_store.models import InMemoryChatFile
from danswer.tools.force import ForceUseTool
from danswer.tools.models import ToolCallFinalResult
from danswer.tools.models import ToolCallKickoff
from danswer.tools.models import ToolResponse
from danswer.tools.tool import Tool

ResponsePart = DanswerAnswerPiece | CitationInfo | None


class LLMCall(BaseModel__v1):
    prompt: list[BaseMessage]
    tools: list[Tool]
    force_use_tool: ForceUseTool
    files: list[InMemoryChatFile]
    pre_call_yields: list[
        str | StreamStopInfo | ToolCallKickoff | ToolResponse | ToolCallFinalResult
    ]

    class Config:
        arbitrary_types_allowed = True


class LLMResponseHandler(abc.ABC):
    @abc.abstractmethod
    def handle_response_part(
        self, response_item: BaseMessage, previous_response_items: list[BaseMessage]
    ) -> ResponsePart:
        raise NotImplementedError

    @abc.abstractmethod
    def finish(self, current_llm_call: LLMCall) -> LLMCall | None:
        raise NotImplementedError


class LLMResponseHandlerManager:
    def __init__(self, handlers: list[LLMResponseHandler]):
        self.handlers = handlers

    def handle_llm_response(
        self,
        stream: Iterator[BaseMessage],
    ) -> Generator[ResponsePart, None, None]:
        messages: list[BaseMessage] = []
        for message in stream:
            for handler in self.handlers:
                response = handler.handle_response_part(message, messages)
                if response:
                    yield response

            messages.append(message)

    def finish(self, llm_call: LLMCall) -> LLMCall | None:
        new_llm_call = None
        for handler in self.handlers:
            new_llm_call_temp = handler.finish(llm_call)

            if new_llm_call and new_llm_call_temp:
                raise RuntimeError(
                    "Multiple handlers are trying to add a new LLM call, this is not allowed."
                )

            if new_llm_call_temp:
                new_llm_call = new_llm_call_temp

        return new_llm_call
