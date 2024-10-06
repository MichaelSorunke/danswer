from collections.abc import Callable
from collections.abc import Iterator
from typing import Any
from typing import cast

from langchain.schema.messages import BaseMessage
from langchain_core.messages import AIMessageChunk

from danswer.chat.models import AnswerQuestionPossibleReturn
from danswer.chat.models import CitationInfo
from danswer.chat.models import DanswerAnswerPiece
from danswer.chat.models import LlmDoc
from danswer.chat.models import StreamStopInfo
from danswer.chat.models import StreamStopReason
from danswer.configs.chat_configs import QA_PROMPT_OVERRIDE
from danswer.file_store.utils import InMemoryChatFile
from danswer.llm.answering.llm_response_handler import LLMCall
from danswer.llm.answering.llm_response_handler import LLMResponseHandler
from danswer.llm.answering.llm_response_handler import LLMResponseHandlerManager
from danswer.llm.answering.models import AnswerStyleConfig
from danswer.llm.answering.models import PreviousMessage
from danswer.llm.answering.models import PromptConfig
from danswer.llm.answering.models import StreamProcessor
from danswer.llm.answering.prompts.build import AnswerPromptBuilder
from danswer.llm.answering.prompts.build import default_build_system_message
from danswer.llm.answering.prompts.build import default_build_user_message
from danswer.llm.answering.stream_processing.citation_processing import (
    build_citation_processor,
)
from danswer.llm.answering.stream_processing.citation_response_handler import (
    CitationResponseHandler,
)
from danswer.llm.answering.stream_processing.quotes_processing import (
    build_quotes_processor,
)
from danswer.llm.answering.stream_processing.utils import DocumentIdOrderMapping
from danswer.llm.answering.stream_processing.utils import map_document_id_order
from danswer.llm.answering.tool.tool_response_handler import ToolResponseHandler
from danswer.llm.interfaces import LLM
from danswer.llm.interfaces import ToolChoiceOptions
from danswer.natural_language_processing.utils import get_tokenizer
from danswer.tools.force import ForceUseTool
from danswer.tools.search.search_tool import FINAL_CONTEXT_DOCUMENTS_ID
from danswer.tools.tool import Tool
from danswer.tools.tool import ToolResponse
from danswer.tools.tool_runner import ToolCallKickoff
from danswer.utils.logger import setup_logger


logger = setup_logger()


def _get_answer_stream_processor(
    context_docs: list[LlmDoc],
    doc_id_to_rank_map: DocumentIdOrderMapping,
    answer_style_configs: AnswerStyleConfig,
) -> StreamProcessor:
    if answer_style_configs.citation_config:
        return build_citation_processor(
            context_docs=context_docs, doc_id_to_rank_map=doc_id_to_rank_map
        )
    if answer_style_configs.quotes_config:
        return build_quotes_processor(
            context_docs=context_docs, is_json_prompt=not (QA_PROMPT_OVERRIDE == "weak")
        )

    raise RuntimeError("Not implemented yet")


AnswerStream = Iterator[AnswerQuestionPossibleReturn | ToolCallKickoff | ToolResponse]


logger = setup_logger()


class Answer:
    def __init__(
        self,
        question: str,
        answer_style_config: AnswerStyleConfig,
        llm: LLM,
        prompt_config: PromptConfig,
        force_use_tool: ForceUseTool,
        # must be the same length as `docs`. If None, all docs are considered "relevant"
        message_history: list[PreviousMessage] | None = None,
        single_message_history: str | None = None,
        # newly passed in files to include as part of this question
        # TODO THIS NEEDS TO BE HANDLED
        latest_query_files: list[InMemoryChatFile] | None = None,
        files: list[InMemoryChatFile] | None = None,
        tools: list[Tool] | None = None,
        # NOTE: for native tool-calling, this is only supported by OpenAI atm,
        #       but we only support them anyways
        # if set to True, then never use the LLMs provided tool-calling functonality
        skip_explicit_tool_calling: bool = False,
        # Returns the full document sections text from the search tool
        return_contexts: bool = False,
        skip_gen_ai_answer_generation: bool = False,
        is_connected: Callable[[], bool] | None = None,
    ) -> None:
        if single_message_history and message_history:
            raise ValueError(
                "Cannot provide both `message_history` and `single_message_history`"
            )

        self.question = question
        self.is_connected: Callable[[], bool] | None = is_connected

        self.latest_query_files = latest_query_files or []
        self.file_id_to_file = {file.file_id: file for file in (files or [])}

        self.tools = tools or []
        self.force_use_tool = force_use_tool

        self.skip_explicit_tool_calling = skip_explicit_tool_calling

        self.message_history = message_history or []
        # used for QA flow where we only want to send a single message
        self.single_message_history = single_message_history

        self.answer_style_config = answer_style_config
        self.prompt_config = prompt_config

        self.llm = llm
        self.llm_tokenizer = get_tokenizer(
            provider_type=llm.config.model_provider,
            model_name=llm.config.model_name,
        )

        self._final_prompt: list[BaseMessage] | None = None

        self._streamed_output: list[str] | None = None
        self._processed_stream: (
            list[AnswerQuestionPossibleReturn | ToolResponse | ToolCallKickoff] | None
        ) = None

        self._return_contexts = return_contexts
        self.skip_gen_ai_answer_generation = skip_gen_ai_answer_generation
        self._is_cancelled = False

    # This method processes the LLM stream and yields the content or stop information
    def _process_llm_stream(
        self,
        prompt: Any,
        tools: list[dict] | None = None,
        tool_choice: ToolChoiceOptions | None = None,
    ) -> Iterator[str | StreamStopInfo]:
        for message in self.llm.stream(
            prompt=prompt, tools=tools, tool_choice=tool_choice
        ):
            if isinstance(message, AIMessageChunk):
                if message.content:
                    if self.is_cancelled:
                        return StreamStopInfo(stop_reason=StreamStopReason.CANCELLED)
                    yield cast(str, message.content)

            if (
                message.additional_kwargs.get("usage_metadata", {}).get("stop")
                == "length"
            ):
                yield StreamStopInfo(stop_reason=StreamStopReason.CONTEXT_LENGTH)

    def _get_tools_list(self) -> list[Tool]:
        if not self.force_use_tool.force_use:
            return self.tools

        tool = next(
            (t for t in self.tools if t.name == self.force_use_tool.tool_name), None
        )
        if tool is None:
            raise RuntimeError(f"Tool '{self.force_use_tool.tool_name}' not found")

        logger.info(
            f"Forcefully using tool='{tool.name}'"
            + (
                f" with args='{self.force_use_tool.args}'"
                if self.force_use_tool.args is not None
                else ""
            )
        )
        return [tool]

    def _get_search_result(self, llm_call: LLMCall) -> list[LlmDoc] | None:
        if not llm_call.pre_call_yields:
            return None

        for yield_item in llm_call.pre_call_yields:
            if (
                isinstance(yield_item, ToolResponse)
                and yield_item.id == FINAL_CONTEXT_DOCUMENTS_ID
            ):
                return cast(list[LlmDoc], yield_item.response)

        return None

    def _get_response(self, llm_calls: list[LLMCall]) -> AnswerStream:
        current_llm_call = llm_calls[-1]

        stream = self.llm.stream(
            prompt=current_llm_call.prompt,
            tools=[tool.tool_definition() for tool in current_llm_call.tools] or None,
            tool_choice=(
                "required"
                if current_llm_call.tools and current_llm_call.force_use_tool.force_use
                else None
            ),
        )

        handlers: list[LLMResponseHandler] = []
        tool_call_handler = ToolResponseHandler(current_llm_call.tools)
        handlers.append(tool_call_handler)

        search_result = self._get_search_result(current_llm_call) or []
        citation_response_handler = CitationResponseHandler(
            context_docs=search_result,
            doc_id_to_rank_map=map_document_id_order(search_result),
        )
        handlers.append(citation_response_handler)

        response_handler_manager = LLMResponseHandlerManager(handlers)
        yield from response_handler_manager.handle_llm_response(stream)

        new_llm_call = response_handler_manager.finish(current_llm_call)
        if new_llm_call:
            yield from iter(new_llm_call.pre_call_yields)
            yield from self._get_response(llm_calls + [new_llm_call])

    @property
    def processed_streamed_output(self) -> AnswerStream:
        if self._processed_stream is not None:
            yield from self._processed_stream
            return

        prompt_builder = AnswerPromptBuilder(self.message_history, self.llm.config)
        prompt_builder.update_system_prompt(
            default_build_system_message(self.prompt_config)
        )
        prompt_builder.update_user_prompt(
            default_build_user_message(
                self.question, self.prompt_config, self.latest_query_files
            )
        )
        llm_call = LLMCall(
            prompt=prompt_builder.build(),
            tools=self._get_tools_list(),
            force_use_tool=self.force_use_tool,
            files=self.latest_query_files,
            pre_call_yields=[],
        )

        processed_stream = []
        for processed_packet in self._get_response([llm_call]):
            processed_stream.append(processed_packet)
            yield processed_packet

        self._processed_stream = processed_stream

    @property
    def llm_answer(self) -> str:
        answer = ""
        for packet in self.processed_streamed_output:
            if isinstance(packet, DanswerAnswerPiece) and packet.answer_piece:
                answer += packet.answer_piece

        return answer

    @property
    def citations(self) -> list[CitationInfo]:
        citations: list[CitationInfo] = []
        for packet in self.processed_streamed_output:
            if isinstance(packet, CitationInfo):
                citations.append(packet)

        return citations

    @property
    def is_cancelled(self) -> bool:
        if self._is_cancelled:
            return True

        if self.is_connected is not None:
            if not self.is_connected():
                logger.debug("Answer stream has been cancelled")
            self._is_cancelled = not self.is_connected()

        return self._is_cancelled
