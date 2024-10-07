from collections.abc import Callable
from collections.abc import Iterator

from langchain.schema.messages import BaseMessage
from langchain_core.messages import AIMessageChunk
from langchain_core.messages import ToolCall

from danswer.chat.models import AnswerQuestionPossibleReturn
from danswer.chat.models import CitationInfo
from danswer.chat.models import DanswerAnswerPiece
from danswer.chat.models import LlmDoc
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
from danswer.natural_language_processing.utils import get_tokenizer
from danswer.tools.force import ForceUseTool
from danswer.tools.search.search_tool import SearchTool
from danswer.tools.tool import Tool
from danswer.tools.tool import ToolResponse
from danswer.tools.tool_runner import (
    check_which_tools_should_run_for_non_tool_calling_llm,
)
from danswer.tools.tool_runner import ToolCallKickoff
from danswer.tools.tool_selection import select_single_tool_for_non_tool_calling_llm
from danswer.tools.utils import explicit_tool_calling_supported
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

        self.skip_explicit_tool_calling = skip_explicit_tool_calling
        self.using_tool_calling_llm = explicit_tool_calling_supported(
            self.llm.config.model_provider, self.llm.config.model_name
        )

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

    @classmethod
    def _get_tool_call_for_non_tool_calling_llm(
        cls, llm_call: LLMCall, llm: LLM
    ) -> tuple[Tool, dict] | None:
        if llm_call.force_use_tool.force_use:
            # if we are forcing a tool, we don't need to check which tools to run
            tool = next(
                (
                    t
                    for t in llm_call.tools
                    if t.name == llm_call.force_use_tool.tool_name
                ),
                None,
            )
            if not tool:
                raise RuntimeError(
                    f"Tool '{llm_call.force_use_tool.tool_name}' not found"
                )

            tool_args = (
                llm_call.force_use_tool.args
                if llm_call.force_use_tool.args is not None
                else tool.get_args_for_non_tool_calling_llm(
                    query=llm_call.prompt_builder.get_user_message_content(),
                    history=llm_call.prompt_builder.raw_message_history,
                    llm=llm,
                    force_run=True,
                )
            )

            if tool_args is None:
                raise RuntimeError(f"Tool '{tool.name}' did not return args")

            return (tool, tool_args)
        else:
            tool_options = check_which_tools_should_run_for_non_tool_calling_llm(
                tools=llm_call.tools,
                query=llm_call.prompt_builder.get_user_message_content(),
                history=llm_call.prompt_builder.raw_message_history,
                llm=llm,
            )

            available_tools_and_args = [
                (llm_call.tools[ind], args)
                for ind, args in enumerate(tool_options)
                if args is not None
            ]

            logger.info(
                f"Selecting single tool from tools: {[(tool.name, args) for tool, args in available_tools_and_args]}"
            )

            chosen_tool_and_args = (
                select_single_tool_for_non_tool_calling_llm(
                    tools_and_args=available_tools_and_args,
                    history=llm_call.prompt_builder.raw_message_history,
                    query=llm_call.prompt_builder.get_user_message_content(),
                    llm=llm,
                )
                if available_tools_and_args
                else None
            )

            logger.notice(f"Chosen tool: {chosen_tool_and_args}")
            return chosen_tool_and_args

    def _get_response(self, llm_calls: list[LLMCall]) -> AnswerStream:
        current_llm_call = llm_calls[-1]

        # special pre-logic for non-tool calling LLM case
        if not self.using_tool_calling_llm and current_llm_call.tools:
            chosen_tool_and_args = self._get_tool_call_for_non_tool_calling_llm(
                current_llm_call, self.llm
            )
            if chosen_tool_and_args:
                tool, tool_args = chosen_tool_and_args

                # make a dummy tool handler
                tool_handler = ToolResponseHandler([tool])
                tool_handler.tool_call_chunk = AIMessageChunk(content="")
                tool_handler.tool_call_requests = [
                    ToolCall(name=tool.name, args=tool_args, id=None)
                ]

                response_handler_manager = LLMResponseHandlerManager([tool_handler])
                new_llm_call = response_handler_manager.finish(current_llm_call)
                if new_llm_call:
                    yield from iter(new_llm_call.pre_call_yields)  # type: ignore
                    yield from self._get_response(llm_calls + [new_llm_call])
                else:
                    raise RuntimeError(
                        "Tool call handler did not return a new LLM call"
                    )

                return

        # set up "handlers" to listen to the LLM response stream and
        # feed back the processed results + handle tool call requests
        # + figure out what the next LLM call should be
        handlers: list[LLMResponseHandler] = []
        tool_call_handler = ToolResponseHandler(current_llm_call.tools)
        handlers.append(tool_call_handler)

        search_result = SearchTool.get_search_result(current_llm_call) or []
        citation_response_handler = CitationResponseHandler(
            context_docs=search_result,
            doc_id_to_rank_map=map_document_id_order(search_result),
        )
        handlers.append(citation_response_handler)

        response_handler_manager = LLMResponseHandlerManager(handlers)

        stream = self.llm.stream(
            prompt=current_llm_call.prompt_builder.build(),
            tools=[tool.tool_definition() for tool in current_llm_call.tools] or None,
            tool_choice=(
                "required"
                if current_llm_call.tools and current_llm_call.force_use_tool.force_use
                else None
            ),
        )
        yield from response_handler_manager.handle_llm_response(stream)

        new_llm_call = response_handler_manager.finish(current_llm_call)
        if new_llm_call:
            yield from iter(new_llm_call.pre_call_yields)  # type: ignore
            yield from self._get_response(llm_calls + [new_llm_call])

    @property
    def processed_streamed_output(self) -> AnswerStream:
        if self._processed_stream is not None:
            yield from self._processed_stream
            return

        prompt_builder = AnswerPromptBuilder(
            user_message=default_build_user_message(
                user_query=self.question,
                prompt_config=self.prompt_config,
                files=self.latest_query_files,
            ),
            message_history=self.message_history,
            llm_config=self.llm.config,
            single_message_history=self.single_message_history,
        )
        prompt_builder.update_system_prompt(
            default_build_system_message(self.prompt_config)
        )
        llm_call = LLMCall(
            prompt_builder=prompt_builder,
            tools=self._get_tools_list(),
            force_use_tool=self.force_use_tool,
            files=self.latest_query_files,
            pre_call_yields=[],
            using_tool_calling_llm=self.using_tool_calling_llm,
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
