import json
from collections.abc import Callable
from datetime import datetime
from typing import cast
from unittest.mock import MagicMock
from unittest.mock import Mock

import pytest
from langchain_core.messages import AIMessageChunk
from langchain_core.messages import HumanMessage
from langchain_core.messages import SystemMessage
from langchain_core.messages import ToolCall
from langchain_core.messages import ToolCallChunk

from danswer.chat.models import DanswerAnswerPiece
from danswer.chat.models import LlmDoc
from danswer.configs.constants import DocumentSource
from danswer.llm.answering.answer import Answer
from danswer.llm.answering.models import AnswerStyleConfig
from danswer.llm.answering.models import CitationConfig
from danswer.llm.answering.models import PromptConfig
from danswer.llm.interfaces import LLM
from danswer.llm.interfaces import LLMConfig
from danswer.tools.force import ForceUseTool
from danswer.tools.models import ToolCallFinalResult
from danswer.tools.models import ToolCallKickoff
from danswer.tools.models import ToolResponse
from danswer.tools.search.search_tool import FINAL_CONTEXT_DOCUMENTS_ID
from danswer.tools.search.search_tool import SearchTool


@pytest.fixture
def mock_llm() -> MagicMock:
    mock_llm_obj = MagicMock()
    mock_llm_obj.config = LLMConfig(
        model_provider="openai",
        model_name="gpt-4o",
        temperature=0.0,
        api_key=None,
        api_base=None,
        api_version=None,
    )
    return mock_llm_obj()


@pytest.fixture
def answer_instance(mock_llm: LLM) -> Answer:
    return Answer(
        question="Test question",
        answer_style_config=AnswerStyleConfig(citation_config=CitationConfig()),
        llm=mock_llm,
        prompt_config=PromptConfig(
            system_prompt="System prompt",
            task_prompt="Task prompt",
            datetime_aware=False,
            include_citations=True,
        ),
        force_use_tool=ForceUseTool(force_use=False, tool_name="", args=None),
    )


def test_basic_answer(answer_instance: Answer) -> None:
    mock_llm = cast(Mock, answer_instance.llm)
    mock_llm.stream.return_value = [
        AIMessageChunk(content="This is a "),
        AIMessageChunk(content="mock answer."),
    ]

    output = list(answer_instance.processed_streamed_output)
    assert len(output) == 2
    assert isinstance(output[0], DanswerAnswerPiece)
    assert isinstance(output[1], DanswerAnswerPiece)

    full_answer = "".join(
        piece.answer_piece
        for piece in output
        if isinstance(piece, DanswerAnswerPiece) and piece.answer_piece is not None
    )
    assert full_answer == "This is a mock answer."

    assert answer_instance.llm_answer == "This is a mock answer."
    assert answer_instance.citations == []

    assert mock_llm.stream.call_count == 1
    mock_llm.stream.assert_called_once_with(
        prompt=[
            SystemMessage(content="System prompt"),
            HumanMessage(content="Task prompt\n\nQUERY:\nTest question"),
        ],
        tools=[],
        tool_choice=None,
    )


def test_answer_with_search_call(answer_instance: Answer) -> None:
    # Mock search results
    mock_search_results: list[LlmDoc] = [
        LlmDoc(
            content="Search result 1",
            source_type=DocumentSource.WEB,
            metadata={"id": "doc1"},
            document_id="doc1",
            blurb="Blurb 1",
            semantic_identifier="Semantic ID 1",
            updated_at=datetime(2023, 1, 1),
            link="https://example.com/doc1",
            source_links={0: "https://example.com/doc1"},
        ),
        LlmDoc(
            content="Search result 2",
            source_type=DocumentSource.WEB,
            metadata={"id": "doc2"},
            document_id="doc2",
            blurb="Blurb 2",
            semantic_identifier="Semantic ID 2",
            updated_at=datetime(2023, 1, 2),
            link="https://example.com/doc2",
            source_links={0: "https://example.com/doc2"},
        ),
    ]
    mock_tool_response = ToolResponse(
        id=FINAL_CONTEXT_DOCUMENTS_ID, response=mock_search_results
    )
    mock_final_results = [
        json.loads(doc.model_dump_json()) for doc in mock_search_results
    ]
    mock_tool_final_response = ToolCallFinalResult(
        tool_name="search",
        tool_args={"query": "test"},
        tool_result=mock_final_results,
    )
    mock_tool_definition: dict = {
        "type": "function",
        "function": {
            "name": "search",
            "description": "Search for information",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query"},
                },
                "required": ["query"],
            },
        },
    }

    # Mock the search tool
    mock_search_tool = MagicMock(spec=SearchTool)
    mock_search_tool.name = "search"
    mock_search_tool.build_tool_message_content.return_value = "search_response"
    mock_search_tool.get_args_for_non_tool_calling_llm.return_value = {}
    mock_search_tool.final_result.return_value = mock_final_results
    mock_search_tool.run.return_value = [mock_tool_response]
    mock_search_tool.tool_definition.return_value = mock_tool_definition
    answer_instance.tools = [mock_search_tool]

    # Set up the LLM mock to return search results and then an answer
    mock_llm = cast(Mock, answer_instance.llm)

    tool_call_chunk = AIMessageChunk(
        content="",
    )
    tool_call_chunk.tool_calls = [
        ToolCall(
            id="search",
            name="search",
            args={"query": "test"},
        )
    ]
    tool_call_chunk.tool_call_chunks = [
        ToolCallChunk(
            id="search",
            name="search",
            args='{"query": "test"}',
            index=0,
        )
    ]
    mock_llm.stream.side_effect = [
        [tool_call_chunk],
        [
            AIMessageChunk(content="Based on the search results, "),
            AIMessageChunk(content="here's the answer: "),
            AIMessageChunk(content="This is the final answer."),
        ],
    ]

    # Process the output
    output = list(answer_instance.processed_streamed_output)

    # Assertions
    print(output)
    assert len(output) == 7
    assert output[0] == DanswerAnswerPiece(answer_piece="")
    assert output[1] == ToolCallKickoff(tool_name="search", tool_args={"query": "test"})
    assert output[2] == ToolResponse(
        id="final_context_documents",
        response=mock_search_results,
    )
    assert output[3] == mock_tool_final_response
    assert output[4] == DanswerAnswerPiece(answer_piece="Based on the search results, ")
    assert output[5] == DanswerAnswerPiece(answer_piece="here's the answer: ")
    assert output[6] == DanswerAnswerPiece(answer_piece="This is the final answer.")

    full_answer = "".join(
        piece.answer_piece
        for piece in output
        if isinstance(piece, DanswerAnswerPiece) and piece.answer_piece is not None
    )
    assert (
        full_answer
        == "Based on the search results, here's the answer: This is the final answer."
    )

    assert answer_instance.llm_answer == full_answer
    assert answer_instance.citations == []

    # Verify LLM calls
    assert mock_llm.stream.call_count == 2
    first_call, second_call = mock_llm.stream.call_args_list

    # First call should include the search tool definition
    assert len(first_call.kwargs["tools"]) == 1
    assert first_call.kwargs["tools"][0] == mock_tool_definition

    # Second call should not include tools (as we're just generating the final answer)
    assert "tools" not in second_call.kwargs or not second_call.kwargs["tools"]

    # Verify that tool_definition was called on the mock_search_tool
    mock_search_tool.tool_definition.assert_called_once()


@pytest.mark.parametrize(
    "is_connected, expected_cancelled",
    [
        (lambda: True, False),
        (lambda: False, True),
    ],
)
def test_is_cancelled(
    answer_instance: Answer, is_connected: Callable[[], bool], expected_cancelled: bool
) -> None:
    answer_instance.is_connected = is_connected
    assert answer_instance.is_cancelled == expected_cancelled
