from langchain_core.messages import BaseMessage

from danswer.chat.models import CitationInfo
from danswer.chat.models import DanswerAnswerPiece
from danswer.chat.models import LlmDoc
from danswer.llm.answering.llm_response_handler import LLMCall
from danswer.llm.answering.llm_response_handler import LLMResponseHandler
from danswer.llm.answering.stream_processing.citation_processing import (
    CitationProcessor,
)
from danswer.llm.answering.stream_processing.utils import DocumentIdOrderMapping


class CitationResponseHandler(LLMResponseHandler):
    def __init__(
        self, context_docs: list[LlmDoc], doc_id_to_rank_map: DocumentIdOrderMapping
    ):
        self.context_docs = context_docs
        self.doc_id_to_rank_map = doc_id_to_rank_map
        self.citation_processor = CitationProcessor(
            context_docs=self.context_docs,
            doc_id_to_rank_map=self.doc_id_to_rank_map,
        )
        self.processed_text = ""
        self.citations: list[CitationInfo] = []

    def handle_response_part(
        self, response_item: BaseMessage, previous_response_items: list[BaseMessage]
    ) -> DanswerAnswerPiece | CitationInfo | None:
        content = (
            response_item.content if isinstance(response_item.content, str) else ""
        )

        # Process the new content through the citation processor
        return self.citation_processor.process_token(content)

    def finish(self, current_llm_call: LLMCall) -> LLMCall | None:
        # Process any remaining content in the citation processor
        if self.citation_processor.curr_segment:
            self.processed_text += self.citation_processor.curr_segment
        return None
