from typing import Any
from llama_index.core.callbacks import CallbackManager, LlamaDebugHandler
from llama_index.core import Settings, PromptTemplate
from llama_index.core.llms.llm import LLM
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.workflow import (
    Workflow,
    Context,
    StartEvent,
    StopEvent,
    step
)
from workflows.workflow_events import QuarterlyResponseEvent, QuarterlySummaryEvent
from workflows.core.financial_analyser_core import FinancialAnalyserCore
import logging

logging.basicConfig(level=logging.INFO)


class Q4FinancialAnalyser(Workflow):
    def __init__(
            self,
            *args: Any,
            llm: LLM | None = None,
            **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.llm = llm or Ollama(model='llama3.1', request_timeout=300)
        self.memory = ChatMemoryBuffer.from_defaults(llm=llm)
        llama_debug = LlamaDebugHandler(print_trace_on_end=True)
        callback_manager = CallbackManager([llama_debug])
        Settings.embed_model = OllamaEmbedding(model_name='all-minilm:33m')
        Settings.callback_manager = callback_manager
        self.file_name = 'OpenText-Reports-Q4-F-2024-Results.pdf'

    @step(pass_context=True)
    async def pre_process(self, ctx: Context, ev: StartEvent) -> QuarterlyResponseEvent:
        try:
            user_query = ev.get("user_query")
            fa = FinancialAnalyserCore(financial_report_file=self.file_name)
            ctx.data['user_query'] = user_query
            response = fa.retriever_query_engine().query(user_query)
            logging.info(f'response from llm: {str(response)}')
            return QuarterlyResponseEvent(response=str(response))
        except Exception as e:
            logging.error(str(e))

    @step(pass_context=True)
    async def prepare_summary(self, ctx: Context, ev: QuarterlyResponseEvent) -> QuarterlySummaryEvent:
        try:
            # get chat context and response
            current_query = ctx.data.get("user_query", [])
            current_context = ev.response
            prompt_tmpl_str = (
                "---------------------\n"
                f"{current_context}\n"
                "---------------------\n"
                "Query: Given the above context, summarize the financial report with Key Highlights, Key Adjustments "
                "and important Performance Indicators along with revenue numbers\n"
                "Answer: "
            )
            prompt_tmpl = PromptTemplate(prompt_tmpl_str)
            summary_response = await self.llm.acomplete(prompt_tmpl_str)
            return QuarterlySummaryEvent(summary=str(summary_response), response=str(current_context), query=str(current_query))
        except Exception as e:
            logging.error(str(e))

    @step(pass_context=True)
    async def save_summary(self, ctx: Context, ev: QuarterlySummaryEvent) -> StopEvent:
        try:
            current_query = ctx.data.get('user_query')
            current_response = ev.response
            current_summary = ev.summary

            with open(f'./{self.file_name.strip(".pdf")}.md', mode='w') as script:
                script.write(f'user_query : {current_query}\n')
                script.write(f'agent_response : {current_response}\n')
                script.write(f'summary : {current_summary}\n')
            return StopEvent(result=str(current_summary))
        except Exception as e:
            logging.error(str(e))
