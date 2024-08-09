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
from workflows.workflow_events import AnnualSummaryEvent
import logging

logging.basicConfig(level=logging.INFO)


class AnnualFinancialAnalyser(Workflow):
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

    @step()
    async def prepare_annual_summary(self, ev: StartEvent) -> AnnualSummaryEvent:
        try:
            current_context = ev.individual_summaries
            prompt_tmpl_str = (
                "---------------------\n"
                f"Q1 Summary: {current_context[0]}\n"
                "---------------------\n"
                f"Q2 Summary: {current_context[1]}\n"
                "----------------------\n"
                f"Q3 Summary: {current_context[2]}\n"
                "----------------------\n"
                f"Q4 Summary: {current_context[3]}\n"
                "----------------------\n"
                "Query: Given the above context, summarize the respective quarterly financial reports "
                "with Key Highlights, Key Adjustments and important Performance Indicators as annual summary report\n"
                "Answer: "
            )
            prompt_tmpl = PromptTemplate(prompt_tmpl_str)
            summary_response = await self.llm.acomplete(prompt_tmpl_str)
            return AnnualSummaryEvent(final_summary=str(summary_response))
        except Exception as e:
            logging.error(str(e))

    @step()
    async def save_annual_summary(self, ev: AnnualSummaryEvent) -> StopEvent:
        try:
            current_summary = ev.final_summary

            with open('./annual_summary.md', mode='w') as script:
                script.write(f'summary : {current_summary}\n')
            return StopEvent(result=str(current_summary))
        except Exception as e:
            logging.error(str(e))
