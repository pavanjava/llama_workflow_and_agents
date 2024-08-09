from llama_index.core.workflow import Event


class QuarterlyResponseEvent(Event):
    response: str


class QuarterlySummaryEvent(Event):
    query: str
    response: str
    summary: str


class AnnualSummaryEvent(Event):
    final_summary: str
