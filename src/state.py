from typing import TypedDict, List


class AgentState(TypedDict, total=False):

    question: str

    chat_history: List[str]

    route: str

    rewritten_query: str

    context: str

    answer: str

    confidence: str

    verification: str

    sources: List[str]

    active_document: str

    web_context: str

    approval_required: bool

    user_approval: str

    retry_count: int