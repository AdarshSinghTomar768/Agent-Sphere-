from langchain_groq import ChatGroq

from src.config import GROQ_API_KEY


llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model_name="llama-3.3-70b-versatile",
    temperature=0
)


def supervisor_decision(question, history):

    question_lower = question.lower()

    references = [
        "it",
        "this",
        "that",
        "second",
        "first",
        "above",
        "previous"
    ]

    # If no history exists, do not rewrite
    if len(history) == 0:
        return question

    # Only rewrite if a reference exists
    if not any(ref in question_lower for ref in references):
        return question

    history_text = "\n".join(history)

    prompt = f"""
You are a conversation supervisor.

Your ONLY job is to resolve references using
the provided conversation history.

Rules:

1. Never invent information.
2. Never introduce new topics.
3. Never use external knowledge.
4. If the question is already clear,
   return it unchanged.
5. Only resolve references such as:
   - it
   - this
   - that
   - second one
   - first project

Conversation History:
{history_text}

Current Question:
{question}

Return ONLY the rewritten standalone question.
"""

    response = llm.invoke(prompt)

    return response.content.strip()