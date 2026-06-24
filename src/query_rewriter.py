from langchain_groq import ChatGroq

from src.config import GROQ_API_KEY


def rewrite_query(question):

    llm = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name="llama-3.3-70b-versatile",
        temperature=0
    )

    prompt = f"""
You are a retrieval query rewriting assistant.

Rules:

1. Do NOT invent entities.
2. Do NOT add topics.
3. Do NOT add assumptions.
4. Preserve the meaning exactly.
5. Improve wording only if it helps retrieval.
6. If the question is already good,
   return it unchanged.

Question:
{question}

Return ONLY the rewritten query.
"""

    response = llm.invoke(prompt)

    return response.content.strip()