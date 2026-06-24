from langchain_groq import ChatGroq

from src.config import GROQ_API_KEY


def solve_math(question):

    llm = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name="llama-3.3-70b-versatile",
        temperature=0
    )

    response = llm.invoke(
        f"Solve step-by-step:\n{question}"
    )

    return response.content