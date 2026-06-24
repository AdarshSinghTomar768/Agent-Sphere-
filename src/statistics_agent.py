from langchain_groq import ChatGroq
from src.config import GROQ_API_KEY


def solve_statistics(question):

    llm = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name="llama-3.3-70b-versatile",
        temperature=0
    )

    prompt = f"""
You are an expert in statistics.

Solve step-by-step.

Topics:
- Mean
- Median
- Mode
- Probability
- Standard Deviation
- Ogive
- Data Interpretation

Question:
{question}
"""

    response = llm.invoke(prompt)

    return response.content