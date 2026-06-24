from langchain_groq import ChatGroq

from src.config import GROQ_API_KEY


def verify_answer(question, answer, context):

    if not context.strip():

        return "UNSUPPORTED"

    llm = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name="llama-3.3-70b-versatile",
        temperature=0
    )

    prompt = f"""
You are a fact verification agent.

Question:
{question}

Retrieved Context:
{context}

Generated Answer:
{answer}

Rules:

1. If the answer is clearly based on the context,
   return SUPPORTED.

2. The answer does NOT need to match
   word-for-word.

3. Summaries and paraphrases are allowed.

4. Return UNSUPPORTED only if:
   - the answer introduces facts not found
     in the context
   - the answer contradicts the context
   - the context contains insufficient
     information

Return ONLY:

SUPPORTED

or

UNSUPPORTED
"""

    result = llm.invoke(prompt)

    verification = (
        result.content.strip().upper()
    )

    print(
        "FACT CHECK RESULT:",
        verification
    )

    if "SUPPORTED" in verification:
        return "SUPPORTED"

    return "UNSUPPORTED"