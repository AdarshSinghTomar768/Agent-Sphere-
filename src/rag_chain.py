from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate

from src.config import GROQ_API_KEY
from src.retriever import get_retriever

from src.query_rewriter import rewrite_query
from src.fact_checker import verify_answer


def get_rag_chain():

    llm = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name="llama-3.3-70b-versatile",
        temperature=0
    )

    retriever = get_retriever()

    prompt = ChatPromptTemplate.from_template(
        """
You are a helpful AI assistant.

Answer ONLY from the provided context.

If the answer is not present in the context,
say:

"I don't know based on the provided documents."

Context:
{context}

Question:
{question}

Answer:
"""
    )

    def ask(question):

        rewritten_question = rewrite_query(question)

        docs = retriever.invoke(rewritten_question)

        context = "\n\n".join(
            [doc.page_content for doc in docs]
        )

        chain = prompt | llm

        response = chain.invoke({
            "context": context,
            "question": question
        })

        verification = verify_answer(
            question,
            response.content,
            context
        )

        confidence = "Low"

        if len(docs) == 3:
            confidence = "High"
        elif len(docs) == 2:
            confidence = "Medium"

        return {
            "answer": response.content,
            "documents": docs,
            "confidence": confidence,
            "rewritten_query": rewritten_question,
            "verification": verification
        }

    return ask