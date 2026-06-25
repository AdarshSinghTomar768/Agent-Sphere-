from langgraph.graph import StateGraph, END

from src.state import AgentState
from src.router import route_question
from src.supervisor import supervisor_decision

from src.query_rewriter import rewrite_query
from src.retriever import get_vectorstore
from src.fact_checker import verify_answer

from src.math_agent import solve_math
from src.logic_agent import solve_logic
from src.statistics_agent import solve_statistics
from src.web_search_agent import web_search
from src.guardrails import safety_check

from langchain_groq import ChatGroq
from src.config import GROQ_API_KEY

import os

# ------------------------------------------
# INITIALIZE COMPONENTS
# ------------------------------------------

llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model_name="llama-3.3-70b-versatile",
    temperature=0
)

# ------------------------------------------
# SUPERVISOR NODE
# ------------------------------------------

def supervisor_node(state: AgentState):

    if not safety_check(state["question"]):

        return {
            "question": state["question"],
            "route": "BLOCKED"
        }

    question = state["question"].lower()

    # Do NOT rewrite PDF/document questions
    if (
        state.get("active_document")
        and (
            "this pdf" in question
            or "this document" in question
            or "summarize pdf" in question
            or "summarize this pdf" in question
            or "what is this pdf about" in question
            or "what is this document about" in question
            or question.strip() == "summarize it"
            or question.strip() == "explain it"
            or question.strip() == "tell me about it"
            or question.strip() == "what is it about"
        )
    ):

        print(
            "Supervisor skipped."
        )

        return {
            "question": state["question"]
        }

    improved_question = supervisor_decision(
        state["question"],
        state.get("chat_history", [])
    )

    print(
        "Supervisor Output:",
        improved_question
    )

    return {
        "question": improved_question
    }

#------------------------------------------
# BLOCKED NODE
#------------------------------------------

def blocked_node(state):

    return {
        "route": "BLOCKED",
        "answer": "❌ Request blocked by safety policy.",
        "confidence": "High",
        "verification": "N/A"
    }


# ------------------------------------------
# ROUTER NODE
# ------------------------------------------

def router_node(state: AgentState):

    # ----------------------------------
    # SAFETY BLOCK
    # ----------------------------------

    if state.get("route") == "BLOCKED":

        print("SAFETY BLOCK TRIGGERED")

        return {
            "route": "BLOCKED"
        }

    # ----------------------------------
    # DOCUMENT MODE
    # ----------------------------------

    active_doc = state.get("active_document")

    if active_doc:

        question = state["question"].lower()

        document_keywords = [
            "who",
            "what",
            "when",
            "where",
            "why",
            "how",
            "resume",
            "profile",
            "skill",
            "skills",
            "experience",
            "education",
            "project",
            "projects",
            "internship",
            "phone",
            "email",
            "contact",
            "summarize",
            "explain",
            "tell me about",
            "this pdf",
            "this document"
        ]

        if any(
            keyword in question
            for keyword in document_keywords
        ):

            print("ACTIVE DOCUMENT FOUND")
            print("ROUTE: RAG")

            return {
                "route": "RAG"
            }
    # ----------------------------------
    # NORMAL ROUTING
    # ----------------------------------

    route = route_question(
        state["question"]
    )

    print(
        "QUESTION:",
        state["question"]
    )

    print(
        "ROUTE:",
        route
    )

    return {
        "route": route
    }
# ------------------------------------------
# QUERY REWRITER NODE
# ------------------------------------------

def rewrite_node(state: AgentState):

    # ----------------------------------
    # DOCUMENT MODE
    # ----------------------------------

    if state.get("active_document"):

        print(
            f"Document Mode -> Rewriter skipped: {state['question']}"
        )

        return {
            "rewritten_query": state["question"],
            "retry_count": state.get(
                "retry_count",
                0
            ) 
        }

    # ----------------------------------
    # NORMAL MODE
    # ----------------------------------

    rewritten = rewrite_query(
        state["question"]
    )

    print(
        f"Original: {state['question']}"
    )

    print(
        f"Rewritten: {rewritten}"
    )

    return {
        "rewritten_query": rewritten,
        "retry_count": state.get(
            "retry_count",
            0
        ) + 1
    }
# ------------------------------------------
# RETRIEVER NODE
# ------------------------------------------

def retrieve_node(state: AgentState):

    vectorstore = get_vectorstore()

    query = state["rewritten_query"]

    active_doc = state.get("active_document")

    print("\n==============================")
    print("ACTIVE DOCUMENT:", active_doc)
    print("==============================")

    # ------------------------------------------
    # DOCUMENT MODE
    # ------------------------------------------

    if active_doc:

        active_name = os.path.basename(
            active_doc
        ).lower().strip()

        print("\nACTIVE DOC:", active_name)

        all_docs = list(
            vectorstore.docstore._dict.values()
        )

        docs = [
            doc
            for doc in all_docs
            if os.path.basename(
                doc.metadata.get(
                    "source",
                    ""
                )
            ).lower().strip() == active_name
        ]

        print(
            "\nFILTERED DOCS:",
            len(docs)
        )

        if len(docs) == 0:

            return {
                "context": "",
                "sources": [],
                "confidence": "Low"
            }

    # ------------------------------------------
    # NORMAL RAG MODE
    # ------------------------------------------

    else:

        docs = vectorstore.similarity_search(
            query,
            k=10
        )

    print("\n===== RETRIEVED DOCUMENTS =====")

    print(
        "DOC COUNT:",
        len(docs)
    )

    for i, doc in enumerate(docs):

        print(f"\nDOCUMENT {i+1}")

        print(
            "SOURCE:",
            doc.metadata.get(
                "source",
                "Unknown Source"
            )
        )

        print(doc.page_content[:500])

        print("-" * 60)

    context = "\n\n".join(
        doc.page_content
        for doc in docs
    )

    sources = list(
        set(
            doc.metadata.get(
                "source",
                "Unknown Source"
            )
            for doc in docs
        )
    )

    confidence = (
        "High"
        if len(docs) >= 3
        else "Medium"
    )

    return {
        "context": context,
        "sources": sources,
        "confidence": confidence
    }

# ------------------------------------------
# APPROVAL NODE
# ------------------------------------------

def approval_node(state: AgentState):

    print("\n===== HUMAN APPROVAL =====")
    print(state["answer"])

    return {
        "approval_required": True,
        "user_approval": state.get(
            "user_approval",
            "NO"
        )
    }

# ------------------------------------------
# ANSWER NODE
# ------------------------------------------

def answer_node(state: AgentState):

    print("\n===== CONTEXT SENT TO LLM =====")

    print(
        state["context"][:2000]
    )

    question = state["question"].lower()

    # ----------------------------------------
    # PDF SUMMARY MODE
    # ----------------------------------------

    if (
        "summarize" in question
        or "what is this pdf about" in question
        or "what is this document about" in question
        or question.strip() == "summarize it"
        or question.strip() == "tell me about it"
        or question.strip() == "explain it"
    ):

        prompt = f"""
            You are an expert document analyst.

            Summarize the document using the provided context.

            Include:
            - Main topic
            - Important information
            - Technologies/tools mentioned
            - Key highlights

            Context:
            {state["context"]}
            """

    # ----------------------------------------
    # QUESTION ANSWERING MODE
    # ----------------------------------------

    else:

        prompt = f"""
        You are answering questions about a document.

        Rules:
        1. Answer ONLY using information present in the context.
        2. If the question asks who a person is, create a short professional summary using:
        - Name
        - Experience
        - Projects
        - Skills
        3. Do not invent information.
        4. If information is missing, say:
        "The document does not contain that information."

        Context:
        {state["context"]}

        Question:
        {state["question"]}
        """

    response = llm.invoke(prompt)

    print("\n===== GENERATED ANSWER =====")

    print(response.content)

    return {
        "answer": response.content
    }
# ------------------------------------------
# WEB SEARCH NODE
# ------------------------------------------

def web_search_node(state: AgentState):

    print("WEB SEARCH NODE")

    web_context = web_search(
        state["question"]
    )

    return {
        "web_context": web_context
    }


# ------------------------------------------
# WEB ANSWER NODE
# ------------------------------------------

def web_answer_node(state: AgentState):

    print("WEB ANSWER NODE")

    prompt = f"""
        Answer using the web search results.

        Question:
        {state['question']}

        Web Search Results:
        {state['web_context']}
        """

    response = llm.invoke(prompt)

    return {
        "answer": response.content,
        "confidence": "Medium",
        "sources": ["Web Search"],
        "context": state["web_context"]
    }

# ------------------------------------------
# MATH AGENT
# ------------------------------------------

def math_node(state: AgentState):

    return {
        "answer": solve_math(state["question"]),
        "confidence": "High",
        "verification": "N/A",
        "sources": []
    }

# ------------------------------------------
# LOGIC AGENT
# ------------------------------------------

def logic_node(state: AgentState):

    return {
        "answer": solve_logic(state["question"]),
        "confidence": "High",
        "verification": "N/A",
        "sources": []
    }

# ------------------------------------------
# STATISTICS AGENT
# ------------------------------------------

def statistics_node(state: AgentState):

    return {
        "answer": solve_statistics(state["question"]),
        "confidence": "High",
        "verification": "N/A",
        "sources": []
    }

# ------------------------------------------
# ROUTING DECISION
# ------------------------------------------

def route_decision(state):

    route = state["route"]

    if route == "BLOCKED":
        return "blocked"

    elif route == "MATH":
        return "math"

    elif route == "LOGIC":
        return "logic"

    elif route == "STATISTICS":
        return "statistics"

    elif route == "GENERAL":
        return "web"

    return "rag"

# ------------------------------------------
# FACT CHECKER NODE
# ------------------------------------------

def verify_node(state: AgentState):

    verification = verify_answer(
        state["question"],
        state["answer"],
        state.get("context", "")
    )

    print(
        "FACT CHECK RESULT:",
        verification
    )

    answer = state["answer"]

    if (
        verification != "SUPPORTED"
        and state.get("retry_count", 0) >= 2
    ):
        answer = (
            "⚠️ Could not fully verify the answer.\n\n"
            + answer
        )

    return {
        "verification": verification,
        "retry_count": state.get(
            "retry_count",
            0
        ),
        "answer": answer
    }


# ------------------------------------------
# VERIFICATION DECISION
# ------------------------------------------

def verification_decision(state):

    retries = state.get(
        "retry_count",
        0
    )

    print(
        "VERIFICATION:",
        state["verification"]
    )

    print(
        "RETRY COUNT:",
        retries
    )

    # Answer supported by context
    if state["verification"] == "SUPPORTED":

        return "supported"

    # Stop infinite retry loops
    if retries >= 2:

        print(
            "Max retries reached. Returning answer with warning."
        )

        return "supported_with_warning"

    # Retry retrieval + answer generation
    return "unsupported"

# ------------------------------------------
# APPROVAL DECISION
# ------------------------------------------

def approval_decision(state):

    approval = state.get(
        "user_approval",
        "NO"
    )

    if approval.upper() == "YES":
        return "approved"

    return "rejected"

# ------------------------------------------
# BUILD GRAPH
# ------------------------------------------

graph = StateGraph(AgentState)

graph.add_node("supervisor", supervisor_node)
graph.add_node("router", router_node)

graph.add_node("rewrite", rewrite_node)
graph.add_node("retrieve", retrieve_node)
graph.add_node("answer", answer_node)
graph.add_node("verify", verify_node)

graph.add_node("math", math_node)
graph.add_node("logic", logic_node)
graph.add_node("statistics", statistics_node)

graph.add_node("web_search", web_search_node)
graph.add_node("web_answer", web_answer_node)
graph.add_node("approval",approval_node)
graph.add_node("blocked",blocked_node)

graph.set_entry_point("supervisor")

graph.add_edge(
    "supervisor",
    "router"
)

graph.add_conditional_edges(
    "router",
    route_decision,
    {
        "blocked": "blocked",
        "math": "math",
        "logic": "logic",
        "statistics": "statistics",
        "web": "web_search",
        "rag": "rewrite"
    }
)

graph.add_edge("rewrite", "retrieve")
graph.add_edge("retrieve", "answer")
graph.add_edge("answer", "verify")
graph.add_edge("blocked",END)

graph.add_conditional_edges(
    "verify",
    verification_decision,
    {
        "supported": "approval",
        "supported_with_warning": "approval",
        "unsupported": "rewrite"
    }
)
graph.add_conditional_edges(
    "approval",
    approval_decision,
    {
        "approved": END,
        "rejected": END
    }
)
graph.add_edge("math", END)
graph.add_edge("logic", END)
graph.add_edge("statistics", END)
graph.add_edge(
    "web_search",
    "web_answer"
)

graph.add_edge(
    "web_answer",
    END
)

agent_graph = graph.compile()