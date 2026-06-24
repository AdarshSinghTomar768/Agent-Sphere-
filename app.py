import streamlit as st
import os
import uuid

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from src.workflow import agent_graph
from src.pdf_ingestor import ingest_pdf

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="AgentSphere",
    page_icon="🤖",
    layout="wide"
)

# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.title("🤖 AgentSphere")
st.caption(
    "Multi-Agent AI System powered by LangGraph + Groq + RAG"
)

# --------------------------------------------------
# PDF UPLOAD
# --------------------------------------------------

st.subheader("📂 Upload Documents")

uploaded_file = st.file_uploader(
    "Upload a PDF",
    type=["pdf"]
)

if uploaded_file is not None:

    if (
        "last_uploaded" not in st.session_state
        or st.session_state.last_uploaded != uploaded_file.name
    ):

        st.session_state.active_document = (
            uploaded_file.name
        )

        st.session_state.last_uploaded = (
            uploaded_file.name
        )

        save_path = (
            f"data/documents/{uploaded_file.name}"
        )

        with open(save_path, "wb") as f:
            f.write(
                uploaded_file.getbuffer()
            )

        with st.spinner(
            "Processing PDF..."
        ):

            chunks = ingest_pdf(
                save_path
            )

        st.success(
            f"PDF ingested successfully. {chunks} chunks added."
        )

        st.rerun()

if "active_document" in st.session_state:

    print(
        "\n=============================="
    )

    print(
        "ACTIVE DOCUMENT IN SESSION:",
        st.session_state.active_document
    )

    print(
        "==============================\n"
    )

    st.info(
        f"📄 Active Document: "
        f"{st.session_state.active_document}"
    )
# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:

    st.header("💬 Chat History")

    st.divider()

    st.subheader("📚 Knowledge Base")

    if os.path.exists("data/documents"):

        files = os.listdir("data/documents")

        if len(files) == 0:
            st.info("No documents uploaded.")

        else:
            for file in files:
                st.write(f"📄 {file}")

    else:
        st.info("Documents folder not found.")

    st.divider()

    if st.button("🗑️ Clear History"):

        st.session_state.messages = []
        st.rerun()

    st.divider()

    if len(st.session_state.messages) == 0:

        st.info("No conversations yet.")

    else:

        for idx, msg in enumerate(
            reversed(st.session_state.messages)
        ):

            st.write(
                f"**Q{len(st.session_state.messages)-idx}:** "
                f"{msg['question']}"
            )


# --------------------------------------------------
# INPUT
# --------------------------------------------------

approval = st.radio(
    "Allow AI to return final answer?",
    ["YES", "NO"],
    horizontal=True
)

question = st.text_input(
    "Ask a question"
)

# --------------------------------------------------
# PROCESS QUESTION
# --------------------------------------------------

if question:

    with st.spinner("Thinking..."):

        # Build conversation history
        chat_history = []

        for item in st.session_state.messages:

            chat_history.append(
                f"User: {item['question']}"
            )

            chat_history.append(
                f"Assistant: {item['answer']}"
            )

        # DEBUG
        print(
            "\n=============================="
        )
        print(
            "ACTIVE DOC SENT TO GRAPH:",
            st.session_state.get(
                "active_document"
            )
        )
        print(
            "QUESTION:",
            question
        )
        print(
            "==============================\n"
        )

        

        print(
            "\n=============================="
        )
        print(
            "ACTIVE DOC SENT TO GRAPH:",
            st.session_state.get(
                "active_document"
            )
        )
        print(
            "QUESTION:",
            question
        )
        print(
            "==============================\n"
        )

        result = agent_graph.invoke(
            {
                "question": question,
                "chat_history": chat_history,
                "active_document": st.session_state.get(
                    "active_document"
                ),
                "user_approval": approval,
                "retry_count": 0
            }
        )
        # Save conversation
        st.session_state.messages.append(
            {
                "question": question,
                "answer": result.get("answer", "")
            }
        )


        with st.expander("🔍 Agent Debug Trace"):

            st.json(
                {
                    "route": result.get("route"),
                    "rewritten_query": result.get(
                        "rewritten_query"
                    ),
                    "confidence": result.get(
                        "confidence"
                    ),
                    "verification": result.get(
                        "verification"
                    ),
                    "retry_count": result.get(
                        "retry_count"
                    )
                }
            )

        # --------------------------------------------------
        # SUPERVISOR OUTPUT
        # --------------------------------------------------

        st.subheader("🧠 Supervisor Question")

        st.info(
            result.get(
                "question",
                question
            )
        )

        # --------------------------------------------------
        # ROUTE
        # --------------------------------------------------

        if "route" in result:

            st.subheader("🎯 Selected Agent")

            route = result["route"]

            if route == "RAG":
                st.info("📄 RAG Agent")

            elif route == "MATH":
                st.info("🧮 Math Agent")

            elif route == "LOGIC":
                st.info("🧠 Logic Agent")

            elif route == "STATISTICS":
                st.info("📊 Statistics Agent")

            else:
                st.info(route)

        # --------------------------------------------------
        # OPTIMIZED QUERY
        # --------------------------------------------------

        if result.get("rewritten_query"):

            st.subheader("🔍 Optimized Query")

            st.info(
                result["rewritten_query"]
            )

        # --------------------------------------------------
        # ANSWER
        # --------------------------------------------------

        st.subheader("💡 Answer")

        st.write(
            result.get(
                "answer",
                "No answer generated."
            )
        )

        # --------------------------------------------------
        # CONFIDENCE
        # --------------------------------------------------

        if "confidence" in result:

            st.subheader("📈 Confidence")

            confidence = result["confidence"]

            if confidence == "High":

                st.success(
                    "🟢 High Confidence"
                )

            elif confidence == "Medium":

                st.warning(
                    "🟡 Medium Confidence"
                )

            else:

                st.error(
                    "🔴 Low Confidence"
                )

        # --------------------------------------------------
        # VERIFICATION
        # --------------------------------------------------

        if "verification" in result:

            st.subheader("✅ Verification")

            if result["verification"] == "SUPPORTED":

                st.success(
                    "Answer verified against retrieved documents."
                )

            elif result["verification"] == "N/A":

                st.info(
                    "Verification not required for this agent."
                )

            else:

                st.error(
                    "Answer could not be fully verified."
                )


        if "retry_count" in result:

            st.subheader("🔄 Retry Attempts")

            st.info(
                str(result["retry_count"])
            )

        # --------------------------------------------------
        # SOURCES
        # --------------------------------------------------

        if result.get("sources"):

            st.subheader("📚 Sources")

            for source in result["sources"]:

                st.write(
                    f"📄 {source}"
                )

# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "AgentSphere • Multi-Agent AI Platform"
)