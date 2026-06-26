from langchain_community.vectorstores import FAISS

from src.embeddings import embeddings
from src.config import VECTORSTORE_PATH


def create_vectorstore(chunks):
    """
    Create and save the FAISS vector store.
    """

    print("\nLoading embeddings model...")

    test_embedding = embeddings.embed_query(
        "hello world"
    )

    print(
        f"Embedding dimension: {len(test_embedding)}"
    )

    vectorstore = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings
    )

    vectorstore.save_local(
        VECTORSTORE_PATH
    )

    print(
        "\n✅ Vector Store Created Successfully"
    )