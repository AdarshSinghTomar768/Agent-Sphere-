import os

from src.ingest import (
    load_documents,
    split_documents,
    create_vectorstore
)


def ingest_pdf(pdf_path):

    # PDF already saved by Streamlit.
    # Rebuild vector store using ALL PDFs.

    documents = load_documents()
 
    chunks = split_documents(
        documents
    )

    create_vectorstore(
        chunks
    )

    return len(chunks)