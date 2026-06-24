import os

from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from src.embeddings import embeddings
from src.config import DATA_PATH, VECTORSTORE_PATH


def load_documents():

    documents = []

    if not os.path.exists(DATA_PATH):
        raise Exception(
            f"Data folder not found: {DATA_PATH}"
        )

    for file in os.listdir(DATA_PATH):

        if file.endswith(".pdf"):

            pdf_path = os.path.join(
                DATA_PATH,
                file
            )

            print(f"\nLoading: {file}")

            loader = UnstructuredPDFLoader(
                pdf_path,
                mode="single"
            )
            docs = loader.load()

            for doc in docs:

                print("\n==============================")
                print(
                    "SOURCE:",
                    doc.metadata.get(
                        "source"
                    )
                )

                print(
                    "PAGE:",
                    doc.metadata.get(
                        "page"
                    )
                )

                print("\nTEXT PREVIEW:\n")

                print(
                    doc.page_content[:1000]
                )

                print(
                    "\n=============================="
                )

            documents.extend(docs)

    return documents


def split_documents(documents):

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = text_splitter.split_documents(
        documents
    )

    chunks = [
        chunk
        for chunk in chunks
        if chunk.page_content.strip()
    ]

    print(
        f"\nLoaded docs: {len(documents)}"
    )

    print(
        f"Chunks created: {len(chunks)}"
    )

    print("\n===== CHUNKS =====")

    for i, chunk in enumerate(chunks):

        print(
            f"\nCHUNK {i+1}"
        )

        print(
            "SOURCE:",
            chunk.metadata.get(
                "source"
            )
        )

        print(
            chunk.page_content[:300]
        )

        print("-" * 60)

    if len(chunks) == 0:

        raise Exception(
            "No text chunks found. PDF may be scanned/image-based."
        )

    return chunks


def create_vectorstore(chunks):

    print(
        "\nLoading embeddings model..."
    )

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


if __name__ == "__main__":

    print("Loading PDFs...")

    docs = load_documents()

    print(
        f"\nLoaded {len(docs)} pages"
    )

    if len(docs) == 0:

        raise Exception(
            "No PDFs found in data folder."
        )

    chunks = split_documents(
        docs
    )

    create_vectorstore(
        chunks
    )