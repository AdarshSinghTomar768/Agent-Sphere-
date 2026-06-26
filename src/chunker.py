from langchain.text_splitter import RecursiveCharacterTextSplitter


def split_documents(documents):
    """
    Split documents into chunks for embeddings.
    """

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = text_splitter.split_documents(
        documents
    )

    # Remove empty chunks
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

        print(f"\nCHUNK {i+1}")

        print(
            "SOURCE:",
            chunk.metadata.get(
                "source",
                "Unknown"
            )
        )

        print(
            chunk.page_content[:300]
        )

        print("-" * 60)

    return chunks