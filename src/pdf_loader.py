from langchain_community.document_loaders import PyPDFLoader
import re


def load_pdf(pdf_path):
    """
    Loads a PDF using PyPDFLoader.

    Returns
    -------
    documents : List[Document]
    has_text : bool
    """

    try:

        loader = PyPDFLoader(pdf_path)

        documents = loader.load()

        full_text = "\n".join(
            doc.page_content
            for doc in documents
        )

        # Remove whitespace and punctuation
        clean_text = re.sub(
            r"\W+",
            "",
            full_text
        )

        has_text = len(clean_text) > 100

        print("\n==============================")
        print(f"PDF: {pdf_path}")
        print(f"Pages: {len(documents)}")
        print(f"Extracted Characters: {len(clean_text)}")
        print(f"Has Text: {has_text}")
        print("==============================")

        return documents, has_text

    except Exception as e:

        print("\n==============================")
        print(f"Failed to read PDF: {pdf_path}")
        print(e)
        print("==============================")

        # Return empty docs so OCR can be used
        return [], False