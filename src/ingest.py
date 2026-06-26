import os

from src.config import DATA_PATH

from src.pdf_loader import load_pdf
from src.ocr import ocr_pdf
from src.text_cleaner import clean_documents
from src.chunker import split_documents
from src.vector_store import create_vectorstore


def load_documents():

    documents = []

    if not os.path.exists(DATA_PATH):
        raise Exception(
            f"Data folder not found: {DATA_PATH}"
        )

    for file in os.listdir(DATA_PATH):

        if not file.endswith(".pdf"):
            continue

        pdf_path = os.path.join(
            DATA_PATH,
            file
        )

        print(f"\nLoading: {file}")

        try:

            # PDF Loader
            docs, has_text = load_pdf(pdf_path)

            # If scanned PDF -> OCR
            if not has_text:

                print(
                    "Scanned PDF detected. Switching to OCR..."
                )

                docs = ocr_pdf(pdf_path)

        except Exception as e:

            print(
                f"PDF Loader failed ({e}). Using OCR..."
            )

            docs = ocr_pdf(pdf_path)

        docs = clean_documents(docs)

        documents.extend(docs)

    return documents


if __name__ == "__main__":

    print("Loading PDFs...")

    documents = load_documents()

    print(
        f"\nLoaded {len(documents)} pages"
    )

    if len(documents) == 0:

        raise Exception(
            "No PDFs found."
        )

    chunks = split_documents(
        documents
    )

    create_vectorstore(
        chunks
    )