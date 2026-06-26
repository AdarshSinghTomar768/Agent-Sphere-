import os
import numpy as np

from pdf2image import convert_from_path
from paddleocr import PaddleOCR
from langchain_core.documents import Document


# Load model once
ocr = PaddleOCR(
    lang="en"
)


def ocr_pdf(pdf_path):
    """
    Extract text from scanned PDFs using PaddleOCR 3.x.
    """

    print(f"\nRunning PaddleOCR on: {os.path.basename(pdf_path)}")

    pages = convert_from_path(pdf_path)

    documents = []

    for page_no, page in enumerate(pages):

        image = np.array(page.convert("RGB"))

        extracted_text = []

        try:

            result = ocr.predict(image)

            for res in result:

                # New PaddleOCR returns recognized text in "rec_texts"
                if isinstance(res, dict):

                    texts = res.get("rec_texts", [])

                    extracted_text.extend(texts)

        except Exception as e:

            print(f"OCR failed on page {page_no + 1}: {e}")

        documents.append(
            Document(
                page_content="\n".join(extracted_text),
                metadata={
                    "source": pdf_path,
                    "page": page_no
                }
            )
        )

    return documents