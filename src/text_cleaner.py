import re


def clean_text(text: str) -> str:
    """
    Clean a single text string.
    """

    if not text:
        return ""

    # Normalize line endings
    text = text.replace("\r", "\n")

    # Remove multiple spaces
    text = re.sub(
        r"[ \t]+",
        " ",
        text
    )

    # Remove excessive blank lines
    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text
    )

    # Remove spaces around newlines
    text = re.sub(
        r" *\n *",
        "\n",
        text
    )

    # Fix words broken across lines
    text = re.sub(
        r"-\n",
        "",
        text
    )

    return text.strip()


def clean_documents(documents):
    """
    Clean every LangChain Document.
    """

    cleaned_documents = []

    for doc in documents:

        doc.page_content = clean_text(
            doc.page_content
        )

        cleaned_documents.append(doc)

    return cleaned_documents