from langchain_community.vectorstores import FAISS

from src.config import VECTORSTORE_PATH
from src.embeddings import embeddings


def get_vectorstore():

    return FAISS.load_local(
        VECTORSTORE_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )