import os
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

# API Keys
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# Documents Folder
DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "documents"
)

# FAISS Storage Folder
VECTORSTORE_PATH = os.path.join(
    BASE_DIR,
    "vectorstore",
    "faiss_index"
)