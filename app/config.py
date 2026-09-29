import os

from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-oss-120b")
# smaller model we try if the main one fails
FALLBACK_MODEL = os.getenv("FALLBACK_MODEL", "openai/gpt-oss-20b")

CHROMA_DIR = os.getenv("CHROMA_DIR", "chroma_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "documents")

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "600"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))
TOP_K = int(os.getenv("TOP_K", "5"))
# cosine distance (0 = same, 2 = opposite). Chunks further than this are ignored.
MAX_DISTANCE = float(os.getenv("MAX_DISTANCE", "0.75"))

MAX_FILE_MB = int(os.getenv("MAX_FILE_MB", "20"))
