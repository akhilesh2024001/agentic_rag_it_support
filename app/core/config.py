from functools import lru_cache
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    app_name: str = "Enterprise IT Support Agentic RAG Copilot"
    app_env: str = "development"

    # Groq LLM
    groq_api_key: str = ""
    groq_model: str = "llama-3.3-70b-versatile"

    # External web search
    tavily_api_key: str = ""

    # Pinecone vector database
    pinecone_api_key: str = ""
    pinecone_index_name: str = "fde-it-support-minilm"
    pinecone_namespace: str = "company-it-kb"

    # Local HuggingFace embeddings
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # RAG configuration
    top_k: int = 4
    max_retries: int = 1

    # Application configuration
    admin_api_key: str = "change-me"
    audit_db_path: str = str(BASE_DIR / "data" / "audit.db")
    upload_dir: str = str(BASE_DIR / "uploads")
    sample_kb_dir: str = str(BASE_DIR / "static" / "data" / "sample_kb")

    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        extra="ignore"
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()