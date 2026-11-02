
from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


BASE_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    # Application settings
    app_name: str = "Enterprise IT Support Agentic RAG Copilot"
    app_env: str = "development"

    # Groq LLM configuration
    groq_api_key: str = ""
    groq_model: str = "llama-3.3-70b-versatile"

    # Tavily web search
    tavily_api_key: str = ""
    web_fallback_enabled: bool = False

    # Pinecone vector database
    pinecone_api_key: str = ""
    pinecone_index_name: str = "fde-it-support-minilm"
    pinecone_namespace: str = "company-it-kb"
    pinecone_cloud: str = "aws"
    pinecone_region: str = "us-east-1"

    # Embedding model configuration
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_cache_dir: str = "/tmp/fastembed-cache"

    # RAG pipeline configuration
    top_k: int = Field(default=4, ge=1, le=20)
    max_retries: int = Field(default=1, ge=0, le=3)

    # Admin security
    admin_api_key: str = ""

    # Project storage paths
    audit_db_path: str = str(BASE_DIR / "data" / "audit.db")
    upload_dir: str = str(BASE_DIR / "uploads")
    sample_kb_dir: str = str(BASE_DIR / "data" / "sample_kb")

    # Load variables from .env
    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        extra="ignore"
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
