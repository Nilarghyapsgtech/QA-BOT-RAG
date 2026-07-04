from pydantic_settings import BaseSettings ,SettingsConfigDict
from functools import lru_cache

class Settings(BaseSettings):
    """Application Setting Loaded from Environment Variables."""

    model_config=SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore"
    )

    # OPENAI Configuration
    openai_api_key:str

    # Qdrant Cloud Configuration
    qdrant_url:str
    qdrant_api_key:str

    # Collection Settings
    collection_name:str="rag_documents"

    # Document Processing Settings
    chunk_size:int=1000
    chunk_overlap:int=200

    # Model Configuration
    embedding_model:str='text-embedding-3-small',
    llm_model:str='gpt-4o-mini',
    llm_temperature:int=0

    # Reteieval Settings
    retrieval_k=4

    # Logging
    log_level="INFO"

    # Ragas Evaluation
    enable_ragas_evaluation:bool=True
    ragas_timeout_in_seconds:int=30
    ragas_log_results:bool=True
    ragas_llm_model: str | None = None  # Defaults to llm_model if not set
    ragas_llm_temperature: float | None = None  # Defaults to llm_temperature if not set
    ragas_embedding_model: str | None = None  # Defaults to embedding_model if not set


    # API Settings
    api_host:str="0.0.0.0"
    api_port:int=8000

    # Application Info
    app_name:str="RAG Q&A System"
    app_Version="0.0.0"

    
@lru_cache
def get_settings()->Settings:
    """Get cached settings instance."""
    return Settings()






