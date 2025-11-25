from pydantic_settings import BaseSettings



class Settings(BaseSettings):
    # Paths
    VECTOR_DB_PATH: str = "data/vector_store"
    UPLOAD_DIR: str = "data/uploads"
    CHECKOUT_HTML_PATH: str = "docs/checkout.html"

    # Models
    EMBEDDING_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"

    # LLM configuration (placeholder – plug your own)
    LLM_PROVIDER: str = "stub"  # e.g. "ollama", "openai" later

    class Config:
        env_file = ".env"


settings = Settings()
