from pathlib import Path
from sentence_transformers import SentenceTransformer
import requests

from .config import settings
from .services.vector_store_service import VectorStoreService
from .services.ingestion_service import IngestionService
from .services.rag_service import RAGService
from .services.test_generation_service import TestGenerationService
from .services.script_generation_service import ScriptGenerationService
from .utils.logging_utils import get_logger

logger = get_logger(__name__)


class OllamaLLMClient:
    """
    LLM client that talks to a local Ollama server.

    Make sure Ollama is running and that you've pulled a model:
      ollama pull llama3.2:1b
    """

    def __init__(self, base_url: str = "http://localhost:11434", model: str = "llama3.2:1b"):
        self.base_url = base_url
        self.model = model

    def generate(self, prompt: str) -> str:
        logger.info(f"Calling Ollama model '{self.model}'...")
        try:
            # timeout can be (connect_timeout, read_timeout)
            resp = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                },
                timeout=(10, 300),  # 10s connect, 300s (5 min) read timeout
            )
            resp.raise_for_status()
            data = resp.json()
            return data.get("response", "")
        except requests.exceptions.ReadTimeout:
            logger.error("Ollama request timed out.")
            # Return something parsable so app doesn't crash
            return "[]"
        except Exception as e:
            logger.error(f"Ollama request failed: {e}")
            return "[]"


# Singletons

logger.info("Loading SentenceTransformer embedder...")
embedder = SentenceTransformer(settings.EMBEDDING_MODEL_NAME)

vector_store = VectorStoreService()
llm_client = OllamaLLMClient()

ingestion_service = IngestionService(vector_store, embedder)
rag_service = RAGService(vector_store, embedder, llm_client)
test_generation_service = TestGenerationService(rag_service)
script_generation_service = ScriptGenerationService(
    rag_service, checkout_html_path=Path(settings.CHECKOUT_HTML_PATH)
)
