from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer

from .vector_store_service import VectorStoreService
from ..utils.logging_utils import get_logger

logger = get_logger(__name__)


class RAGService:
    def __init__(self, vector_store: VectorStoreService, embedder: SentenceTransformer, llm_client):
        self.vector_store = vector_store
        self.embedder = embedder
        self.llm_client = llm_client

    def retrieve_context(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        logger.info(f"Retrieving context for query: {query}")
        query_embedding = self.embedder.encode([query], convert_to_numpy=True)[0]
        return self.vector_store.similarity_search(query_embedding, top_k=top_k)

    def build_context_block(self, chunks: List[Dict[str, Any]]) -> str:
        context_str = ""
        for i, item in enumerate(chunks):
            src = item["metadata"].get("source", "unknown")
            context_str += f"[{i + 1}] Source: {src}\n{item['text']}\n\n"
        return context_str

    def generate_from_context(self, prompt: str) -> str:
        logger.info("Sending prompt to LLM client.")
        return self.llm_client.generate(prompt)
