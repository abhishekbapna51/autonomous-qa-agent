from sentence_transformers import SentenceTransformer
from backend.app.services.vector_store_service import VectorStoreService
from backend.app.services.rag_service import RAGService
from backend.app.deps import StubLLMClient


def test_rag_empty():
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    vs = VectorStoreService()
    llm = StubLLMClient()
    rag = RAGService(vs, embedder, llm)

    chunks = rag.retrieve_context("discount code")
    assert chunks == []
