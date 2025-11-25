from pathlib import Path
from sentence_transformers import SentenceTransformer
from backend.app.services.vector_store_service import VectorStoreService
from backend.app.services.rag_service import RAGService
from backend.app.services.script_generation_service import ScriptGenerationService
from backend.app.deps import StubLLMClient
from backend.app.models import TestCase


def test_script_generation_basic():
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    vs = VectorStoreService()
    llm = StubLLMClient()
    rag = RAGService(vs, embedder, llm)

    sg = ScriptGenerationService(rag, Path("docs/checkout.html"))
    tc = TestCase(
        test_id="TC-001",
        feature="Discount Code",
        scenario="Apply valid discount",
        preconditions=[],
        steps=[],
        expected_result="Total reduced by 15%",
        grounded_in=["product_specs.md"],
    )

    script = sg.generate_script(tc)
    assert "selenium" in script.lower()
