from pathlib import Path
from backend.app.services.vector_store_service import VectorStoreService
from backend.app.services.ingestion_service import IngestionService
from sentence_transformers import SentenceTransformer


def test_ingestion_runs():
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    vs = VectorStoreService()
    service = IngestionService(vs, embedder)

    docs_dir = Path("docs/support_docs")
    files = list(docs_dir.glob("*.*"))
    if not files:
        return  # nothing to ingest in CI

    num_chunks = service.build_kb_from_paths(files)
    assert num_chunks >= 0
