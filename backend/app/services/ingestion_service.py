from typing import List
from pathlib import Path
from sentence_transformers import SentenceTransformer

from .vector_store_service import VectorStoreService
from ..utils import parsers
from ..utils.logging_utils import get_logger

logger = get_logger(__name__)


class IngestionService:
    def __init__(self, vector_store: VectorStoreService, embedder: SentenceTransformer):
        self.vector_store = vector_store
        self.embedder = embedder

    def _load_and_parse_file(self, path: Path) -> str:
        suffix = path.suffix.lower()
        if suffix in [".txt", ".md"]:
            return parsers.parse_text_file(path)
        if suffix == ".json":
            return parsers.parse_json_file(path)
        if suffix == ".html":
            return parsers.parse_html_file(path)
        # default
        return parsers.parse_text_file(path)

    def _simple_chunk(self, text: str, chunk_size: int = 300) -> List[str]:
        words = text.split()
        chunks = []
        for i in range(0, len(words), chunk_size):
            chunks.append(" ".join(words[i:i + chunk_size]))
        return chunks

    def build_kb_from_paths(self, paths: List[Path]) -> int:
        all_chunks: List[str] = []
        all_metadata = []

        for path in paths:
            logger.info(f"Ingesting file: {path}")
            raw_text = self._load_and_parse_file(path)
            chunks = self._simple_chunk(raw_text)
            for idx, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                all_metadata.append({"source": path.name, "chunk_id": idx})

        if not all_chunks:
            logger.warning("No text chunks generated from provided documents.")
            return 0

        embeddings = self.embedder.encode(all_chunks, convert_to_numpy=True)
        self.vector_store.add_documents(embeddings, all_chunks, all_metadata)
        logger.info(f"Added {len(all_chunks)} chunks to vector store.")
        return len(all_chunks)
