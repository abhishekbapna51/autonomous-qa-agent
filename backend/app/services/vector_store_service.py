from typing import List, Dict, Any
import numpy as np


class VectorStoreService:
    def __init__(self):
        self._embeddings: List[np.ndarray] = []
        self._documents: List[Dict[str, Any]] = []

    def add_documents(self, embeddings: List[np.ndarray], texts: List[str],
                      metadatas: List[Dict[str, Any]]) -> None:
        for emb, txt, meta in zip(embeddings, texts, metadatas):
            self._embeddings.append(emb)
            self._documents.append({"text": txt, "metadata": meta})

    def similarity_search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        if not self._embeddings:
            return []

        emb_matrix = np.vstack(self._embeddings)  # (n, d)
        q = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        m_norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True) + 1e-8
        emb_normed = emb_matrix / m_norms
        scores = emb_normed @ q  # cosine

        top_idx = scores.argsort()[::-1][:top_k]
        results = []
        for idx in top_idx:
            doc = self._documents[idx]
            results.append({
                "text": doc["text"],
                "metadata": doc["metadata"],
                "score": float(scores[idx]),
            })
        return results
