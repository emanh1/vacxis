"""Public retrieval interfaces, loaded lazily to avoid loading model libraries at import."""
from importlib import import_module

_EXPORTS = {
    "HybridSearcher": ".hybrid_searcher", "convert_sparse_to_qdrant": ".hybrid_searcher",
    "BGEReranker": ".reranker", "VacxisRetriever": ".retriever", "DenseRetriever": ".dense",
    "BM25Retriever": ".bm25", "build_retriever": ".factory",
    "LocalBGEM3Embedder": ".lightrag_embedder",
    "LightRAGRetriever": ".lightrag_retriever",
    "HybridLightRAGRetriever": ".hybrid_lightrag",
    "KG_CONTEXT_CHUNK_ID": ".lightrag_retriever",
    "KG_CONTENT_TYPE": ".lightrag_retriever",
    "RetrievalConfig": ".schemas", "RetrievedChunk": ".schemas",
}
__all__ = list(_EXPORTS)


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(name)
    return getattr(import_module(_EXPORTS[name], __name__), name)
