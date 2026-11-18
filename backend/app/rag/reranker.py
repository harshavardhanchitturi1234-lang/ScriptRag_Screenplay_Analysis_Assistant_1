"""
Reciprocal Rank Fusion (RRF) Reranker for Hybrid Screenwriting Retrieval.
Combines dense semantic ranking with sparse keyword ranking and exact-concept boosting.
"""

from typing import List, Dict, Any, Tuple


class HybridReranker:
    """
    Reranks candidate results using Reciprocal Rank Fusion (RRF) and metadata bonuses.
    Formula: RRF(d) = sum( weight_i / (k + rank_i(d)) )
    """

    def __init__(self, k_constant: int = 60, dense_weight: float = 0.55, sparse_weight: float = 0.45):
        self.k_constant = k_constant
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight

    def rerank(
        self,
        query: str,
        dense_results: List[Dict[str, Any]],
        sparse_results: List[Tuple[str, float, Dict[str, Any]]],
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Merges dense search results and sparse BM25 results into a unified, reranked list.
        """
        combined_scores: Dict[str, float] = {}
        candidate_data: Dict[str, Dict[str, Any]] = {}
        dense_scores_map: Dict[str, float] = {}
        sparse_scores_map: Dict[str, float] = {}

        # 1. Process dense rankings
        for rank, item in enumerate(dense_results, start=1):
            chunk_id = item["chunk_id"]
            dense_scores_map[chunk_id] = item.get("dense_score", 0.0)
            candidate_data[chunk_id] = {
                "chunk_id": chunk_id,
                "content": item.get("content", ""),
                "metadata": item.get("metadata", {})
            }
            rrf_score = self.dense_weight / (self.k_constant + rank)
            combined_scores[chunk_id] = combined_scores.get(chunk_id, 0.0) + rrf_score

        # 2. Process sparse BM25 rankings
        for rank, (chunk_id, bm25_score, meta) in enumerate(sparse_results, start=1):
            sparse_scores_map[chunk_id] = bm25_score
            if chunk_id not in candidate_data:
                candidate_data[chunk_id] = {
                    "chunk_id": chunk_id,
                    "content": meta.get("content", ""),
                    "metadata": meta
                }
            rrf_score = self.sparse_weight / (self.k_constant + rank)
            combined_scores[chunk_id] = combined_scores.get(chunk_id, 0.0) + rrf_score

        # 3. Apply exact technical term match bonus
        query_lower = query.lower()
        for chunk_id, data in candidate_data.items():
            meta = data.get("metadata", {})
            concept = meta.get("concept_name", "").lower()
            section = meta.get("chapter_or_section", "").lower()
            
            # Boost if query mentions the exact concept or section
            if concept and concept in query_lower:
                combined_scores[chunk_id] = combined_scores.get(chunk_id, 0.0) + 0.02
            if section and section in query_lower:
                combined_scores[chunk_id] = combined_scores.get(chunk_id, 0.0) + 0.01

        # 4. Normalize combined scores to [0.0, 1.0]
        max_rrf = max(combined_scores.values()) if combined_scores else 1.0
        final_list = []

        for chunk_id, rrf_val in combined_scores.items():
            norm_score = round(min(1.0, rrf_val / max_rrf), 4)
            data = candidate_data[chunk_id]
            final_list.append({
                "chunk_id": chunk_id,
                "content": data["content"],
                "metadata": data["metadata"],
                "dense_score": dense_scores_map.get(chunk_id, 0.0),
                "sparse_score": sparse_scores_map.get(chunk_id, 0.0),
                "combined_score": norm_score
            })

        # Sort descending by combined score
        final_list.sort(key=lambda x: x["combined_score"], reverse=True)
        return final_list[:top_k]

