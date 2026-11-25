"""
Reciprocal Rank Fusion (RRF) Algorithm
======================================
Combines multiple ranked candidate lists into a single consolidated ranking.
Based on Cormack, Clarke, and Büttcher (SIGIR 2009).
"""

from typing import Any, Dict, List, Optional, Tuple


def compute_rrf_score(rank: int, weight: float = 1.0, k: int = 60) -> float:
    """
    Computes reciprocal rank score for an item at 1-based rank position.
    Formula: weight / (k + rank)
    """
    if rank < 1:
        return 0.0
    return weight / (k + rank)


def fuse_rankings(
    vector_results: List[Any],
    keyword_results: List[Any],
    id_getter: Any = lambda x: getattr(x, "id", None) or x.get("id"),
    alpha: float = 0.5,
    rrf_k: int = 60,
) -> List[Dict[str, Any]]:
    """
    Fuses a vector candidate list and keyword candidate list using Reciprocal Rank Fusion.
    
    Args:
        vector_results: Ranked list from dense vector search.
        keyword_results: Ranked list from sparse keyword search.
        id_getter: Callable extracting unique ID from candidate items.
        alpha: Weight assigned to vector search; (1 - alpha) is given to keyword search.
        rrf_k: Smoothing constant (standard default: 60).
        
    Returns:
        List of fused candidate dicts sorted descending by combined RRF score.
    """
    fused_scores: Dict[str, float] = {}
    items_by_id: Dict[str, Any] = {}
    v_ranks: Dict[str, int] = {}
    k_ranks: Dict[str, int] = {}

    w_vector = alpha
    w_keyword = 1.0 - alpha

    # 1. Score Vector candidates
    for rank, item in enumerate(vector_results, start=1):
        item_id = str(id_getter(item))
        items_by_id[item_id] = item
        v_ranks[item_id] = rank
        fused_scores[item_id] = compute_rrf_score(rank=rank, weight=w_vector, k=rrf_k)

    # 2. Score Keyword candidates
    for rank, item in enumerate(keyword_results, start=1):
        item_id = str(id_getter(item))
        if item_id not in items_by_id:
            items_by_id[item_id] = item
        k_ranks[item_id] = rank
        kw_score = compute_rrf_score(rank=rank, weight=w_keyword, k=rrf_k)
        fused_scores[item_id] = fused_scores.get(item_id, 0.0) + kw_score

    # 3. Sort descending by fused score
    sorted_candidates = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)

    results = []
    for item_id, combined_score in sorted_candidates:
        item = items_by_id[item_id]
        v_r = v_ranks.get(item_id)
        k_r = k_ranks.get(item_id)

        if v_r is not None and k_r is not None:
            matched_by = "both"
        elif v_r is not None:
            matched_by = "vector_only"
        else:
            matched_by = "keyword_only"

        results.append({
            "id": item_id,
            "item": item,
            "combined_score": round(combined_score, 6),
            "vector_rank": v_r,
            "keyword_rank": k_r,
            "matched_by": matched_by,
        })

    return results


def fuse_weighted_scores(
    vector_results: List[Any],
    keyword_results: List[Any],
    id_getter: Any = lambda x: getattr(x, "id", None) or x.get("id"),
    alpha: float = 0.5,
) -> List[Dict[str, Any]]:
    """
    Fuses a vector candidate list and keyword candidate list using normalized convex combination:
    combined_score = alpha * vector_similarity + (1 - alpha) * (keyword_rank / max_keyword_rank)
    """
    items_by_id: Dict[str, Any] = {}
    v_sims: Dict[str, float] = {}
    v_ranks: Dict[str, int] = {}
    k_scores: Dict[str, float] = {}
    k_ranks: Dict[str, int] = {}

    for rank, item in enumerate(vector_results, start=1):
        item_id = str(id_getter(item))
        items_by_id[item_id] = item
        v_ranks[item_id] = rank
        sim_val = getattr(item, "similarity", None)
        v_sims[item_id] = float(sim_val) if sim_val is not None else 0.0

    max_kw = 1.0
    for rank, item in enumerate(keyword_results, start=1):
        item_id = str(id_getter(item))
        if item_id not in items_by_id:
            items_by_id[item_id] = item
        k_ranks[item_id] = rank
        kw_val = float(getattr(item, "rank", 0.0) or 0.0)
        k_scores[item_id] = kw_val
        if kw_val > max_kw:
            max_kw = kw_val

    results = []
    for item_id, item in items_by_id.items():
        v_sim = v_sims.get(item_id, 0.0)
        k_sc = k_scores.get(item_id, 0.0)
        norm_k = k_sc / max_kw if max_kw > 0.0 else 0.0
        combined = (alpha * v_sim) + ((1.0 - alpha) * norm_k)

        v_r = v_ranks.get(item_id)
        k_r = k_ranks.get(item_id)
        if v_r is not None and k_r is not None:
            matched_by = "both"
        elif v_r is not None:
            matched_by = "vector_only"
        else:
            matched_by = "keyword_only"

        results.append({
            "id": item_id,
            "item": item,
            "combined_score": round(combined, 6),
            "vector_rank": v_r,
            "keyword_rank": k_r,
            "vector_similarity": v_sim if v_r is not None else None,
            "keyword_score": k_sc if k_r is not None else None,
            "matched_by": matched_by,
        })

    results.sort(key=lambda x: x["combined_score"], reverse=True)
    return results
