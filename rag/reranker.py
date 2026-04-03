"""
rag/reranker.py
- bge-reranker-v2-m3 모델로 검색 결과를 재정렬(Reranking)
- CrossEncoder: (query, document) 쌍의 관련도 점수를 직접 계산
- 초기 벡터 검색 결과 TOP_K개 → 재정렬 후 RERANK_TOP_K개 반환
"""

import config
from sentence_transformers import CrossEncoder

_reranker = None        # 싱글톤 인스턴스
_reranker_available = None  # 로드 가능 여부 캐시


def get_reranker():
    """
    bge-reranker CrossEncoder 모델 반환 (최초 1회 로드).
    모델 파일이 없으면 None 반환 → rerank()에서 원본 순서로 폴백.
    """
    global _reranker, _reranker_available
    if _reranker_available is None:   # 아직 시도 안 함
        try:
            _reranker = CrossEncoder(
                config.RERANKER_MODEL_PATH,
                max_length=512,
                device="cpu",
            )
            _reranker_available = True
        except Exception as e:
            print(f"[reranker] 모델 로드 실패 (폴백 모드): {e}")
            _reranker_available = False
    return _reranker if _reranker_available else None


def rerank(query: str, documents: list, top_k: int = None) -> list:
    """
    문서 목록을 query 관련도 순으로 재정렬한다.

    Args:
        query: 검색 질문
        documents: 초기 검색 결과 텍스트 목록
        top_k: 반환할 상위 문서 수 (기본: config.RERANK_TOP_K)

    Returns:
        재정렬된 상위 문서 텍스트 목록
    """
    if not documents:
        return []

    k = top_k or config.RERANK_TOP_K
    reranker = get_reranker()

    # 모델 없으면 원본 순서에서 상위 k개만 반환 (폴백)
    if reranker is None:
        return documents[:k]

    pairs = [(query, doc) for doc in documents]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(scores.tolist(), documents), key=lambda x: x[0], reverse=True)
    return [doc for _, doc in ranked[:k]]
