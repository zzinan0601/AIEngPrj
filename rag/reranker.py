"""
rag/reranker.py
- bge-reranker-v2-m3 모델로 검색 결과를 재정렬(Reranking)
- CrossEncoder: (query, document) 쌍의 관련도 점수를 직접 계산
- 초기 벡터 검색 결과 TOP_K개 → 재정렬 후 RERANK_TOP_K개 반환
"""

import config
from sentence_transformers import CrossEncoder

_reranker = None  # 싱글톤 인스턴스


def get_reranker() -> CrossEncoder:
    """
    bge-reranker CrossEncoder 모델 반환 (최초 1회 로드).
    모델 경로: config.RERANKER_MODEL_PATH (로컬 디렉토리)
    """
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder(
            config.RERANKER_MODEL_PATH,
            max_length=512,     # 입력 토큰 최대 길이
            device="cpu",       # 폐쇄망 CPU 환경
        )
    return _reranker


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

    # (query, document) 쌍 생성
    pairs = [(query, doc) for doc in documents]

    # 관련도 점수 계산
    scores = reranker.predict(pairs)

    # 점수 기준 내림차순 정렬 후 상위 k개 반환
    ranked = sorted(
        zip(scores.tolist(), documents),
        key=lambda x: x[0],
        reverse=True,
    )
    return [doc for _, doc in ranked[:k]]
