"""
rag/embeddings.py
- bge-m3 모델을 로컬 경로에서 로드하는 임베딩 싱글톤
- 최초 1회만 모델을 로드하고 이후 재사용 (메모리 절약)
- encode_kwargs normalize=True: 코사인 유사도 계산에 최적화
"""

import config
from langchain_huggingface import HuggingFaceEmbeddings

_embeddings = None  # 싱글톤 인스턴스


def get_embeddings() -> HuggingFaceEmbeddings:
    """
    bge-m3 임베딩 모델을 반환 (최초 1회 로드).
    모델 경로: config.EMBEDDING_MODEL_PATH (로컬 디렉토리)
    """
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(
            model_name=config.EMBEDDING_MODEL_PATH,  # 로컬 경로
            model_kwargs={
                "device": "cpu",           # 폐쇄망 CPU 환경
                "trust_remote_code": True, # bge-m3 필수 옵션
            },
            encode_kwargs={
                "normalize_embeddings": True,  # 코사인 유사도용 정규화
                "batch_size": 16,              # 배치 처리로 속도 개선
            },
        )
    return _embeddings


def embed_texts(texts: list) -> list:
    """텍스트 목록을 벡터로 변환"""
    emb = get_embeddings()
    return emb.embed_documents(texts)


def embed_query(query: str) -> list:
    """단일 쿼리 텍스트를 벡터로 변환"""
    emb = get_embeddings()
    return emb.embed_query(query)
