"""
rag/vector_store.py
- Qdrant 로컬 DB 연결 및 CRUD 작업 모듈
- qdrant-client를 직접 사용해 payload 구조를 완전 제어
- 문서 컬렉션과 스키마 컬렉션을 분리 관리
"""

import uuid
import config
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    FilterSelector,
)
from rag.embeddings import embed_texts, embed_query

_client: QdrantClient = None  # 싱글톤


def get_client() -> QdrantClient:
    """Qdrant 로컬 클라이언트 싱글톤 반환"""
    global _client
    if _client is None:
        _client = QdrantClient(path=config.QDRANT_PATH)
    return _client


def ensure_collection(collection_name: str = config.COLLECTION_NAME):
    """컬렉션이 없으면 자동 생성"""
    client = get_client()
    existing = {c.name for c in client.get_collections().collections}
    if collection_name not in existing:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=config.VECTOR_SIZE,
                distance=Distance.COSINE,
            ),
        )


def save_chunks(
    chunks: list,
    metadata: dict,
    collection_name: str = config.COLLECTION_NAME,
) -> int:
    """
    텍스트 청크를 임베딩하여 Qdrant에 저장.

    Args:
        chunks: 텍스트 청크 목록
        metadata: 각 포인트에 저장할 메타데이터 (filename 등)
        collection_name: 저장할 컬렉션 이름

    Returns:
        저장된 포인트(청크) 수
    """
    if not chunks:
        return 0

    ensure_collection(collection_name)
    client = get_client()

    # 텍스트 → 벡터 변환
    vectors = embed_texts(chunks)

    # 포인트 리스트 생성
    points = [
        PointStruct(
            id=str(uuid.uuid4()),  # UUID 형식
            vector=vector,
            payload={
                "page_content": chunk,       # 원문 텍스트
                "metadata": {
                    **metadata,
                    "chunk_index": i,
                },
            },
        )
        for i, (chunk, vector) in enumerate(zip(chunks, vectors))
    ]

    # 배치 업서트 (upsert = insert or update)
    client.upsert(collection_name=collection_name, points=points)
    return len(points)


def search_documents(
    query: str,
    top_k: int = config.TOP_K,
    collection_name: str = config.COLLECTION_NAME,
) -> list:
    """
    쿼리 벡터로 유사 문서 검색.

    Returns:
        텍스트 청크 문자열 목록
    """
    ensure_collection(collection_name)
    client = get_client()

    query_vector = embed_query(query)
    results = client.search(
        collection_name=collection_name,
        query_vector=query_vector,
        limit=top_k,
        with_payload=True,
    )
    return [r.payload.get("page_content", "") for r in results]


def delete_by_filename(
    filename: str,
    collection_name: str = config.COLLECTION_NAME,
):
    """파일명으로 해당 파일의 모든 청크를 삭제"""
    client = get_client()
    client.delete(
        collection_name=collection_name,
        points_selector=FilterSelector(
            filter=Filter(
                must=[
                    FieldCondition(
                        key="metadata.filename",
                        match=MatchValue(value=filename),
                    )
                ]
            )
        ),
    )


def get_file_list(collection_name: str = config.COLLECTION_NAME) -> list:
    """
    저장된 파일 목록을 [{filename, chunks}] 형태로 반환.
    파일명 기준으로 포인트 수를 집계.
    """
    try:
        ensure_collection(collection_name)
        client = get_client()

        # 전체 포인트 스크롤 (limit은 충분히 크게)
        points, _ = client.scroll(
            collection_name=collection_name,
            with_payload=True,
            limit=50000,
        )

        # 파일명별 청크 수 집계
        file_map: dict = {}
        for point in points:
            fname = point.payload.get("metadata", {}).get("filename", "unknown")
            file_map[fname] = file_map.get(fname, 0) + 1

        return [{"filename": k, "chunks": v} for k, v in sorted(file_map.items())]
    except Exception:
        return []


def get_file_chunks(
    filename: str,
    collection_name: str = config.COLLECTION_NAME,
) -> list:
    """특정 파일의 모든 청크 내용을 반환 (chunk_index 순 정렬)"""
    try:
        client = get_client()
        points, _ = client.scroll(
            collection_name=collection_name,
            scroll_filter=Filter(
                must=[
                    FieldCondition(
                        key="metadata.filename",
                        match=MatchValue(value=filename),
                    )
                ]
            ),
            with_payload=True,
            limit=5000,
        )
        # chunk_index 기준 정렬
        sorted_points = sorted(
            points,
            key=lambda p: p.payload.get("metadata", {}).get("chunk_index", 0),
        )
        return [p.payload.get("page_content", "") for p in sorted_points]
    except Exception:
        return []
