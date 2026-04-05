"""
rag/vector_store.py
- Qdrant 로컬 DB 연결 및 CRUD 작업 모듈
- 로컬 파일 경로 방식은 프로세스당 1개 인스턴스만 허용
- @st.cache_resource 로 Streamlit 전체 세션에서 단일 클라이언트 공유
"""

import uuid
import config
import streamlit as st
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


@st.cache_resource
def get_client() -> QdrantClient:
    """
    Qdrant 로컬 클라이언트를 반환.
    @st.cache_resource 로 앱 전체에서 단 1개 인스턴스만 생성·공유한다.
    (로컬 파일 경로는 동시 접근 불가 → 싱글톤 필수)
    """
    return QdrantClient(path=config.QDRANT_PATH)


def warmup():
    """앱 시작 시 클라이언트 + 기본 컬렉션을 미리 초기화"""
    try:
        ensure_collection(config.COLLECTION_NAME)
        ensure_collection(config.SCHEMA_COLLECTION)
    except Exception:
        pass


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

    # 저장 시각 (한 번만 생성해서 모든 청크에 동일하게 기록)
    from datetime import datetime
    upload_date = datetime.now().strftime("%Y-%m-%d %H:%M")

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
                    "upload_date": upload_date,   # 등록일
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
    # qdrant-client >= 1.7.0 : search() 폐기 → query_points() 사용
    response = client.query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=top_k,
        with_payload=True,
    )
    return [r.payload.get("page_content", "") for r in response.points]


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
    저장된 파일 목록을 반환.
    각 파일마다 {filename, chunks, file_size, upload_date} 포함.
    오류 발생 시 빈 리스트를 반환하지 않고 예외를 그대로 올린다
    (호출부 UI에서 st.error로 표시).
    """
    ensure_collection(collection_name)
    client = get_client()

    points, _ = client.scroll(
        collection_name=collection_name,
        with_payload=True,
        limit=50000,
    )

    file_map: dict = {}
    for point in points:
        meta = point.payload.get("metadata", {})
        fname = meta.get("filename", "unknown")
        if fname not in file_map:
            file_map[fname] = {
                "filename": fname,
                "chunks": 0,
                "file_size": meta.get("file_size", 0),
                "upload_date": meta.get("upload_date", "-"),
            }
        file_map[fname]["chunks"] += 1

    return sorted(file_map.values(), key=lambda x: x["filename"])


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
