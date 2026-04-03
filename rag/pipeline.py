"""
rag/pipeline.py
- 문서 업로드 처리 전체 파이프라인
  1. 파일 로드 (PDF / TXT / DOCX)
  2. 텍스트 청킹 (RecursiveCharacterTextSplitter)
  3. 임베딩 후 Qdrant 저장
- 검색 파이프라인
  1. 벡터 유사도 검색 (TOP_K)
  2. bge-reranker 재정렬 (RERANK_TOP_K)
"""

import os
import config
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from rag.vector_store import save_chunks, search_documents
from rag.reranker import rerank

# 지원하는 파일 확장자 → 로더 매핑
LOADER_MAP = {
    ".pdf": PyPDFLoader,
    ".txt": TextLoader,
    ".docx": Docx2txtLoader,
}


def load_document(file_path: str) -> list:
    """
    파일 경로로 문서를 로드하고 langchain Document 리스트 반환.
    지원 형식: .pdf, .txt, .docx
    """
    ext = os.path.splitext(file_path)[1].lower()
    loader_cls = LOADER_MAP.get(ext)
    if not loader_cls:
        raise ValueError(f"지원하지 않는 파일 형식: {ext}  (지원: pdf, txt, docx)")

    # TextLoader는 encoding 지정 필요
    if ext == ".txt":
        loader = loader_cls(file_path, encoding="utf-8")
    else:
        loader = loader_cls(file_path)

    return loader.load()


def split_documents(docs: list) -> list:
    """
    문서를 청크로 분할.
    chunk_size / chunk_overlap 은 .env 설정값 사용.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", "!", "?", " ", ""],
    )
    return splitter.split_documents(docs)


def process_and_store(file_path: str, filename: str) -> int:
    """
    문서 로드 → 청킹 → 임베딩 → Qdrant 저장 전체 파이프라인.

    Args:
        file_path: 저장된 임시 파일 경로
        filename: 원본 파일명 (메타데이터에 기록)

    Returns:
        저장된 청크 수
    """
    # 1. 문서 로드
    docs = load_document(file_path)

    # 2. 청킹
    chunks = split_documents(docs)
    chunk_texts = [c.page_content for c in chunks]

    # 3. 임베딩 & 저장
    metadata = {"filename": filename, "source": file_path}
    count = save_chunks(chunk_texts, metadata)
    return count


def search_and_rerank(query: str) -> list:
    """
    벡터 검색 → reranker 재정렬 파이프라인.

    Args:
        query: 검색 질문

    Returns:
        재정렬된 상위 문서 텍스트 목록
    """
    # 1. 벡터 유사도 검색 (TOP_K개)
    initial_results = search_documents(query, top_k=config.TOP_K)

    if not initial_results:
        return []

    # 2. bge-reranker로 재정렬 (RERANK_TOP_K개)
    return rerank(query, initial_results)
