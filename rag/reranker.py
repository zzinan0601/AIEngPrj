"""
rag/reranker.py
- bge-reranker-v2-m3 로드 방식: transformers 직접 사용
  (CrossEncoder/FlagEmbedding 모두 xlm-roberta 연결 문제로 실패하는 경우 대응)
- AutoTokenizer + XLMRobertaForSequenceClassification 직접 로드
- 모델 없거나 로드 실패 시 graceful fallback (원본 순서 반환)
"""

import torch
import config

_tokenizer  = None
_model      = None
_reranker_available = None  # None=미시도, True=성공, False=실패


def get_reranker():
    """
    tokenizer + model 을 직접 로드해 반환.
    실패 시 None 반환 → rerank() 에서 원본 순서 폴백.
    """
    global _tokenizer, _model, _reranker_available

    if _reranker_available is None:
        try:
            from transformers import AutoTokenizer, XLMRobertaForSequenceClassification

            path = config.RERANKER_MODEL_PATH
            _tokenizer = AutoTokenizer.from_pretrained(path)
            _model = XLMRobertaForSequenceClassification.from_pretrained(path)
            _model.eval()  # 추론 모드
            _reranker_available = True
            print("[reranker] XLMRobertaForSequenceClassification 로드 성공")

        except Exception as e:
            print(f"[reranker] 모델 로드 실패 (폴백 모드): {e}")
            _reranker_available = False

    return (_tokenizer, _model) if _reranker_available else (None, None)


def _compute_scores(query: str, documents: list) -> list:
    """tokenizer + model 로 (query, doc) 쌍의 관련도 점수 계산"""
    tokenizer, model = get_reranker()
    pairs = [[query, doc] for doc in documents]

    inputs = tokenizer(
        pairs,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )

    with torch.no_grad():
        outputs = model(**inputs)
        # logits shape: (N, 1) 또는 (N, 2)
        logits = outputs.logits
        if logits.shape[-1] == 1:
            scores = logits.squeeze(-1).tolist()
        else:
            # 2-class: yes/no → yes(index 1) 점수 사용
            scores = logits[:, 1].tolist()

    return scores if isinstance(scores, list) else [scores]


def rerank(query: str, documents: list, top_k: int = None) -> list:
    """
    문서 목록을 query 관련도 순으로 재정렬.

    Args:
        query    : 검색 질문
        documents: 텍스트 문자열 목록
        top_k    : 반환할 상위 문서 수 (기본: config.RERANK_TOP_K)

    Returns:
        재정렬된 상위 문서 텍스트 목록
    """
    if not documents:
        return []

    k = top_k or config.RERANK_TOP_K
    tokenizer, model = get_reranker()

    # 모델 없으면 원본 순서 상위 k개 반환
    if tokenizer is None or model is None:
        return documents[:k]

    scores = _compute_scores(query, documents)
    ranked = sorted(zip(scores, documents), key=lambda x: x[0], reverse=True)
    return [doc for _, doc in ranked[:k]]
