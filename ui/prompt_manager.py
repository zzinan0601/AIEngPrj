"""
ui/prompt_manager.py
- 시스템 프롬프트와 퓨샷(Few-Shot) 예제를 JSON 파일로 관리
- 저장 위치: config.PROMPTS_PATH/prompts.json
- UI(modals.py)에서 호출해 CRUD 작업 수행
"""

import json
import os
import config

# 프롬프트 저장 파일 경로
PROMPTS_FILE = os.path.join(config.PROMPTS_PATH, "prompts.json")

# 기본 설정값
DEFAULT_CONFIG = {
    "system_prompt": "당신은 친절하고 정확한 AI 어시스턴트입니다. 한국어로 답변하세요.",
    "fewshots": [],
}


def load_prompt_config() -> dict:
    """
    저장된 프롬프트 설정을 반환.
    파일이 없으면 기본값 반환.
    """
    os.makedirs(config.PROMPTS_PATH, exist_ok=True)
    if not os.path.exists(PROMPTS_FILE):
        return DEFAULT_CONFIG.copy()

    try:
        with open(PROMPTS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        # 필수 키가 없으면 기본값으로 보완
        data.setdefault("system_prompt", DEFAULT_CONFIG["system_prompt"])
        data.setdefault("fewshots", [])
        return data
    except (json.JSONDecodeError, IOError):
        return DEFAULT_CONFIG.copy()


def save_prompt_config(data: dict):
    """프롬프트 설정을 JSON 파일에 저장"""
    os.makedirs(config.PROMPTS_PATH, exist_ok=True)
    with open(PROMPTS_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def update_system_prompt(prompt: str):
    """시스템 프롬프트만 수정"""
    data = load_prompt_config()
    data["system_prompt"] = prompt
    save_prompt_config(data)


def add_fewshot(question: str, answer: str):
    """퓨샷 예제 추가"""
    data = load_prompt_config()
    data["fewshots"].append({"question": question, "answer": answer})
    save_prompt_config(data)


def update_fewshot(index: int, question: str, answer: str):
    """특정 인덱스의 퓨샷 예제 수정"""
    data = load_prompt_config()
    if 0 <= index < len(data["fewshots"]):
        data["fewshots"][index] = {"question": question, "answer": answer}
        save_prompt_config(data)


def delete_fewshot(index: int):
    """특정 인덱스의 퓨샷 예제 삭제"""
    data = load_prompt_config()
    if 0 <= index < len(data["fewshots"]):
        data["fewshots"].pop(index)
        save_prompt_config(data)


def reset_to_default():
    """모든 프롬프트 설정을 기본값으로 초기화"""
    save_prompt_config(DEFAULT_CONFIG.copy())
