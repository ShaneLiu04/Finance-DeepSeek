"""
Pydantic 数据模型定义
兼容 OpenAI Chat Completions API 风格
"""

from typing import List, Optional, Literal
from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"] = "user"
    content: str


class ChatCompletionRequest(BaseModel):
    messages: List[ChatMessage]
    mode: Literal["closed-book", "rag", "rag+reasoning"] = "rag+reasoning"
    stream: bool = False
    temperature: float = Field(default=0.6, ge=0.0, le=0.7)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    max_tokens: int = Field(default=1024, ge=1, le=2048)


class ChunkSource(BaseModel):
    text: str = ""
    source_url: str = ""
    category: str = ""
    title: str = ""
    score: float = 0.0


class ParsedOutput(BaseModel):
    reasoning_steps: List[str] = []
    final_answer: str = ""
    confidence: float = 0.95
    sources: List[str] = []
    fallback_triggered: bool = False
    fallback_reason: str = ""


class ChatCompletionResponse(BaseModel):
    id: str = "finance-deepseek-001"
    object: str = "chat.completion"
    choices: List[dict]
    usage: dict = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


class IngestRequest(BaseModel):
    documents: List[dict]  # 每个元素含 text, source_url, category, title


class HealthResponse(BaseModel):
    status: str = "ok"
    model_loaded: bool = False
    adapter_loaded: bool = False
    index_loaded: bool = False
    gpu_memory_used_gb: float = 0.0
    gpu_memory_total_gb: float = 0.0
    index_doc_count: int = 0
    version: str = "1.0.0"
