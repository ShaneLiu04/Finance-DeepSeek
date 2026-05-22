"""
FastAPI 服务入口
支持 OpenAI-compatible Chat Completions、文档摄入与健康检查
"""

import os
import sys
import uuid
import asyncio
import logging
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Optional

import yaml
import torch
import psutil
from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.responses import StreamingResponse
from transformers import BitsAndBytesConfig

# 确保项目根目录在路径中
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from finance_deepseek.api.schemas import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    IngestRequest,
    HealthResponse,
)
from finance_deepseek.api.dependencies import app_state
from finance_deepseek.api.inference import InferenceEngine
from finance_deepseek.rag.indexer import FaissIndexer, build_from_corpus_dir
from finance_deepseek.reasoning.chain_parser import ChainParser

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def load_config() -> dict:
    config_path = PROJECT_ROOT / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


cfg = load_config()

# 并发控制 semaphore
MAX_CONCURRENCY = cfg["api"].get("max_concurrency", 1)
infer_semaphore = asyncio.Semaphore(MAX_CONCURRENCY)

# 基础敏感词过滤
SENSITIVE_WORDS = set(cfg["api"].get("sensitive_words", []))


def _check_sensitive_content(text: str) -> Optional[str]:
    """检查文本是否包含敏感词，返回匹配到的敏感词或 None"""
    if not SENSITIVE_WORDS:
        return None
    for word in SENSITIVE_WORDS:
        if word in text:
            return word
    return None


def get_quantization_config():
    if not torch.cuda.is_available():
        return None
    qcfg = cfg["quantization"]
    return BitsAndBytesConfig(
        load_in_4bit=qcfg["load_in_4bit"],
        bnb_4bit_quant_type=qcfg["bnb_4bit_quant_type"],
        bnb_4bit_use_double_quant=qcfg["bnb_4bit_use_double_quant"],
        bnb_4bit_compute_dtype=getattr(torch, qcfg["bnb_4bit_compute_dtype"]),
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    logger.info("Starting Finance-DeepSeek API server...")

    # 加载模型（优先使用本地路径）
    local_path = cfg["model"].get("local_path")
    base_model = local_path if local_path else cfg["model"]["base_model"]
    adapter_dir = Path(cfg["lora"]["adapter_output_dir"]) / "final_adapter"
    app_state.load_model(
        base_model=base_model,
        adapter_path=adapter_dir,
        quantization_config=get_quantization_config(),
    )

    # 加载检索器
    app_state.load_retriever()

    # 加载推理链解析器
    app_state.load_chain_parser(min_steps=cfg["reasoning"].get("min_reasoning_steps", 2))

    logger.info("All components loaded. Ready to serve.")
    yield
    logger.info("Shutting down...")
    app_state.unload()


app = FastAPI(
    title="Finance-DeepSeek API",
    description="基于 DeepSeek-R1-Distill-Qwen-1.5B 的金融推理增强问答系统",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/")
async def root():
    """欢迎页面"""
    return {
        "name": "Finance-DeepSeek API",
        "version": "1.0.0",
        "description": "基于 DeepSeek-R1-Distill-Qwen-1.5B 的金融领域推理增强问答系统",
        "endpoints": {
            "health": "GET /v1/health",
            "chat_completions": "POST /v1/chat/completions",
            "ingest": "POST /v1/ingest",
        },
        "modes": ["closed-book", "rag", "rag+reasoning"],
        "docs": "FastAPI 自动文档: /docs 或 /redoc",
    }


@app.get("/v1/health", response_model=HealthResponse)
async def health():
    """健康检查与状态监控"""
    gpu_used = 0.0
    gpu_total = 0.0
    if torch.cuda.is_available():
        gpu_used = torch.cuda.memory_allocated() / 1024 ** 3
        gpu_total = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3

    index_count = 0
    if app_state.retriever and app_state.retriever.indexer.index:
        index_count = app_state.retriever.indexer.index.ntotal

    return HealthResponse(
        status="ok",
        model_loaded=app_state.model is not None,
        adapter_loaded=app_state.adapter_path is not None,
        index_loaded=app_state.retriever is not None and app_state.retriever.indexer.index is not None,
        gpu_memory_used_gb=round(gpu_used, 2),
        gpu_memory_total_gb=round(gpu_total, 2),
        index_doc_count=index_count,
        version="1.0.0",
    )


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """
    OpenAI-compatible Chat Completions 端点
    支持 mode: closed-book | rag | rag+reasoning
    """
    if app_state.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # 输入安全过滤
    for msg in request.messages:
        hit = _check_sensitive_content(msg.content)
        if hit:
            raise HTTPException(status_code=400, detail=f"Input contains sensitive content: '{hit}'")

    async with infer_semaphore:
        engine = InferenceEngine(
            model=app_state.model,
            tokenizer=app_state.tokenizer,
            retriever=app_state.retriever,
            chain_parser=app_state.chain_parser,
        )

        if request.stream:
            prompt, sources = engine.build_prompt(request.messages, request.mode)

            async def event_generator():
                async for chunk in engine.stream_sse(
                    prompt=prompt,
                    mode=request.mode,
                    sources=sources,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    max_new_tokens=request.max_tokens,
                ):
                    yield chunk
                # 请求结束后尝试释放缓存
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            return StreamingResponse(
                event_generator(),
                media_type="text/event-stream",
            )

        # 非流式
        result = engine.run_inference(
            messages=request.messages,
            mode=request.mode,
            temperature=request.temperature,
            top_p=request.top_p,
            max_new_tokens=request.max_tokens,
        )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        content = result["raw"]
        if request.mode == "rag+reasoning" and result["parsed"]:
            # 返回结构化 JSON 字符串供前端解析
            parsed = result["parsed"]
            content = f"{parsed['raw_think']}\n\n{parsed['final_answer']}"

        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
            "object": "chat.completion",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }


@app.post("/v1/ingest")
async def ingest_documents(request: IngestRequest):
    """
    实时文档摄入：分块、embedding、增量更新 FAISS 索引
    """
    if app_state.retriever is None:
        raise HTTPException(status_code=503, detail="Retriever not initialized")

    # 输入安全过滤
    for doc in request.documents:
        hit = _check_sensitive_content(doc.get("text", ""))
        if hit:
            raise HTTPException(status_code=400, detail=f"Document contains sensitive content: '{hit}'")

    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=cfg["rag"]["chunk_size"],
        chunk_overlap=cfg["rag"]["chunk_overlap"],
    )

    all_chunks = []
    for doc in request.documents:
        texts = splitter.split_text(doc.get("text", ""))
        for t in texts:
            chunk = {
                "text": t,
                "source_url": doc.get("source_url", ""),
                "category": doc.get("category", ""),
                "title": doc.get("title", ""),
            }
            all_chunks.append(chunk)

    # 增量添加（生产级，无需全量重建）
    indexer = app_state.retriever.indexer
    added = indexer.add_chunks(all_chunks)
    indexer.save()

    return {"status": "success", "chunks_indexed": added, "total_indexed": indexer.index.ntotal if indexer.index else 0}


if __name__ == "__main__":
    import uvicorn

    host = cfg["api"]["host"]
    port = cfg["api"]["port"]
    uvicorn.run("finance_deepseek.api.main:app", host=host, port=port, reload=False)
