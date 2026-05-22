"""
推理引擎
支持 Closed-Book / RAG / RAG+Reasoning 三种模式，SSE 流式输出
"""

import os
import re
import uuid
import logging
from typing import AsyncGenerator, List, Dict, Optional

import torch
from transformers import TextIteratorStreamer
from threading import Thread

from finance_deepseek.api.schemas import ChatMessage, ParsedOutput, ChunkSource
from finance_deepseek.reasoning.chain_parser import ChainParser
from finance_deepseek.reasoning.validator import ChainValidator

logger = logging.getLogger(__name__)


SYSTEM_PROMPT_TEMPLATE = """你是一位资深金融分析师。请仅根据以下参考材料回答问题。
如果材料不足以回答，请明确说明"根据现有资料无法确定"。
思考过程请放在 <think> 标签内，最终答案放在 <answer> 标签内。

[参考材料]
{context}

[用户问题]
{query}

[回答]
"""

CLOSED_BOOK_SYSTEM_PROMPT = """你是一位资深金融分析师。思考过程请放在 <think> 标签内，最终答案放在 <answer> 标签内。

[用户问题]
{query}

[回答]
"""


class InferenceEngine:
    """
    推理引擎：负责模式路由、Prompt 拼接、生成与解析
    """

    def __init__(self, model, tokenizer, retriever=None, chain_parser=None, device="cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.retriever = retriever
        self.chain_parser = chain_parser or ChainParser()
        self.validator = ChainValidator()
        self.device = device

    def build_prompt(
        self,
        messages: List[ChatMessage],
        mode: str,
        max_input_length: int = 1024,
    ) -> str:
        """
        根据模式构建最终输入文本
        Token 级精确截断，确保不超过模型上下文限制
        """
        # 提取用户最新 query
        user_messages = [m for m in messages if m.role == "user"]
        query = user_messages[-1].content if user_messages else ""

        if mode in ("rag", "rag+reasoning") and self.retriever is not None:
            try:
                chunks = self.retriever.retrieve(query)
                context = self.retriever.format_context(chunks)
                sources = [c.get("source_url", "") for c in chunks]
            except Exception as e:
                logger.warning(f"Retrieval failed: {e}")
                context = "（检索服务暂不可用）"
                sources = []
            prompt_text = SYSTEM_PROMPT_TEMPLATE.format(context=context, query=query)
        else:
            prompt_text = CLOSED_BOOK_SYSTEM_PROMPT.format(query=query)
            sources = []

        # Token 级精确截断
        prompt_text = self._truncate_by_tokens(prompt_text, max_input_length)
        return prompt_text, sources

    def _truncate_by_tokens(self, text: str, max_tokens: int) -> str:
        """使用 tokenizer 进行精确的 token 级截断"""
        try:
            encoded = self.tokenizer.encode(text, add_special_tokens=False)
            if len(encoded) > max_tokens:
                truncated = encoded[:max_tokens]
                text = self.tokenizer.decode(truncated, skip_special_tokens=True)
                logger.warning(f"Prompt truncated from {len(encoded)} to {max_tokens} tokens")
        except Exception as e:
            logger.warning(f"Token truncation failed: {e}, falling back to char truncation")
            if len(text) > max_tokens * 4:
                text = text[: max_tokens * 4]
        return text

    def generate_sync(
        self,
        prompt: str,
        temperature: float = 0.6,
        top_p: float = 0.9,
        max_new_tokens: int = 1024,
    ) -> str:
        """同步非流式生成"""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        input_len = inputs["input_ids"].shape[1]

        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        generated = outputs[0][input_len:]
        text = self.tokenizer.decode(generated, skip_special_tokens=True)
        return text

    def generate_stream(
        self,
        prompt: str,
        temperature: float = 0.6,
        top_p: float = 0.9,
        max_new_tokens: int = 1024,
    ) -> TextIteratorStreamer:
        """
        启动流式生成，返回 streamer 对象
        注意：需要在单独线程中调用 model.generate()
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        input_len = inputs["input_ids"].shape[1]

        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )

        generation_kwargs = dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()
        return streamer, input_len

    async def stream_sse(
        self,
        prompt: str,
        mode: str,
        sources: List[str],
        temperature: float = 0.6,
        top_p: float = 0.9,
        max_new_tokens: int = 1024,
    ) -> AsyncGenerator[str, None]:
        """
        SSE 流式输出生成器
        对于 rag+reasoning 模式，先传推理链、再传答案
        """
        streamer, input_len = self.generate_stream(
            prompt, temperature=temperature, top_p=top_p, max_new_tokens=max_new_tokens
        )

        buffer = ""
        think_started = False
        think_ended = False
        answer_started = False

        for token in streamer:
            buffer += token
            yield f"data: {token}\n\n"

        # 流结束后，进行结构化解析（仅 rag+reasoning）
        if mode == "rag+reasoning":
            parsed = self.chain_parser.parse(buffer, sources=sources)
            # 发送结构化结束标记
            yield f"event: parsed\ndata: {parsed.to_json(ensure_ascii=False)}\n\n"

        yield "data: [DONE]\n\n"

    def run_inference(
        self,
        messages: List[ChatMessage],
        mode: str,
        temperature: float = 0.6,
        top_p: float = 0.9,
        max_new_tokens: int = 1024,
    ) -> Dict:
        """
        非流式推理入口，返回完整响应字典
        """
        prompt, sources = self.build_prompt(messages, mode)
        raw_output = self.generate_sync(
            prompt,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
        )

        if mode == "rag+reasoning":
            parsed = self.chain_parser.parse(raw_output, sources=sources)
            is_valid, val_reason = self.validator.validate(
                parsed.reasoning_steps, parsed.final_answer
            )
            return {
                "raw": raw_output,
                "parsed": parsed.to_dict(),
                "validation": {"consistent": is_valid, "reason": val_reason},
            }

        return {
            "raw": raw_output,
            "parsed": None,
            "validation": None,
        }
