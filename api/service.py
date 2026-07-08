"""
Business logic service layer for the Finance-DeepSeek FastAPI application.
Handles model loading, RAG retrieval, generation, parsing, and validation.
"""

import os
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


def load_config() -> dict:
    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class FinanceDeepSeekService:
    """
    Singleton-style service encapsulating all business logic:
      - LLM loading & generation
      - RAG retrieval (DenseRetriever)
      - Chain parsing & validation
    """

    _instance: Optional["FinanceDeepSeekService"] = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, config: Optional[dict] = None):
        if self._initialized:
            return

        self.cfg = config or load_config()
        self._model: Optional[AutoModelForCausalLM] = None
        self._tokenizer: Optional[AutoTokenizer] = None
        self._retriever: Optional[Any] = None
        self._parser: Optional[Any] = None
        self._validator: Optional[Any] = None
        self._initialized = True

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def model_loaded(self) -> bool:
        return self._model is not None and self._tokenizer is not None

    @property
    def index_loaded(self) -> bool:
        return self._retriever is not None and self._retriever.indexer.index is not None

    # ------------------------------------------------------------------
    # Lazy initializers
    # ------------------------------------------------------------------

    def _load_model(self) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """Lazy-load the causal LM and tokenizer."""
        if self._model is not None and self._tokenizer is not None:
            return self._model, self._tokenizer

        mcfg = self.cfg["model"]
        model_path = mcfg.get("local_path") or mcfg["base_model"]

        logger.info(f"Loading model from {model_path} ...")

        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            use_fast=True,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id

        load_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": getattr(torch, mcfg.get("torch_dtype", "bfloat16")),
        }
        if torch.cuda.is_available():
            load_kwargs["device_map"] = "auto"
        else:
            logger.warning("CUDA not available; loading to CPU (slow)")

        model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
        model.eval()

        self._model = model
        self._tokenizer = tokenizer
        logger.info("Model loaded successfully.")
        return model, tokenizer

    def _load_retriever(self):
        """Lazy-load the DenseRetriever."""
        if self._retriever is not None:
            return self._retriever

        from rag import DenseRetriever

        self._retriever = DenseRetriever()
        logger.info("DenseRetriever initialized.")
        return self._retriever

    def _load_parser(self):
        """Lazy-load the ChainParser."""
        if self._parser is not None:
            return self._parser

        from reasoning import ChainParser

        self._parser = ChainParser(min_steps=self.cfg.get("reasoning", {}).get("min_steps", 2))
        return self._parser

    def _load_validator(self):
        """Lazy-load the ChainValidator."""
        if self._validator is not None:
            return self._validator

        from reasoning import ChainValidator

        tol = self.cfg.get("reasoning", {}).get("numeric_tolerance", 0.01)
        self._validator = ChainValidator(numeric_tolerance=tol)
        return self._validator

    # ------------------------------------------------------------------
    # RAG Retrieval
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        rerank_top_k: Optional[int] = None,
    ) -> Tuple[List[Dict[str, Any]], str]:
        """
        Retrieve relevant chunks and return them plus formatted context.

        Returns:
            (chunks, context_string)
        """
        retriever = self._load_retriever()
        chunks = retriever.retrieve(query, top_k=top_k, rerank_top_k=rerank_top_k)
        context = retriever.format_context(chunks)
        return chunks, context

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate(
        self,
        query: str,
        context: Optional[str] = None,
        temperature: float = 0.6,
        max_new_tokens: int = 1024,
        top_p: float = 0.9,
    ) -> str:
        """
        Generate a reasoning chain + answer for the given query.

        Args:
            query: User question.
            context: Optional context string (if None, no RAG context injected).
            temperature: Sampling temperature.
            max_new_tokens: Max new tokens to generate.
            top_p: Nucleus sampling p.

        Returns:
            Raw generated text.
        """
        model, tokenizer = self._load_model()

        # Build prompt with optional context
        if context:
            prompt = (
                "你是一位资深金融分析师。请根据以下上下文，针对问题给出逐步推理过程，"
                "将思考过程放在 <think> 标签内，最终答案放在 <answer> 标签内。\n\n"
                f"上下文：\n{context}\n\n"
                f"问题：{query}\n"
            )
        else:
            prompt = (
                "你是一位资深金融分析师。请针对以下问题给出逐步推理过程，"
                "将思考过程放在 <think> 标签内，最终答案放在 <answer> 标签内。\n\n"
                f"问题：{query}\n"
            )

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        generated = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )
        return generated.strip()

    # ------------------------------------------------------------------
    # Parsing & Validation
    # ------------------------------------------------------------------

    def parse_raw_output(self, raw_text: str, sources: Optional[List[str]] = None) -> Dict[str, Any]:
        """Parse raw model output into structured reasoning."""
        parser = self._load_parser()
        parsed = parser.parse(raw_text, sources=sources or [])
        return parsed.to_dict()

    def validate_chain(
        self,
        reasoning_steps: List[str],
        final_answer: str,
    ) -> Tuple[bool, str]:
        """Validate consistency between reasoning chain and final answer."""
        validator = self._load_validator()
        return validator.validate(reasoning_steps, final_answer)

    # ------------------------------------------------------------------
    # End-to-end: RAG + Generate + Parse + Validate
    # ------------------------------------------------------------------

    def answer(
        self,
        query: str,
        use_rag: bool = True,
        top_k: int = 5,
        rerank_top_k: Optional[int] = None,
        temperature: float = 0.6,
        max_new_tokens: int = 1024,
        top_p: float = 0.9,
    ) -> Dict[str, Any]:
        """
        Full pipeline: retrieve context (optional), generate, parse, validate.

        Returns:
            Dict with all intermediate and final results.
        """
        context = ""
        sources: List[str] = []
        chunks: List[Dict[str, Any]] = []

        if use_rag:
            try:
                chunks, context = self.retrieve(query, top_k=top_k, rerank_top_k=rerank_top_k)
                sources = list({c.get("source_url", "") for c in chunks if c.get("source_url")})
            except Exception as e:
                logger.warning(f"RAG retrieval failed: {e}; continuing without context")

        raw_output = self.generate(
            query=query,
            context=context or None,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            top_p=top_p,
        )

        parsed = self.parse_raw_output(raw_output, sources=sources)

        validated, validation_msg = self.validate_chain(
            parsed["reasoning_steps"],
            parsed["final_answer"],
        )

        return {
            "query": query,
            "reasoning_steps": parsed["reasoning_steps"],
            "final_answer": parsed["final_answer"],
            "confidence": parsed["confidence"],
            "fallback_triggered": parsed["fallback_triggered"],
            "fallback_reason": parsed.get("fallback_reason", ""),
            "sources": parsed.get("sources", []),
            "raw_output": raw_output,
            "validated": validated,
            "validation_message": validation_msg,
            "chunks": chunks,
            "context": context,
        }

    # ------------------------------------------------------------------
    # Index Management
    # ------------------------------------------------------------------

    def build_index(self, corpus_dir: Optional[str] = None) -> Dict[str, Any]:
        """Build FAISS index from corpus directory."""
        from rag.indexer import build_from_corpus_dir

        indexer = build_from_corpus_dir(corpus_dir)
        return {
            "success": True,
            "vectors_added": indexer.index.ntotal,
            "index_path": indexer.index_path,
            "metadata_path": indexer.metadata_path,
        }

    def get_index_status(self) -> Dict[str, Any]:
        """Get current index status."""
        if self._retriever is None:
            return {
                "index_loaded": False,
                "total_vectors": 0,
                "index_path": None,
                "metadata_path": None,
            }
        idx = self._retriever.indexer
        return {
            "index_loaded": idx.index is not None,
            "total_vectors": idx.index.ntotal if idx.index is not None else 0,
            "index_path": idx.index_path,
            "metadata_path": idx.metadata_path,
        }
