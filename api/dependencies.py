"""
全局单例管理
负责模型、检索器、解析器的懒加载与生命周期
"""

import logging
import torch
from pathlib import Path
from typing import Optional

from finance_deepseek.rag.retriever import DenseRetriever
from finance_deepseek.reasoning.chain_parser import ChainParser

logger = logging.getLogger(__name__)


class AppState:
    """
    应用全局状态（单例）
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self.model = None
        self.tokenizer = None
        self.retriever: Optional[DenseRetriever] = None
        self.chain_parser: Optional[ChainParser] = None
        self.adapter_path: Optional[str] = None
        self._initialized = True

    def load_model(self, base_model: str, adapter_path: str = None, quantization_config=None):
        """加载 4-bit 基座模型与可选 LoRA Adapter"""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel

        if self.model is not None:
            logger.info("Model already loaded")
            return

        logger.info(f"Loading base model: {base_model}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model,
            trust_remote_code=True,
            use_fast=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        load_kwargs = {
            "trust_remote_code": True,
        }
        if torch.cuda.is_available():
            load_kwargs["quantization_config"] = quantization_config
            load_kwargs["device_map"] = "auto"
            load_kwargs["dtype"] = torch.bfloat16
        else:
            logger.warning("CUDA not available, loading model in float32 CPU mode (slow)")
            load_kwargs["dtype"] = torch.float32
        
        self.model = AutoModelForCausalLM.from_pretrained(base_model, **load_kwargs)
        self.model.eval()

        if adapter_path and Path(adapter_path).exists():
            logger.info(f"Loading LoRA adapter: {adapter_path}")
            self.model = PeftModel.from_pretrained(self.model, adapter_path)
            self.adapter_path = str(adapter_path)
        else:
            logger.warning("No LoRA adapter found; using base distill model only")
            self.adapter_path = None

        logger.info("Model loaded successfully")

    def load_retriever(self):
        """加载 RAG 检索器"""
        if self.retriever is not None:
            return
        try:
            self.retriever = DenseRetriever()
            logger.info("Retriever loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load retriever: {e}")
            self.retriever = None

    def load_chain_parser(self, min_steps: int = 2):
        """加载推理链解析器"""
        if self.chain_parser is not None:
            return
        self.chain_parser = ChainParser(min_steps=min_steps)
        logger.info("Chain parser loaded")

    def unload(self):
        """释放显存"""
        self.model = None
        self.tokenizer = None
        self.retriever = None
        self.chain_parser = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("AppState unloaded")


# 全局单例实例
app_state = AppState()
