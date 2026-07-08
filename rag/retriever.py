"""
稠密检索 (Dense Retrieval) 模块
支持 Top-K 检索与可选 Cross-Encoder 轻量级精排
"""

import logging
from typing import List, Dict, Tuple

import numpy as np
import faiss
import yaml
from pathlib import Path

from .embeddings import EmbeddingProvider
from .indexer import FaissIndexer, load_config
from .reranker import CrossEncoderReranker

logger = logging.getLogger(__name__)


class DenseRetriever:
    """
    基于 FAISS + Embedding 的稠密检索器
    支持可选 Cross-Encoder 精排
    """

    def __init__(
        self,
        indexer: FaissIndexer = None,
        embedding_provider: EmbeddingProvider = None,
        reranker: CrossEncoderReranker = None,
    ):
        self.cfg = load_config()["rag"]
        self.provider = embedding_provider or EmbeddingProvider(
            model_name=self.cfg["embedding_model"],
            fallback_model_name=self.cfg["fallback_embedding_model"],
        )
        self.indexer = indexer or FaissIndexer(embedding_provider=self.provider)

        # 初始化可选的 Cross-Encoder 精排器
        self.reranker = None
        if self.cfg.get("use_reranker", False):
            rerank_cfg = self.cfg.get("reranker", {})
            rerank_model = rerank_cfg.get("model_name")
            if reranker is not None:
                self.reranker = reranker
            elif rerank_model:
                self.reranker = CrossEncoderReranker(
                    model_name=rerank_model,
                    max_length=rerank_cfg.get("max_length", 512),
                    batch_size=rerank_cfg.get("batch_size", 16),
                )
            else:
                self.reranker = CrossEncoderReranker()

        # 尝试加载已有索引
        try:
            self.indexer.load()
        except FileNotFoundError:
            logger.warning("No existing FAISS index found. Please run indexer.build_from_corpus_dir() first.")
            self.indexer.index = None

    def retrieve(
        self,
        query: str,
        top_k: int = None,
        rerank_top_k: int = None,
    ) -> List[Dict]:
        """
        检索与查询最相关的 chunks

        Args:
            query: 用户查询
            top_k: 初始召回数量，默认读取 config
            rerank_top_k: 精排后返回数量，默认读取 config

        Returns:
            List[dict]，每个 dict 包含 chunk 元数据 + "score" 相似度分数
        """
        if self.indexer.index is None:
            raise RuntimeError("FAISS index not loaded. Build or load index first.")

        top_k = top_k or self.cfg["top_k"]
        rerank_top_k = rerank_top_k or self.cfg.get("rerank_top_k", top_k)

        query_vec = self.provider.encode_query(query, normalize=True)
        query_vec = np.expand_dims(query_vec, axis=0).astype("float32")

        scores, indices = self.indexer.index.search(query_vec, top_k)
        scores = scores[0]
        indices = indices[0]

        results = []
        for score, idx in zip(scores, indices):
            if idx < 0 or idx >= len(self.indexer.metadata):
                continue
            meta = dict(self.indexer.metadata[idx])
            meta["score"] = float(score)
            results.append(meta)

        # 可选：简单精排（此处基于分数截断，若启用 cross-encoder 可扩展）
        if self.cfg.get("use_reranker", False):
            results = self._rerank(query, results, rerank_top_k)
        else:
            results = results[:rerank_top_k]

        logger.debug(f"Retrieved {len(results)} chunks for query: {query[:40]}...")
        return results

    def _rerank(self, query: str, candidates: List[Dict], top_k: int) -> List[Dict]:
        """
        调用 Cross-Encoder 精排器对候选结果重排序

        若 reranker 未初始化（如模型加载失败），则回退到按原分数截断。
        """
        if self.reranker is None:
            logger.warning("Reranker requested but not initialized; falling back to bi-encoder scores")
            return candidates[:top_k]
        return self.reranker.rerank(query, candidates, top_k=top_k)

    def format_context(self, chunks: List[Dict]) -> str:
        """
        将检索结果格式化为 Prompt 可用的上下文字符串
        """
        parts = []
        for i, chunk in enumerate(chunks, 1):
            text = chunk.get("text", "")
            source = chunk.get("source_url", "未知来源")
            parts.append(f"[{i}] {text}\n（来源: {source}）")
        return "\n\n".join(parts)
