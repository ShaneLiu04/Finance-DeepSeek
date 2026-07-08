"""
Cross-Encoder 精排模块
基于 sentence-transformers 的 CrossEncoder 对候选 chunk 进行重排序
"""

import logging
from typing import List, Dict

import numpy as np
import torch
from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """
    Cross-Encoder 重排序器

    使用 cross-encoder 模型对 (query, candidate_text) 对进行打分，
    获得比 bi-encoder 更精确的语义相关性排序。
    """

    DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def __init__(
        self,
        model_name: str = None,
        device: str = None,
        max_length: int = 512,
        batch_size: int = 16,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name or self.DEFAULT_MODEL
        self.max_length = max_length
        self.batch_size = batch_size
        self.model = self._load_model()

    def _load_model(self) -> CrossEncoder:
        """加载 CrossEncoder 模型"""
        try:
            model = CrossEncoder(
                self.model_name,
                device=self.device,
                max_length=self.max_length,
            )
            logger.info(
                f"CrossEncoder loaded: {self.model_name}, device={self.device}"
            )
            return model
        except Exception as e:
            logger.warning(f"Failed to load cross-encoder {self.model_name}: {e}")
            raise RuntimeError(
                f"CrossEncoder model '{self.model_name}' failed to load. "
                f"Please ensure the model name is valid and network is available."
            ) from e

    def rerank(
        self,
        query: str,
        candidates: List[Dict],
        top_k: int = None,
    ) -> List[Dict]:
        """
        对候选 chunk 进行 Cross-Encoder 重排序

        Args:
            query: 用户查询
            candidates: 候选 chunk 列表，每个 dict 至少包含 "text" 键
            top_k: 返回前 K 个结果，默认返回全部重排序后的结果

        Returns:
            按 cross-encoder 分数降序排列的候选列表
        """
        if not candidates:
            return []

        # 构建 (query, text) 对
        pairs = []
        for c in candidates:
            text = c.get("text", "")
            # 截断过长文本，避免超出 max_length
            if len(text) > self.max_length * 3:
                text = text[: self.max_length * 3]
            pairs.append((query, text))

        # 批量打分
        scores = self.model.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )

        # 将分数附加到候选并排序
        scored_candidates = []
        for candidate, score in zip(candidates, scores):
            item = dict(candidate)
            item["rerank_score"] = float(score)
            # 保留原始 bi-encoder 分数用于调试
            item["bi_encoder_score"] = item.get("score", 0.0)
            # 用 rerank_score 覆盖主 score，供下游使用
            item["score"] = float(score)
            scored_candidates.append(item)

        # 按 cross-encoder 分数降序排列
        scored_candidates.sort(key=lambda x: x["rerank_score"], reverse=True)

        if top_k is not None:
            scored_candidates = scored_candidates[:top_k]

        logger.debug(
            f"Reranked {len(candidates)} candidates -> {len(scored_candidates)} "
            f"(top score={scored_candidates[0]['rerank_score']:.4f} if scored_candidates else 'N/A')"
        )
        return scored_candidates
