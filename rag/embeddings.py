"""
Embedding 封装模块
支持主选金融 Embedding 与 FinBERT 备选回退
"""

import os
import logging
from typing import List
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class EmbeddingProvider:
    """
    统一的 Embedding 提供者，支持自动回退策略
    """

    def __init__(self, model_name: str, fallback_model_name: str = "yiyanghkust/finbert-tone", device: str = "cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model = self._load_model(model_name, fallback_model_name)
        self.embedding_dim = self.model.get_embedding_dimension()
        logger.info(f"Embedding model loaded: dim={self.embedding_dim}, device={self.device}")

    def _load_model(self, primary: str, fallback: str):
        """尝试加载主模型，失败则回退"""
        for name in [primary, fallback]:
            try:
                model = SentenceTransformer(name, device=self.device, trust_remote_code=True)
                logger.info(f"Successfully loaded embedding model: {name}")
                return model
            except Exception as e:
                logger.warning(f"Failed to load {name}: {e}")
        raise RuntimeError("All embedding models failed to load")

    def encode(self, texts: List[str], normalize: bool = True, batch_size: int = 32) -> np.ndarray:
        """
        将文本列表编码为向量

        Args:
            texts: 文本列表
            normalize: 是否做 L2 归一化（用于 InnerProduct 近似余弦相似度）
            batch_size: 编码批次大小

        Returns:
            numpy array of shape (len(texts), embedding_dim)
        """
        if not texts:
            return np.zeros((0, self.embedding_dim), dtype=np.float32)

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=normalize,
            device=self.device,
        )
        return embeddings.astype(np.float32)

    def encode_query(self, query: str, normalize: bool = True) -> np.ndarray:
        """单条查询编码"""
        return self.encode([query], normalize=normalize)[0]
