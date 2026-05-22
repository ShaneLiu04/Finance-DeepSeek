"""
FAISS 索引构建模块
支持 IndexFlatIP（<5万条）与 IndexHNSWFlat（>=5万条）自动切换
"""

import os
import json
import logging
from pathlib import Path
from typing import List, Dict

import faiss
import numpy as np
import yaml

from .embeddings import EmbeddingProvider

logger = logging.getLogger(__name__)


def load_config() -> dict:
    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class FaissIndexer:
    """
    FAISS 向量索引管理器
    """

    def __init__(self, embedding_provider: EmbeddingProvider = None):
        self.cfg = load_config()["rag"]
        self.provider = embedding_provider or EmbeddingProvider(
            model_name=self.cfg["embedding_model"],
            fallback_model_name=self.cfg["fallback_embedding_model"],
        )
        self.index_path = self.cfg["index_path"]
        self.metadata_path = self.cfg["metadata_path"]
        self.index = None
        self.metadata: List[Dict] = []

    def build_index(self, chunks: List[Dict[str, str]]) -> faiss.Index:
        """
        从 chunk 列表构建 FAISS 索引（全量重建）

        Args:
            chunks: 每个元素为 dict，至少含 "text" 键
                    推荐额外包含 "source_url", "category", "title"

        Returns:
            faiss.Index
        """
        if not chunks:
            raise ValueError("No chunks provided for indexing")

        texts = [c["text"] for c in chunks]
        self.metadata = [
            {k: v for k, v in c.items() if k != "text"} for c in chunks
        ]

        logger.info(f"Encoding {len(texts)} chunks...")
        embeddings = self.provider.encode(texts, normalize=True)
        dim = embeddings.shape[1]
        n = len(embeddings)

        # 自动选择索引类型
        threshold_hnsw = 50000
        if n < threshold_hnsw:
            logger.info(f"Using IndexFlatIP (n={n} < {threshold_hnsw})")
            self.index = faiss.IndexFlatIP(dim)
        else:
            logger.info(f"Using IndexHNSWFlat (n={n} >= {threshold_hnsw})")
            m = 16
            ef_construction = 200
            self.index = faiss.IndexHNSWFlat(dim, m)
            self.index.hnsw.efConstruction = ef_construction
            self.index.metric_type = faiss.METRIC_INNER_PRODUCT

        self.index.add(embeddings)
        logger.info(f"Index built with {self.index.ntotal} vectors, dim={dim}")
        return self.index

    def add_chunks(self, chunks: List[Dict[str, str]]) -> int:
        """
        增量添加 chunk 到已有 FAISS 索引
        生产环境推荐方式，无需全量重建

        Args:
            chunks: 新增 chunk 列表

        Returns:
            新增向量数
        """
        if not chunks:
            return 0
        if self.index is None:
            logger.info("Index not initialized, building new index from chunks")
            self.build_index(chunks)
            return len(chunks)

        texts = [c["text"] for c in chunks]
        new_metadata = [{k: v for k, v in c.items() if k != "text"} for c in chunks]

        logger.info(f"Incrementally encoding {len(texts)} new chunks...")
        embeddings = self.provider.encode(texts, normalize=True)

        self.index.add(embeddings)
        self.metadata.extend(new_metadata)
        logger.info(f"Added {len(texts)} vectors, total={self.index.ntotal}")
        return len(texts)

    def save(self, index_path: str = None, metadata_path: str = None):
        """持久化索引与元数据"""
        index_path = index_path or self.index_path
        metadata_path = metadata_path or self.metadata_path

        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        faiss.write_index(self.index, index_path)
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)
        logger.info(f"Index saved to {index_path}, metadata to {metadata_path}")

    def load(self, index_path: str = None, metadata_path: str = None):
        """加载已有索引"""
        index_path = index_path or self.index_path
        metadata_path = metadata_path or self.metadata_path

        if not os.path.exists(index_path):
            raise FileNotFoundError(f"Index not found: {index_path}")
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata not found: {metadata_path}")

        self.index = faiss.read_index(index_path)
        with open(metadata_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)
        logger.info(f"Index loaded: {self.index.ntotal} vectors")
        return self.index


def build_from_corpus_dir(corpus_dir: str, indexer: FaissIndexer = None) -> FaissIndexer:
    """
    从语料目录一键构建索引
    目录下应为 .jsonl 文件，每行 {"text": "...", "source_url": "...", ...}
    """
    cfg = load_config()["rag"]
    indexer = indexer or FaissIndexer()
    corpus_dir = corpus_dir or cfg["corpus_dir"]

    chunks = []
    corpus_path = Path(corpus_dir)
    if not corpus_path.exists():
        logger.warning(f"Corpus dir not found: {corpus_dir}, creating placeholder")
        corpus_path.mkdir(parents=True, exist_ok=True)
        # 创建示例语料
        _create_demo_corpus(corpus_path)

    for fp in sorted(corpus_path.glob("*.jsonl")):
        with open(fp, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if "text" in obj:
                    chunks.append(obj)

    if not chunks:
        raise ValueError(f"No valid chunks found in {corpus_dir}")

    indexer.build_index(chunks)
    indexer.save()
    return indexer


def _create_demo_corpus(path: Path):
    """生成示例语料，确保项目可一键运行"""
    demo = [
        {
            "text": "市盈率（P/E Ratio）是股价与每股收益的比率，计算公式为：市盈率 = 股票价格 / 每股收益（EPS）。该指标用于衡量股票估值水平。",
            "source_url": "https://www.investopedia.com/terms/p/price-earningsratio.asp",
            "category": "估值指标",
            "title": "市盈率 P/E Ratio",
        },
        {
            "text": "净资产收益率（ROE）= 净利润 / 平均股东权益。杜邦分析将 ROE 拆解为销售净利率 × 资产周转率 × 权益乘数。",
            "source_url": "https://www.investopedia.com/terms/r/returnonequity.asp",
            "category": "盈利能力",
            "title": "ROE 与杜邦分析",
        },
        {
            "text": "复利计算公式：A = P * (1 + r/n)^(nt)。其中 P 为本金，r 为年利率，n 为每年复利次数，t 为投资年限。",
            "source_url": "https://www.investopedia.com/terms/c/compoundinterest.asp",
            "category": "基础概念",
            "title": "复利计算",
        },
        {
            "text": "每股收益（EPS）= （净利润 - 优先股股利）/ 流通在外普通股加权平均股数。EPS 是衡量上市公司盈利能力的重要指标。",
            "source_url": "https://www.investopedia.com/terms/e/eps.asp",
            "category": "估值指标",
            "title": "每股收益 EPS",
        },
        {
            "text": "折现现金流（DCF）模型通过将未来自由现金流以适当折现率折现至当前，计算企业内在价值。基本公式：PV = CF / (1+r)^t。",
            "source_url": "https://www.investopedia.com/terms/d/dcf.asp",
            "category": "估值方法",
            "title": "DCF 折现现金流",
        },
    ]
    demo_path = path / "demo_corpus.jsonl"
    with open(demo_path, "w", encoding="utf-8") as f:
        for item in demo:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    logger.info(f"Demo corpus created at {demo_path}")
