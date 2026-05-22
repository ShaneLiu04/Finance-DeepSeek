#!/usr/bin/env python3
"""
一键生成 SFT 数据与 FAISS 索引
"""

import sys
from pathlib import Path

# 添加项目根目录到路径，支持直接运行
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root.parent))

from finance_deepseek.training.data_generator import SFTDataGenerator, load_config
from finance_deepseek.rag.indexer import build_from_corpus_dir


def main():
    cfg = load_config()
    gen = SFTDataGenerator()

    # 生成基础 Alpaca 数据
    gen.generate_alpaca_base(
        output_path=cfg["training"]["train_data_path"].replace("with_think", "alpaca_base"),
        num_samples=500,
    )

    # 生成带推理链的 FinQA 风格数据
    gen.generate_from_finqa(
        finqa_path="./data/finetune/finqa_raw.jsonl",
        output_path=cfg["training"]["train_data_path"],
        max_samples=2000,
    )

    # 构建 FAISS 索引（若语料目录为空会自动创建 demo 语料）
    build_from_corpus_dir(cfg["rag"]["corpus_dir"])
    print("Data generation and indexing complete.")


if __name__ == "__main__":
    main()
