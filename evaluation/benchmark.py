"""
批量评测脚本
自动执行消融实验并输出对比表格与可视化
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import List, Dict

import yaml
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from finance_deepseek.api.inference import InferenceEngine
from finance_deepseek.api.dependencies import app_state
from finance_deepseek.evaluation.metrics import MetricsSuite
from finance_deepseek.rag.retriever import DenseRetriever
from finance_deepseek.reasoning.chain_parser import ChainParser

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def load_config() -> dict:
    with open(PROJECT_ROOT / "config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_test_data(path: str) -> List[Dict]:
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            samples.append(obj)
    return samples


def run_benchmark(
    test_data: List[Dict],
    engine: InferenceEngine,
    mode: str,
    max_samples: int = None,
) -> Dict:
    """
    对测试集执行批量推理与评测
    """
    max_samples = max_samples or len(test_data)
    predictions = []
    references = []
    raw_outputs = []
    contexts_list = []

    for i, item in enumerate(test_data[:max_samples]):
        messages = item.get("messages", [])
        # 取最后一条 user message 作为 query
        if not messages:
            continue
        # 构造推理请求
        result = engine.run_inference(
            messages=messages,
            mode=mode,
            temperature=0.6,
            max_new_tokens=1024,
        )
        raw = result["raw"]
        raw_outputs.append(raw)

        # 提取预测答案
        if result["parsed"]:
            pred = result["parsed"]["final_answer"]
        else:
            # 尝试从 raw 中简单提取最后一行
            pred = raw.strip().split("\n")[-1]
        predictions.append(pred)

        # 提取参考答案（从 assistant message 中解析 <answer>）
        ref = ""
        for m in messages:
            if m.get("role") == "assistant":
                content = m.get("content", "")
                import re
                m_ans = re.search(r"<answer>(.*?)</answer>", content, re.DOTALL)
                if m_ans:
                    ref = m_ans.group(1).strip()
                else:
                    ref = content.strip()
                break
        references.append(ref)

        # 记录上下文（RAG 模式下）
        if mode in ("rag", "rag+reasoning") and engine.retriever:
            user_msgs = [m for m in messages if m.get("role") == "user"]
            if user_msgs:
                query = user_msgs[-1].get("content", "")
                try:
                    chunks = engine.retriever.retrieve(query)
                    ctxs = [c.get("text", "") for c in chunks]
                except Exception:
                    ctxs = []
                contexts_list.append(ctxs)
        else:
            contexts_list.append([])

    metrics = MetricsSuite()
    scores = metrics.compute_all(
        predictions=predictions,
        references=references,
        raw_outputs=raw_outputs,
        contexts_list=contexts_list if contexts_list[0] else None,
    )
    scores["mode"] = mode
    scores["samples"] = len(predictions)
    return scores


def run_ablation(cfg: dict, test_path: str, max_samples: int = 100) -> List[Dict]:
    """
    执行消融实验表格中的全部配置
    """
    # 确保模型已加载
    if app_state.model is None:
        adapter_dir = Path(cfg["lora"]["adapter_output_dir"]) / "final_adapter"
        from transformers import BitsAndBytesConfig
        import torch
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=cfg["quantization"]["load_in_4bit"],
            bnb_4bit_quant_type=cfg["quantization"]["bnb_4bit_quant_type"],
            bnb_4bit_use_double_quant=cfg["quantization"]["bnb_4bit_use_double_quant"],
            bnb_4bit_compute_dtype=getattr(torch, cfg["quantization"]["bnb_4bit_compute_dtype"]),
        )
        app_state.load_model(
            base_model=cfg["model"]["base_model"],
            adapter_path=adapter_dir,
            quantization_config=bnb_config,
        )
        app_state.load_retriever()
        app_state.load_chain_parser()

    test_data = load_test_data(test_path)

    results = []
    modes = [
        ("closed-book", "Closed-Book (Base Distill)"),
        ("rag", "RAG (Base Distill)"),
        ("rag+reasoning", "RAG + QLoRA + Chain Parser"),
    ]

    for mode, label in modes:
        logger.info(f"Running ablation: {label} (mode={mode})")
        engine = InferenceEngine(
            model=app_state.model,
            tokenizer=app_state.tokenizer,
            retriever=app_state.retriever if mode != "closed-book" else None,
            chain_parser=app_state.chain_parser if mode == "rag+reasoning" else None,
        )
        scores = run_benchmark(test_data, engine, mode=mode, max_samples=max_samples)
        scores["label"] = label
        results.append(scores)
        logger.info(f"Results: {scores}")

    return results


def plot_results(results: List[Dict], output_dir: str):
    """
    绘制消融实验对比柱状图
    """
    os.makedirs(output_dir, exist_ok=True)
    labels = [r["label"] for r in results]
    metrics_keys = ["exact_match", "numeric_em", "faithfulness_at_8", "chain_completeness"]
    available_keys = [k for k in metrics_keys if k in results[0]]

    x = np.arange(len(labels))
    width = 0.2
    fig, ax = plt.subplots(figsize=(10, 6))

    for i, key in enumerate(available_keys):
        values = [r.get(key, 0.0) for r in results]
        ax.bar(x + i * width, values, width, label=key)

    ax.set_ylabel("Score")
    ax.set_title("Finance-DeepSeek Ablation Study")
    ax.set_xticks(x + width * (len(available_keys) - 1) / 2)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.legend()
    ax.set_ylim(0, 1.0)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "ablation_comparison.png"))
    plt.close()
    logger.info(f"Plot saved to {output_dir}/ablation_comparison.png")


def print_table(results: List[Dict]):
    """打印消融实验 Markdown 表格"""
    print("\n### 消融实验结果\n")
    print("| 配置 | EM | Numeric EM | Faithfulness@8 | Chain Completeness | 备注 |")
    print("|---|---|---|---|---|---|")
    for r in results:
        em = f"{r.get('exact_match', 0):.3f}"
        nem = f"{r.get('numeric_em', 0):.3f}"
        faith = f"{r.get('faithfulness_at_8', 0):.3f}" if "faithfulness_at_8" in r else "-"
        chain = f"{r.get('chain_completeness', 0):.3f}" if "chain_completeness" in r else "-"
        note = r.get("label", "")
        print(f"| {note} | {em} | {nem} | {faith} | {chain} | - |")


def main():
    parser = argparse.ArgumentParser(description="Finance-DeepSeek Benchmark")
    parser.add_argument("--test_data", type=str, default=None, help="Test JSONL path")
    parser.add_argument("--max_samples", type=int, default=100, help="Max samples to evaluate")
    parser.add_argument("--output_dir", type=str, default="./evaluation/results", help="Output dir")
    args = parser.parse_args()

    cfg = load_config()
    test_path = args.test_data or cfg["evaluation"]["test_data_path"]
    if not os.path.exists(test_path):
        logger.error(f"Test data not found: {test_path}")
        # 使用训练数据作为演示
        test_path = cfg["training"]["train_data_path"]
        logger.warning(f"Falling back to train data for demo: {test_path}")

    results = run_ablation(cfg, test_path, max_samples=args.max_samples)
    print_table(results)
    plot_results(results, args.output_dir)

    # 保存 JSON
    json_path = os.path.join(args.output_dir, "ablation_results.json")
    os.makedirs(args.output_dir, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info(f"Results saved to {json_path}")


if __name__ == "__main__":
    main()
