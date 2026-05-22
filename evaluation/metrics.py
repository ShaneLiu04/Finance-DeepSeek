"""
评测指标模块
涵盖 EM, Numeric EM, Faithfulness@8, Chain Completeness, BLEU-4, ROUGE-L, Recall@K, MRR
"""

import re
import logging
from typing import List, Dict, Optional
from decimal import Decimal

from rouge_score import rouge_scorer
import sacrebleu
import numpy as np

from finance_deepseek.reasoning.validator import ChainValidator

logger = logging.getLogger(__name__)


class MetricsSuite:
    """
    多维评测指标套件
    """

    def __init__(self, numeric_tolerance: float = 0.01):
        self.numeric_tolerance = numeric_tolerance
        self.validator = ChainValidator(numeric_tolerance=numeric_tolerance)
        self.rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        self._nli_model = None

    def exact_match(self, predictions: List[str], references: List[str]) -> float:
        """字符串完全匹配率"""
        assert len(predictions) == len(references)
        matches = sum(p.strip() == r.strip() for p, r in zip(predictions, references))
        return matches / len(predictions) if predictions else 0.0

    def numeric_em(self, predictions: List[str], references: List[str]) -> float:
        """数值答案相对误差 1% 内匹配率"""
        assert len(predictions) == len(references)
        matches = 0
        for p, r in zip(predictions, references):
            if self.validator.safe_numeric_compare(p, r):
                matches += 1
        return matches / len(predictions) if predictions else 0.0

    def chain_completeness(self, raw_outputs: List[str], min_steps: int = 2) -> float:
        """推理链完整度：包含非空 <think> 且步骤数 >= min_steps"""
        complete = 0
        for text in raw_outputs:
            match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
            if match:
                think_text = match.group(1)
                lines = [l.strip() for l in think_text.splitlines() if l.strip()]
                if len(lines) >= min_steps:
                    complete += 1
        return complete / len(raw_outputs) if raw_outputs else 0.0

    def recall_at_k(
        self,
        retrieved_ids_list: List[List[str]],
        relevant_ids_list: List[List[str]],
        k: int = 8,
    ) -> float:
        """
        Recall@K：标准答案相关文档是否落在 Top-K 检索结果内
        Args:
            retrieved_ids_list: 每个查询检索到的文档 ID 列表（按相似度排序）
            relevant_ids_list: 每个查询的相关文档 ID 列表
        """
        assert len(retrieved_ids_list) == len(relevant_ids_list)
        recalls = []
        for retrieved, relevant in zip(retrieved_ids_list, relevant_ids_list):
            top_k = set(retrieved[:k])
            rel_set = set(relevant)
            if not rel_set:
                continue
            recall = len(top_k & rel_set) / len(rel_set)
            recalls.append(recall)
        return sum(recalls) / len(recalls) if recalls else 0.0

    def mrr(
        self,
        retrieved_ids_list: List[List[str]],
        relevant_ids_list: List[List[str]],
    ) -> float:
        """
        MRR（Mean Reciprocal Rank）：首个相关文档排名的倒数均值
        """
        assert len(retrieved_ids_list) == len(relevant_ids_list)
        rr_scores = []
        for retrieved, relevant in zip(retrieved_ids_list, relevant_ids_list):
            rel_set = set(relevant)
            for rank, doc_id in enumerate(retrieved, start=1):
                if doc_id in rel_set:
                    rr_scores.append(1.0 / rank)
                    break
            else:
                rr_scores.append(0.0)
        return sum(rr_scores) / len(rr_scores) if rr_scores else 0.0

    def _get_nli_model(self):
        """延迟加载 NLI 模型，用于 Faithfulness 判断"""
        if self._nli_model is not None:
            return self._nli_model
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            model_name = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSequenceClassification.from_pretrained(model_name)
            if hasattr(model, "to"):
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
                model = model.to(device)
            self._nli_model = (tokenizer, model)
            logger.info("NLI model loaded for Faithfulness evaluation")
            return self._nli_model
        except Exception as e:
            logger.warning(f"NLI model load failed: {e}, falling back to heuristic")
            return None

    def _nli_entailment_score(self, premise: str, hypothesis: str) -> float:
        """
        使用 NLI 模型判断 premise（上下文）是否蕴含 hypothesis（生成文本）
        返回 entailment 概率（0~1）
        """
        nli = self._get_nli_model()
        if nli is None:
            return -1.0  # 标记为未使用
        tokenizer, model = nli
        import torch

        inputs = tokenizer(
            premise,
            hypothesis,
            truncation=True,
            return_tensors="pt",
            max_length=512,
        )
        if hasattr(model, "device"):
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
            # 标签顺序: entailment=0, neutral=1, contradiction=2
            entailment_prob = probs[0][0].item()
        return entailment_prob

    def faithfulness_at_k(
        self,
        predictions: List[str],
        contexts_list: List[List[str]],
        k: int = 8,
        use_nli: bool = True,
    ) -> float:
        """
        事实保真度 Faithfulness@K
        生产级实现：优先使用 NLI 模型判断 entailment；失败时回退到启发式匹配
        """
        faithful = 0
        for pred, ctxs in zip(predictions, contexts_list):
            ctx_text = " ".join(ctxs[:k])
            if not ctx_text.strip():
                faithful += 1
                continue

            if use_nli:
                score = self._nli_entailment_score(ctx_text, pred)
                if score >= 0:
                    # NLI 可用：entailment 概率 >= 0.5 视为 faithful
                    if score >= 0.5:
                        faithful += 1
                    continue
                # NLI 失败，回退启发式

            # 启发式回退：检查预测中的关键数字/实体是否出现在上下文中
            pred_tokens = set(self._extract_simple_tokens(pred))
            ctx_tokens = set(self._extract_simple_tokens(ctx_text))
            if not pred_tokens:
                faithful += 1
                continue
            overlap = len(pred_tokens & ctx_tokens) / len(pred_tokens)
            if overlap >= 0.5:
                faithful += 1

        return faithful / len(predictions) if predictions else 0.0

    def bleu4(self, predictions: List[str], references: List[str]) -> float:
        """sacrebleu BLEU-4"""
        refs = [[r] for r in references]
        bleu = sacrebleu.corpus_bleu(predictions, list(zip(*refs)))
        return bleu.score

    def rouge_l(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """ROUGE-L F1 均值"""
        scores = [self.rouge.score(ref, pred)["rougeL"].fmeasure for pred, ref in zip(predictions, references)]
        return {"rouge_l_f1": sum(scores) / len(scores) if scores else 0.0}

    def _extract_simple_tokens(self, text: str) -> List[str]:
        """提取简单 token 用于启发式匹配"""
        numbers = re.findall(r"\d+\.?\d*", text.replace(",", ""))
        words = re.findall(r"[A-Za-z]{4,}", text)
        return numbers + [w.lower() for w in words]

    def compute_all(
        self,
        predictions: List[str],
        references: List[str],
        raw_outputs: List[str] = None,
        contexts_list: List[List[str]] = None,
        retrieved_ids_list: List[List[str]] = None,
        relevant_ids_list: List[List[str]] = None,
    ) -> Dict[str, float]:
        """一键计算所有指标"""
        results = {
            "exact_match": self.exact_match(predictions, references),
            "numeric_em": self.numeric_em(predictions, references),
            "bleu4": self.bleu4(predictions, references),
        }
        results.update(self.rouge_l(predictions, references))

        if raw_outputs:
            results["chain_completeness"] = self.chain_completeness(raw_outputs)
        if contexts_list:
            results["faithfulness_at_8"] = self.faithfulness_at_k(predictions, contexts_list, k=8)
        if retrieved_ids_list and relevant_ids_list:
            results["recall_at_8"] = self.recall_at_k(retrieved_ids_list, relevant_ids_list, k=8)
            results["mrr"] = self.mrr(retrieved_ids_list, relevant_ids_list)

        return results
