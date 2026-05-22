"""
推理链解析引擎 (Reasoning Chain Parser)
负责从 DeepSeek-R1 蒸馏模型的输出中提取 <think>...</think> 与 <answer>...</answer>
"""

import re
import json
import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class ParsedReasoning:
    """结构化推理结果"""
    reasoning_steps: List[str]
    final_answer: str
    raw_think: str
    raw_answer: str
    confidence: float = 0.95
    sources: List[str] = None
    fallback_triggered: bool = False
    fallback_reason: str = ""

    def to_dict(self) -> Dict:
        d = asdict(self)
        return d

    def to_json(self, ensure_ascii: bool = False, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=ensure_ascii, indent=indent)


class ChainParser:
    """
    推理链解析器
    支持正则提取、Fallback 策略、步骤拆分与结构化输出
    """

    THINK_START = "<think>"
    THINK_END = "</think>"
    ANSWER_START = "<answer>"
    ANSWER_END = "</answer>"

    def __init__(self, min_steps: int = 2):
        self.min_steps = min_steps

    def parse(self, raw_text: str, sources: List[str] = None) -> ParsedReasoning:
        """
        解析模型原始输出

        Args:
            raw_text: 模型生成的完整文本
            sources: 可选的溯源 URL 列表

        Returns:
            ParsedReasoning 结构化对象
        """
        raw_text = raw_text.strip()
        think_content, answer_content = self._extract_tags(raw_text)

        fallback_triggered = False
        fallback_reason = ""

        # Fallback 1: 无 <think> 标签
        if think_content is None:
            fallback_triggered = True
            fallback_reason = "Missing <think> tags; treating full text as final answer"
            logger.warning(fallback_reason)
            think_content = ""
            # 不直接赋值 answer_content，让 Fallback 2 尝试提取结构化答案
            answer_content = None

        # Fallback 2: 无 <answer> 标签
        if answer_content is None or answer_content.strip() == "":
            fallback_triggered = True
            fallback_reason += "; Missing <answer> tags; extracting last numeric/sentence after </think>"
            logger.warning(fallback_reason)
            answer_content = self._extract_fallback_answer(raw_text, think_content)

        steps = self._split_steps(think_content)

        # Fallback 3: 步骤数不足
        if len(steps) < self.min_steps and not fallback_triggered:
            logger.warning(f"Reasoning steps ({len(steps)}) below minimum ({self.min_steps})")

        confidence = self._estimate_confidence(steps, fallback_triggered)

        return ParsedReasoning(
            reasoning_steps=steps,
            final_answer=answer_content.strip(),
            raw_think=think_content.strip(),
            raw_answer=answer_content.strip(),
            confidence=confidence,
            sources=sources or [],
            fallback_triggered=fallback_triggered,
            fallback_reason=fallback_reason.strip("; "),
        )

    def _extract_tags(self, text: str) -> Tuple[Optional[str], Optional[str]]:
        """
        使用正则提取 think 与 answer 内容
        返回 (think_content, answer_content)
        """
        # 非贪婪匹配 <think>...</think>
        think_match = re.search(
            re.escape(self.THINK_START) + r"(.*?)" + re.escape(self.THINK_END),
            text,
            re.DOTALL,
        )
        answer_match = re.search(
            re.escape(self.ANSWER_START) + r"(.*?)" + re.escape(self.ANSWER_END),
            text,
            re.DOTALL,
        )

        think_content = think_match.group(1).strip() if think_match else None
        answer_content = answer_match.group(1).strip() if answer_match else None
        return think_content, answer_content

    def _split_steps(self, think_text: str) -> List[str]:
        """
        将 think 文本拆分为步骤列表
        支持：换行拆分、中文/阿拉伯数字编号拆分、句号拆分
        """
        if not think_text:
            return []

        # 先按换行拆分
        lines = [line.strip() for line in think_text.splitlines() if line.strip()]
        if not lines:
            return []

        # 若行数过少，尝试按编号进一步拆分
        steps = []
        for line in lines:
            # 匹配 "步骤1:", "步骤1：", "1.", "(1)", "①" 等前缀并拆分
            # 使用更宽松的正则，捕获步骤编号后的内容
            split_by_number = re.split(
                r"(?:步骤\s*\d+[\s:：.]+)|(?:Step\s*\d+[\s:：.]+)|(?:\d+[:：.]\s+)|(?:\(\d+\)\s+)|(?:[①②③④⑤⑥⑦⑧⑨⑩]\s*)",
                line,
            )
            split_by_number = [s.strip() for s in split_by_number if s.strip()]
            if len(split_by_number) > 1:
                steps.extend(split_by_number)
            else:
                # 若无编号，尝试按中文句号拆分（单行长文本）
                sentences = [s.strip() for s in line.split("。") if s.strip()]
                if len(sentences) > 1:
                    steps.extend(sentences)
                else:
                    steps.append(line)

        # 去重并过滤过短片段（允许单个中文字符+标点的短步骤）
        filtered = [s for s in steps if len(s) >= 2]
        return filtered

    def _extract_fallback_answer(self, raw_text: str, think_content: str) -> str:
        """
        Fallback：从 </think> 之后提取最后一行包含数字或句子的内容作为答案
        若无 </think>，尝试从全文中提取最后的数值或简短结论
        """
        # 定位 </think> 之后的内容
        idx = raw_text.find(self.THINK_END)
        if idx != -1:
            after_think = raw_text[idx + len(self.THINK_END):]
        else:
            after_think = raw_text

        # 尝试提取 <answer> 缺失时的内容
        lines = [l.strip() for l in after_think.splitlines() if l.strip()]
        if not lines:
            return ""

        # 优先返回最后一行（通常包含最终答案）
        candidate = lines[-1]

        # 若最后一行过短，尝试倒数第二行
        if len(candidate) < 2 and len(lines) >= 2:
            candidate = lines[-2]

        # 尝试提取行中最后的数值作为更干净的答案
        nums = re.findall(r"[-+]?\d+\.?\d*", candidate.replace(",", ""))
        if nums:
            # 若整行较短或包含答案关键词，提取数字作为干净答案
            if len(candidate) <= 22 or "答案" in candidate or "结果" in candidate or "answer" in candidate.lower():
                return nums[-1]

        return candidate

    def _estimate_confidence(self, steps: List[str], fallback: bool) -> float:
        """
        简单的启发式置信度估计
        """
        base = 0.95
        if fallback:
            base -= 0.15
        if len(steps) < self.min_steps:
            base -= 0.10
        if len(steps) >= 3:
            base += 0.03
        return max(0.0, min(1.0, base))

    def validate_format(self, raw_text: str) -> Tuple[bool, str]:
        """
        快速校验文本是否包含标准标签
        返回 (is_valid, message)
        """
        has_think = self.THINK_START in raw_text and self.THINK_END in raw_text
        has_answer = self.ANSWER_START in raw_text and self.ANSWER_END in raw_text
        if has_think and has_answer:
            return True, "Format valid"
        return False, f"Missing tags: think={has_think}, answer={has_answer}"
