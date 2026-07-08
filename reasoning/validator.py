"""
推理链-答案一致性校验模块
检查推理链中的数值计算是否与最终答案一致
"""

import re
import logging
from typing import Tuple, Optional
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP

logger = logging.getLogger(__name__)


class ChainValidator:
    """
    校验推理链与最终答案的一致性
    """

    def __init__(self, numeric_tolerance: float = 0.01):
        self.numeric_tolerance = numeric_tolerance

    def validate(
        self,
        reasoning_steps: list,
        final_answer: str,
    ) -> Tuple[bool, str]:
        """
        校验推理链与答案是否一致

        Returns:
            (is_consistent, reason)
        """
        if not reasoning_steps:
            return False, "Empty reasoning steps"

        if not final_answer or not final_answer.strip():
            return False, "Empty final answer"

        # 提取推理链中所有数值
        chain_numbers = self._extract_numbers(" ".join(reasoning_steps))
        answer_numbers = self._extract_numbers(final_answer)

        if not answer_numbers:
            # 非数值答案，只做语义级基本检查（通过）
            return True, "Non-numeric answer; semantic consistency assumed"

        # 检查最终答案中的主要数值是否出现在推理链中
        primary_answer_num = answer_numbers[-1]  # 通常最后一个数值是答案
        matched = False
        for cn in chain_numbers:
            if self._close_enough(cn, primary_answer_num):
                matched = True
                break

        if not matched:
            msg = (
                f"Answer number {primary_answer_num} not found in reasoning chain numbers "
                f"({chain_numbers[-5:]})")
            logger.warning(msg)
            return False, msg

        return True, "Consistent"

    def _extract_numbers(self, text: str) -> list:
        """从文本中提取所有数值（支持千分位、百分号、货币符号）"""
        # 移除千分位逗号
        cleaned = text.replace(",", "")
        # 匹配数字（含小数、百分号、货币符号后的数字）
        pattern = r"[-+]?\d+\.?\d*%?"
        matches = re.findall(pattern, cleaned)
        numbers = []
        for m in matches:
            try:
                if "%" in m:
                    val = Decimal(m.replace("%", "")) / Decimal("100")
                else:
                    val = Decimal(m)
                numbers.append(float(val))
            except (InvalidOperation, ValueError):
                continue
        return numbers

    def _close_enough(self, a: float, b: float) -> bool:
        """判断两个数值是否在容忍度范围内相等"""
        if a == 0 and b == 0:
            return True
        if a == 0 or b == 0:
            return abs(a - b) <= self.numeric_tolerance
        rel_diff = abs(a - b) / max(abs(a), abs(b))
        return rel_diff <= self.numeric_tolerance

    def safe_numeric_compare(self, pred: str, gold: str) -> bool:
        """
        安全地比较两个数值字符串，用于评测 Numeric EM
        容忍相对误差 1%
        """
        pred_nums = self._extract_numbers(pred)
        gold_nums = self._extract_numbers(gold)
        if not pred_nums or not gold_nums:
            return False
        # 比较各自最后一个数值（通常为主要答案）
        return self._close_enough(pred_nums[-1], gold_nums[-1])
