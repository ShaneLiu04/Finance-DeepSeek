#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
推理链解析模块单元测试
"""

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT.parent))

from finance_deepseek.reasoning.chain_parser import ChainParser, ParsedReasoning


class TestChainParser(unittest.TestCase):

    def setUp(self):
        self.parser = ChainParser(min_steps=2)

    def test_parse_standard(self):
        raw = (
            "<think>\n"
            "步骤1: 识别关键财务指标，股价 = 50 元，EPS = 2.5 元。\n"
            "步骤2: 代入市盈率公式 P/E = 50 / 2.5 = 20。\n"
            "步骤3: 验证结果合理性。\n"
            "</think>\n"
            "<answer>\n20.0\n</answer>"
        )
        parsed = self.parser.parse(raw)
        self.assertIsInstance(parsed, ParsedReasoning)
        self.assertTrue(len(parsed.reasoning_steps) >= 2)
        self.assertEqual(parsed.final_answer.strip(), "20.0")
        self.assertFalse(parsed.fallback_triggered)
        self.assertGreaterEqual(parsed.confidence, 0.9)

    def test_parse_missing_think(self):
        raw = "Final answer is 15.5"
        parsed = self.parser.parse(raw)
        self.assertTrue(parsed.fallback_triggered)
        self.assertEqual(parsed.reasoning_steps, [])
        self.assertEqual(parsed.final_answer.strip(), "15.5")

    def test_parse_missing_answer(self):
        raw = (
            "<think>\n"
            "Step1: extract principal 10000, rate 5%.\n"
            "Step2: calculate 10000 * 1.05^3 = 11576.25.\n"
            "</think>\n"
            "11576.25"
        )
        parsed = self.parser.parse(raw)
        self.assertTrue(parsed.fallback_triggered)
        self.assertIn("Missing <answer>", parsed.fallback_reason)
        self.assertEqual(parsed.final_answer.strip(), "11576.25")

    def test_validate_format(self):
        valid, msg = self.parser.validate_format("<think>ok</think><answer>ok</answer>")
        self.assertTrue(valid)
        self.assertEqual(msg, "Format valid")

        invalid, msg = self.parser.validate_format("no tags")
        self.assertFalse(invalid)
        self.assertIn("Missing tags", msg)

    def test_split_steps_numbered(self):
        think = "Step1: A. Step2: B. Step3: C."
        steps = self.parser._split_steps(think)
        self.assertTrue(len(steps) >= 2)


class TestChainValidator(unittest.TestCase):

    def setUp(self):
        from finance_deepseek.reasoning.validator import ChainValidator
        self.validator = ChainValidator(numeric_tolerance=0.01)

    def test_validate_consistent(self):
        steps = ["Step1: price 50, EPS 2.5", "Step2: compute 50/2.5=20"]
        answer = "20.0"
        ok, reason = self.validator.validate(steps, answer)
        self.assertTrue(ok)
        self.assertEqual(reason, "Consistent")

    def test_validate_inconsistent(self):
        steps = ["Step1: price 50, EPS 2.5", "Step2: compute 50/2.5=20"]
        answer = "25.0"
        ok, reason = self.validator.validate(steps, answer)
        self.assertFalse(ok)
        self.assertIn("not found", reason)

    def test_safe_numeric_compare(self):
        self.assertTrue(self.validator.safe_numeric_compare("20.0", "20"))
        self.assertTrue(self.validator.safe_numeric_compare("11576.25", "11576.3"))
        self.assertFalse(self.validator.safe_numeric_compare("10", "20"))


if __name__ == "__main__":
    unittest.main()
