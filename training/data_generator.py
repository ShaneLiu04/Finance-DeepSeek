"""
带  标签的 SFT 数据生成器
利用教师模型（DeepSeek-R1 API / 本地 7B 蒸馏版）生成包含结构化推理链的样本
"""

import os
import json
import logging
import re
from pathlib import Path
from typing import List, Dict, Optional
from decimal import Decimal, ROUND_HALF_UP

import yaml

logger = logging.getLogger(__name__)


def load_config() -> dict:
    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# 教师模型 Prompt 模板（生成带  与 <answer> 的金融解答）
TEACHER_PROMPT_TEMPLATE = """你是一位资深金融分析师。请针对以下金融问题，给出完整的逐步推理过程，并将思考过程放在  标签内，最终答案放在 <answer> 标签内。

问题：{question}
上下文：{context}

要求：
1.  内必须包含至少 2 个清晰的推理步骤。
2. 所有数值计算必须展示中间过程。
3. <answer> 内只包含最终结论或数值，保留 2~4 位小数。
4. 如果信息不足，在 <answer> 中明确说明"根据现有资料无法确定"。

请直接输出  与 <answer> 标签内容。"""


class SFTDataGenerator:
    """
    SFT 数据生成流水线
    支持教师模型 API 调用与本地模型回退
    """

    def __init__(self, teacher_backend: str = "api"):
        self.cfg = load_config()
        self.teacher_backend = teacher_backend
        self.min_steps = 2
        self.numeric_tolerance = 0.01
        self._api_client = None

    def _get_api_client(self):
        """延迟初始化 OpenAI 兼容客户端（DeepSeek-R1 API）"""
        if self._api_client is not None:
            return self._api_client
        try:
            from openai import OpenAI
            api_key = os.getenv("DEEPSEEK_API_KEY", "")
            base_url = os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com")
            if not api_key:
                logger.warning("DEEPSEEK_API_KEY not set, teacher model API unavailable")
                return None
            self._api_client = OpenAI(api_key=api_key, base_url=base_url)
            logger.info("Teacher API client initialized")
            return self._api_client
        except Exception as e:
            logger.warning(f"Failed to init teacher API client: {e}")
            return None

    def _call_teacher_model(self, question: str, context: str) -> Optional[str]:
        """
        调用教师模型（DeepSeek-R1 API 或本地 7B）生成带推理链的解答
        返回原始文本（应包含  与 <answer>）
        """
        prompt = TEACHER_PROMPT_TEMPLATE.format(question=question, context=context)
        client = self._get_api_client()

        if client is not None:
            try:
                resp = client.chat.completions.create(
                    model="deepseek-reasoner",  # DeepSeek-R1 满血版
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.6,
                    max_tokens=2048,
                )
                raw = resp.choices[0].message.content or ""
                logger.info("Teacher API generated reasoning chain")
                return raw
            except Exception as e:
                logger.warning(f"Teacher API call failed: {e}, falling back to local")

        # 回退：本地 DeepSeek-R1-Distill-Qwen-7B（若存在）
        local_7b = "./models/base/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
        if os.path.exists(local_7b):
            try:
                return self._call_local_teacher(local_7b, prompt)
            except Exception as e:
                logger.warning(f"Local 7B teacher failed: {e}")

        logger.error("No teacher model available; cannot generate training data")
        return None

    def _call_local_teacher(self, model_path: str, prompt: str) -> str:
        """本地模型回退（简化版，实际使用需加载 tokenizer + model）"""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=1024,
                temperature=0.6,
                top_p=0.9,
                do_sample=True,
            )
        return tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    def _validate_sample(self, raw_text: str, question: str) -> bool:
        """
        校验生成的样本是否满足格式要求
        """
        from reasoning import ChainParser

        parser = ChainParser(min_steps=self.min_steps)
        parsed = parser.parse(raw_text)

        if parsed.fallback_triggered:
            logger.warning(f"Sample fallback triggered: {parsed.fallback_reason}")
            return False

        if len(parsed.reasoning_steps) < self.min_steps:
            logger.warning(f"Too few steps: {len(parsed.reasoning_steps)}")
            return False

        # 数值一致性校验
        from reasoning import ChainValidator
        validator = ChainValidator(numeric_tolerance=self.numeric_tolerance)
        ok, reason = validator.validate(parsed.reasoning_steps, parsed.final_answer)
        if not ok:
            logger.warning(f"Numeric inconsistency: {reason}")
            return False

        return True

    def generate_dataset(
        self,
        questions: List[Dict[str, str]],
        output_path: str,
        max_retries: int = 3,
    ) -> int:
        """
        批量生成 SFT 数据集

        Args:
            questions: 每个元素为 {"question": str, "context": str, "answer": str}
            output_path: 输出 .jsonl 文件路径
            max_retries: 每个样本最大重试次数

        Returns:
            成功生成的样本数
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        valid_count = 0

        with open(output_path, "w", encoding="utf-8") as f:
            for item in questions:
                question = item["question"]
                context = item.get("context", "")

                for attempt in range(max_retries):
                    raw = self._call_teacher_model(question, context)
                    if raw is None:
                        break

                    if self._validate_sample(raw, question):
                        # 构造 messages 格式（兼容 Qwen chat template）
                        sample = {
                            "messages": [
                                {"role": "system", "content": "你是一位资深金融分析师。请给出逐步推理过程，将思考放在  标签内，最终答案放在 <answer> 标签内。"},
                                {"role": "user", "content": question},
                                {"role": "assistant", "content": raw},
                            ]
                        }
                        f.write(json.dumps(sample, ensure_ascii=False) + "\n")
                        valid_count += 1
                        logger.info(f"Generated valid sample {valid_count}: {question[:40]}...")
                        break
                    else:
                        logger.warning(f"Attempt {attempt + 1}/{max_retries} failed for: {question[:40]}...")

        logger.info(f"Dataset generation complete: {valid_count}/{len(questions)} valid samples -> {output_path}")
        return valid_count


# =============================================================================
# 内置金融问题模板（用于快速生成训练数据）
# =============================================================================

DEMO_QUESTIONS = [
    {
        "question": "某公司股价50元，每股收益2.5元，求市盈率？",
        "context": "市盈率（P/E Ratio）= 股价 / 每股收益（EPS）",
        "answer": "20.0",
    },
    {
        "question": "投资本金10000元，年利率5%，按年复利投资3年，本利和是多少？",
        "context": "复利公式：A = P * (1 + r)^t",
        "answer": "11576.25",
    },
    {
        "question": "某公司净利润5000万元，平均股东权益2.5亿元，求ROE？",
        "context": "ROE = 净利润 / 平均股东权益",
        "answer": "0.20",
    },
    {
        "question": "解释净资产收益率（ROE）及其在财务分析中的意义。",
        "context": "ROE 衡量公司利用股东权益创造利润的效率。杜邦分析将其拆解为净利率 × 资产周转率 × 权益乘数。",
        "answer": "ROE 是衡量股东权益回报效率的核心指标，杜邦分析可进一步拆解其驱动因素。",
    },
    {
        "question": "DCF 折现现金流模型中，若第1年自由现金流1000万，折现率10%，永续增长率3%，求企业价值？",
        "context": "两阶段 DCF：预测期现值 + 终值现值。终值 = FCF_n * (1+g) / (r-g)",
        "answer": "约14285.71万",
    },
]


def main():
    """CLI 入口：生成示例训练数据"""
    generator = SFTDataGenerator(teacher_backend="api")
    output = "./data/sft/train_with_think.jsonl"
    count = generator.generate_dataset(DEMO_QUESTIONS, output)
    print(f"Generated {count} training samples at {output}")


if __name__ == "__main__":
    main()
