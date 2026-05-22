"""
带 <think> 标签的 SFT 数据生成器
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


# 教师模型 Prompt 模板（生成带 <think> 与 <answer> 的金融解答）
TEACHER_PROMPT_TEMPLATE = """你是一位资深金融分析师。请针对以下金融问题，给出完整的逐步推理过程，并将思考过程放在 <think> 标签内，最终答案放在 <answer> 标签内。

问题：{question}
上下文：{context}

要求：
1. <think> 内必须包含至少 2 个清晰的推理步骤。
2. 所有数值计算必须展示中间过程。
3. <answer> 内只包含最终结论或数值，保留 2~4 位小数。
4. 如果信息不足，在 <answer> 中明确说明"根据现有资料无法确定"。

请直接输出 <think> 与 <answer> 标签内容。"""


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
        返回原始文本（应包含 <think> 与 <answer>）
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

        return None

    def _call_local_teacher(self, model_path: str, prompt: str) -> str:
        """本地 7B 蒸馏版作为教师模型"""
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
        text = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        # 清理显存
        del model
        torch.cuda.empty_cache()
        return text

    def generate_from_finqa(
        self,
        finqa_path: str,
        output_path: str,
        max_samples: int = 2000,
    ) -> str:
        """
        从 FinQA 格式数据集生成带推理链的 SFT 数据
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        samples = []

        if not os.path.exists(finqa_path):
            logger.warning(f"FinQA data not found at {finqa_path}, using demo samples")
            raw_samples = self._demo_finqa_samples()
        else:
            raw_samples = self._load_finqa(finqa_path)

        for i, item in enumerate(raw_samples[:max_samples]):
            try:
                formatted = self._format_sample(item)
                if formatted and self._validate_sample(formatted):
                    samples.append(formatted)
            except Exception as e:
                logger.warning(f"Sample {i} generation failed: {e}")

        with open(output_path, "w", encoding="utf-8") as f:
            for s in samples:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")

        logger.info(f"Generated {len(samples)} SFT samples -> {output_path}")
        return output_path

    def _load_finqa(self, path: str) -> List[Dict]:
        """加载 FinQA 格式数据"""
        samples = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                samples.append(obj)
        return samples

    def _format_sample(self, item: Dict) -> Optional[Dict]:
        """
        将原始 FinQA 样本转换为 Alpaca 格式，并注入 <think> 与 <answer>
        优先使用教师模型生成的推理链，其次使用已有 gold_reasoning，最后回退到占位符
        """
        question = item.get("question", "")
        answer = item.get("answer", "")
        context = item.get("context", item.get("pre_text", "") + " " + item.get("post_text", ""))
        gold_reasoning = item.get("reasoning", "")

        # 优先级 1: 已有高质量推理链
        if gold_reasoning and self._looks_like_reasoning(gold_reasoning):
            think = self._sanitize_think(gold_reasoning)
        else:
            # 优先级 2: 调用教师模型（DeepSeek-R1 API / 本地 7B）
            teacher_output = self._call_teacher_model(question, context)
            if teacher_output:
                parsed_think, parsed_answer = self._extract_think_answer(teacher_output)
                if parsed_think and parsed_answer:
                    think = parsed_think
                    answer = parsed_answer
                else:
                    think = self._sanitize_think(teacher_output)
            else:
                # 优先级 3: 占位符规则生成
                think = self._generate_think_placeholder(question, context, answer)

        final_answer = self._sanitize_answer(answer)

        system_msg = (
            "你是一位资深金融分析师。思考过程请放在 <think> 标签内，"
            "最终答案放在 <answer> 标签内。"
        )
        user_msg = f"问题：{question}\n上下文：{context}".strip()
        assistant_msg = f"<think>\n{think}\n</think>\n<answer>\n{final_answer}\n</answer>"

        return {
            "messages": [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": assistant_msg},
            ],
            "source": item.get("source", "finqa"),
        }

    def _extract_think_answer(self, text: str) -> tuple:
        """从教师模型输出中提取 <think> 和 <answer> 内容"""
        think_match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
        answer_match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
        think = think_match.group(1).strip() if think_match else ""
        answer = answer_match.group(1).strip() if answer_match else ""
        return think, answer

    def _looks_like_reasoning(self, text: str) -> bool:
        """判断文本是否包含推理步骤"""
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        return len(lines) >= self.min_steps

    def _sanitize_think(self, text: str) -> str:
        """清洗推理链，去除已有标签"""
        text = text.replace("<think>", "").replace("</think>", "")
        text = text.replace("<answer>", "").replace("</answer>", "")
        return text.strip()

    def _sanitize_answer(self, text: str) -> str:
        """清洗答案，保留数值精度，避免科学计数法"""
        text = str(text).strip()
        try:
            d = Decimal(text)
            # 保留最多4位小数，去除末尾的0，但避免 normalize() 产生科学计数法
            quantized = d.quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP)
            # 转为字符串后去除末尾0和小数点
            result = str(quantized)
            if "." in result:
                result = result.rstrip("0").rstrip(".")
            return result if result else str(quantized)
        except Exception:
            return text

    def _generate_think_placeholder(self, question: str, context: str, answer: str) -> str:
        """当无教师模型时，生成简单占位推理链"""
        numbers = re.findall(r"[-+]?\d+\.?\d*", question + " " + context)
        steps = ["步骤1: 识别问题中的关键财务指标与已知数值。"]
        if numbers:
            steps.append(f"步骤2: 提取关键数值：{', '.join(numbers[:4])}。")
        steps.append("步骤3: 应用相关金融公式进行计算与验证。")
        steps.append(f"步骤4: 得出最终结论 {answer}，并检查合理性。")
        return "\n".join(steps)

    def _validate_sample(self, sample: Dict) -> bool:
        """规则校验生成样本的质量，包含数值一致性检查"""
        assistant = sample["messages"][2]["content"]
        has_think = "<think>" in assistant and "</think>" in assistant
        has_answer = "<answer>" in assistant and "</answer>" in assistant

        if not (has_think and has_answer):
            return False

        think_match = re.search(r"<think>(.*?)</think>", assistant, re.DOTALL)
        if not think_match:
            return False

        think_text = think_match.group(1)
        lines = [l.strip() for l in think_text.splitlines() if l.strip()]
        if len(lines) < self.min_steps:
            return False

        # 数值一致性校验：提取 think 和 answer 中的数值，检查是否兼容
        answer_match = re.search(r"<answer>(.*?)</answer>", assistant, re.DOTALL)
        if answer_match:
            answer_text = answer_match.group(1).strip()
            if not self._numeric_consistency_check(think_text, answer_text):
                logger.warning("Numeric inconsistency detected between think and answer")
                return False

        return True

    def _numeric_consistency_check(self, think_text: str, answer_text: str) -> bool:
        """检查推理链中的数值与最终答案是否一致（1%容差）"""
        from finance_deepseek.reasoning.validator import ChainValidator
        validator = ChainValidator(numeric_tolerance=self.numeric_tolerance)
        # 提取 think 中的最后一行数字作为预期结果
        think_nums = re.findall(r"[-+]?\d+\.?\d*", think_text)
        answer_nums = re.findall(r"[-+]?\d+\.?\d*", answer_text)
        if not think_nums or not answer_nums:
            return True  # 无数值可比较，放行
        # 比较最后一个数值（通常是最接近答案的）
        return validator.safe_numeric_compare(think_nums[-1], answer_nums[0])

    def generate_alpaca_base(
        self,
        output_path: str,
        num_samples: int = 500,
    ) -> str:
        """生成通用金融术语解释 Alpaca 数据"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        templates = [
            ("什么是市盈率（P/E Ratio）？", "市盈率是股票价格与每股收益（EPS）的比率，用于衡量股票估值水平。计算公式：P/E = 股价 / EPS。"),
            ("解释净资产收益率（ROE）及其意义。", "ROE = 净利润 / 平均股东权益，反映股东权益的收益水平，是衡量公司盈利能力的核心指标。"),
            ("什么是复利？", "复利是指利息也计入本金产生新的利息。公式：A = P(1 + r/n)^(nt)。"),
            ("DCF 模型的核心思想是什么？", "将未来自由现金流按适当折现率折现到当前，加总得到企业内在价值。"),
            ("什么是 EPS？", "每股收益（EPS）=（净利润 - 优先股股利）/ 流通在外普通股加权平均股数。"),
        ]
        samples = []
        system_msg = "你是一位资深金融分析师。思考过程请放在 <think> 标签内，最终答案放在 <answer> 标签内。"
        for i in range(num_samples):
            q, a = templates[i % len(templates)]
            think = "步骤1: 理解问题涉及的核心概念。\n步骤2: 给出定义、公式及实际意义。"
            assistant = f"<think>\n{think}\n</think>\n<answer>\n{a}\n</answer>"
            samples.append({
                "messages": [
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": q},
                    {"role": "assistant", "content": assistant},
                ],
                "source": "alpaca_base",
            })

        with open(output_path, "w", encoding="utf-8") as f:
            for s in samples:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")

        logger.info(f"Generated {len(samples)} base Alpaca samples -> {output_path}")
        return output_path

    def _demo_finqa_samples(self) -> List[Dict]:
        """内置演示金融算术样本"""
        return [
            {
                "question": "某公司股价为 50 元，每股收益 EPS 为 2.5 元，求市盈率？",
                "answer": "20.0",
                "context": "市盈率 = 股价 / EPS",
                "reasoning": "步骤1: 识别已知条件：股价 = 50 元，EPS = 2.5 元。\n步骤2: 应用市盈率公式：P/E = 50 / 2.5 = 20。\n步骤3: 验证：20 倍属于合理估值区间，计算无误。",
            },
            {
                "question": "投资本金 10000 元，年利率 5%，按年复利投资 3 年，本利和是多少？",
                "answer": "11576.25",
                "context": "复利公式 A = P(1+r)^t",
                "reasoning": "步骤1: 识别参数：P=10000, r=0.05, t=3, n=1。\n步骤2: 代入公式：A = 10000 * (1+0.05)^3 = 10000 * 1.157625 = 11576.25。\n步骤3: 验证：逐年计算 10500, 11025, 11576.25，结果一致。",
            },
            {
                "question": "公司净利润 5000 万，股东权益 2 亿，求 ROE？",
                "answer": "25.0%",
                "context": "ROE = 净利润 / 股东权益",
                "reasoning": "步骤1: 提取数值：净利润 = 5000 万，股东权益 = 20000 万。\n步骤2: 计算 ROE = 5000 / 20000 = 0.25 = 25%。\n步骤3: 使用杜邦分析复核：ROE 处于健康水平。",
            },
        ]
