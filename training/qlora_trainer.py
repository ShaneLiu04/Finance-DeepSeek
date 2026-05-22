"""
QLoRA 微调入口
基于 DeepSeek-R1-Distill-Qwen-1.5B 的领域对齐微调
"""

import os
import sys
import logging
import argparse
from pathlib import Path

import yaml
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    set_seed,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
from datasets import load_dataset

# 将项目根目录加入路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def load_config() -> dict:
    config_path = PROJECT_ROOT / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def setup_quantization(cfg: dict) -> BitsAndBytesConfig:
    qcfg = cfg["quantization"]
    return BitsAndBytesConfig(
        load_in_4bit=qcfg["load_in_4bit"],
        bnb_4bit_quant_type=qcfg["bnb_4bit_quant_type"],
        bnb_4bit_use_double_quant=qcfg["bnb_4bit_use_double_quant"],
        bnb_4bit_compute_dtype=getattr(torch, qcfg["bnb_4bit_compute_dtype"]),
    )


def setup_lora(cfg: dict) -> LoraConfig:
    lcfg = cfg["lora"]
    return LoraConfig(
        r=lcfg["r"],
        lora_alpha=lcfg["lora_alpha"],
        target_modules=lcfg["target_modules"],
        lora_dropout=lcfg["lora_dropout"],
        bias=lcfg["bias"],
        task_type=lcfg["task_type"],
    )


def format_chat_messages(example):
    """
    将 messages 列表格式化为单条文本，兼容 Qwen chat template
    """
    messages = example["messages"]
    # 使用 Qwen 的 chat template（训练时直接拼接也可）
    # 此处做简单拼接，确保 <think> 与 <answer> 结构保留
    text_parts = []
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        if role == "system":
            text_parts.append(f"<|im_start|>system\n{content}<|im_end|>")
        elif role == "user":
            text_parts.append(f"<|im_start|>user\n{content}<|im_end|>")
        elif role == "assistant":
            text_parts.append(f"<|im_start|>assistant\n{content}<|im_end|>")
    return {"text": "\n".join(text_parts)}


def main():
    parser = argparse.ArgumentParser(description="Finance-DeepSeek QLoRA Trainer")
    parser.add_argument("--config", type=str, default=None, help="Override config path")
    args = parser.parse_args()

    cfg = load_config()
    set_seed(cfg["project"]["seed"])

    mcfg = cfg["model"]
    lcfg = cfg["lora"]
    tcfg = cfg["training"]
    qcfg = cfg["quantization"]

    local_path = mcfg.get("local_path")
    model_name = local_path if local_path else mcfg["base_model"]
    output_dir = tcfg["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    logger.info(f"Loading base model: {model_name}")

    # 量化配置
    bnb_config = setup_quantization(cfg)

    # 加载 Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # 加载模型（GPU 用 4-bit，CPU 回退到 float32）
    load_kwargs = {
        "trust_remote_code": True,
    }
    if torch.cuda.is_available():
        load_kwargs["quantization_config"] = bnb_config
        load_kwargs["device_map"] = "auto"
        load_kwargs["torch_dtype"] = getattr(torch, qcfg["bnb_4bit_compute_dtype"])
    else:
        logger.warning("CUDA not available, training in CPU float32 mode (very slow)")
        load_kwargs["torch_dtype"] = torch.float32
    
    model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    # 为 4-bit 训练做准备
    model = prepare_model_for_kbit_training(model)

    # LoRA
    peft_config = setup_lora(cfg)
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # 加载数据集
    train_path = tcfg["train_data_path"]
    eval_path = tcfg.get("eval_data_path")

    if not os.path.exists(train_path):
        logger.error(f"Training data not found: {train_path}")
        logger.error("Please run data_generator.py first to generate SFT data.")
        sys.exit(1)

    data_files = {"train": train_path}
    # 若 alpaca_base 数据存在，合并进去扩充训练集
    alpaca_path = train_path.replace("with_think", "alpaca_base")
    if os.path.exists(alpaca_path):
        data_files["train_alpaca"] = alpaca_path
    if eval_path and os.path.exists(eval_path):
        data_files["validation"] = eval_path

    dataset = load_dataset("json", data_files=data_files)
    # 合并所有 train* 分片
    train_datasets = [dataset[k] for k in dataset.keys() if k.startswith("train")]
    from datasets import concatenate_datasets
    dataset["train"] = concatenate_datasets(train_datasets)
    dataset = dataset.map(format_chat_messages, remove_columns=dataset["train"].column_names)

    # 训练参数
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=tcfg["per_device_train_batch_size"],
        gradient_accumulation_steps=tcfg["gradient_accumulation_steps"],
        num_train_epochs=tcfg["num_train_epochs"],
        learning_rate=tcfg["learning_rate"],
        warmup_ratio=tcfg["warmup_ratio"],
        lr_scheduler_type=tcfg["lr_scheduler_type"],
        logging_steps=tcfg["logging_steps"],
        save_strategy=tcfg["save_strategy"],
        gradient_checkpointing=tcfg["gradient_checkpointing"],
        optim=tcfg["optim"],
        group_by_length=tcfg["group_by_length"],
        report_to=tcfg.get("report_to", "none"),
        fp16=False,
        bf16=qcfg["bnb_4bit_compute_dtype"] == "bfloat16",
        max_grad_norm=0.3,
        warmup_ratio=tcfg.get("warmup_ratio", 0.03),
        save_total_limit=2,
        load_best_model_at_end=False,
        # 显存控制
        dataloader_num_workers=0,
        remove_unused_columns=False,
    )

    # SFT Trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("validation"),
        max_seq_length=mcfg["max_seq_length"],
        args=training_args,
        dataset_text_field="text",
    )

    logger.info("Starting QLoRA training...")
    trainer.train()

    # 保存 Adapter
    adapter_path = os.path.join(output_dir, "final_adapter")
    trainer.model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)
    logger.info(f"Adapter saved to {adapter_path}")

    # 显存统计
    if torch.cuda.is_available():
        max_mem = torch.cuda.max_memory_allocated() / 1024 ** 3
        logger.info(f"Peak GPU memory: {max_mem:.2f} GB")
        if max_mem > cfg["hardware"]["training_peak_memory_gb"]:
            logger.warning(f"Peak memory {max_mem:.2f}GB exceeded budget {cfg['hardware']['training_peak_memory_gb']}GB")


if __name__ == "__main__":
    main()
