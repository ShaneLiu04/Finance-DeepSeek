#!/usr/bin/env python3
"""
部署验证脚本：加载模型、做一次推理、测试 RAG 检索
"""

import sys
import os
import time
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root.parent))

os.environ["USE_TF"] = "0"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from finance_deepseek.rag.retriever import DenseRetriever
from finance_deepseek.reasoning.chain_parser import ChainParser


def test_model():
    print("=" * 60)
    print("Test 1: Loading base model (deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B)")
    print("=" * 60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    # 优先使用本地路径
    model_path = "./models/base/deepseek-ai/DeepSeek-R1-Distill-Qwen-1___5B"
    if not Path(model_path).exists():
        model_path = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # CPU mode: no quantization
    if device == "cpu":
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            dtype=torch.float32,
        )
    else:
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
    
    model.eval()
    print("Model loaded successfully!")
    
    # Quick inference test
    print("\nRunning quick inference test...")
    prompt = "<|im_start|>user\n什么是市盈率？<|im_end|>\n<|im_start|>assistant\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    start = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.6,
            top_p=0.9,
            do_sample=True,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
    elapsed = time.time() - start
    
    generated = outputs[0][inputs["input_ids"].shape[1]:]
    text = tokenizer.decode(generated, skip_special_tokens=True)
    print(f"\nGenerated in {elapsed:.1f}s:\n{text[:500]}...")
    
    return model, tokenizer


def test_rag():
    print("\n" + "=" * 60)
    print("Test 2: RAG Retrieval")
    print("=" * 60)
    
    retriever = DenseRetriever()
    query = "市盈率计算公式"
    results = retriever.retrieve(query, top_k=3)
    print(f"Query: {query}")
    print(f"Retrieved {len(results)} chunks:")
    for i, r in enumerate(results, 1):
        print(f"  [{i}] {r.get('title', 'N/A')} (score: {r.get('score', 0):.3f})")
    
    return retriever


def test_chain_parser():
    print("\n" + "=" * 60)
    print("Test 3: Reasoning Chain Parser")
    print("=" * 60)
    
    parser = ChainParser()
    sample = (
        "<think>\n"
        "Step 1: Identify stock price = 50 yuan, EPS = 2.5 yuan.\n"
        "Step 2: Apply P/E formula: 50 / 2.5 = 20.\n"
        "</think>\n"
        "<answer>20.0</answer>"
    )
    parsed = parser.parse(sample)
    print(f"Parsed steps: {len(parsed.reasoning_steps)}")
    print(f"Final answer: {parsed.final_answer}")
    print(f"Confidence: {parsed.confidence}")
    print("Chain parser OK!")


if __name__ == "__main__":
    print("Finance-DeepSeek Deployment Verification")
    print(f"Python: {sys.version}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    try:
        test_chain_parser()
        test_rag()
        test_model()
        print("\n" + "=" * 60)
        print("ALL TESTS PASSED!")
        print("=" * 60)
    except Exception as e:
        print(f"\nFAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
