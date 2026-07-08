#!/bin/bash
# Finance-DeepSeek API 快速测试脚本 (curl)
# 用法: bash test_api.sh

set -e

BASE_URL="${BASE_URL:-http://localhost:8000}"

echo "=========================================="
echo "Finance-DeepSeek API 快速测试"
echo "Base URL: $BASE_URL"
echo "=========================================="

# 1. 健康检查
echo -e "\n[1/7] GET /health"
curl -s "$BASE_URL/health" | python -m json.tool 2>/dev/null || curl -s "$BASE_URL/health"

# 2. 构建索引
echo -e "\n\n[2/7] POST /index/build"
curl -s -X POST "$BASE_URL/index/build" \
  -H "Content-Type: application/json" \
  -d '{"corpus_dir": "./data/corpus"}' | python -m json.tool 2>/dev/null || true

# 3. 检索
echo -e "\n\n[3/7] POST /retrieve"
curl -s -X POST "$BASE_URL/retrieve" \
  -H "Content-Type: application/json" \
  -d '{"query": "什么是市盈率？", "top_k": 10, "rerank_top_k": 5}' | python -m json.tool 2>/dev/null || true

# 4. 端到端生成（带 RAG）
echo -e "\n\n[4/7] POST /generate (with RAG) - 首次请求较慢，请等待..."
curl -s -X POST "$BASE_URL/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "某公司股价50元，每股收益2.5元，求市盈率？",
    "use_rag": true,
    "temperature": 0.6,
    "max_new_tokens": 1024,
    "top_p": 0.9
  }' | python -m json.tool 2>/dev/null || true

# 5. 推理链解析
echo -e "\n\n[5/7] POST /parse"
curl -s -X POST "$BASE_URL/parse" \
  -H "Content-Type: application/json" \
  -d '{
    "raw_text": "\u003cthink\u003e\n步骤1: 识别参数 P=10000, r=0.05, t=3\n步骤2: 计算 A = 10000 * (1+0.05)^3 = 11576.25\n\u003c/think\u003e\n\u003canswer\u003e\n11576.25\n\u003c/answer\u003e"
  }' | python -m json.tool 2>/dev/null || true

# 6. 数值校验（通过）
echo -e "\n\n[6/7] POST /validate (pass)"
curl -s -X POST "$BASE_URL/validate" \
  -H "Content-Type: application/json" \
  -d '{
    "reasoning_steps": ["步骤1: 股价 = 50 元，EPS = 2.5 元", "步骤2: P/E = 50 / 2.5 = 20"],
    "final_answer": "20.0"
  }' | python -m json.tool 2>/dev/null || true

# 7. 数值校验（失败）
echo -e "\n\n[7/7] POST /validate (fail)"
curl -s -X POST "$BASE_URL/validate" \
  -H "Content-Type: application/json" \
  -d '{
    "reasoning_steps": ["步骤1: 股价 = 50 元，EPS = 2.5 元", "步骤2: P/E = 50 / 2.5 = 20"],
    "final_answer": "25.0"
  }' | python -m json.tool 2>/dev/null || true

echo -e "\n\n=========================================="
echo "测试完成"
echo "=========================================="
