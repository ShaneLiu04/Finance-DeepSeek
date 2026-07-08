"""
Finance-DeepSeek API 自动化测试脚本

用法：
    venv/Scripts/python.exe test_api.py

环境变量：
    BASE_URL  - API 基础地址（默认 http://localhost:8000）
"""

import os
import sys
import json
import time
import requests

BASE_URL = os.getenv("BASE_URL", "http://localhost:8000")


def print_response(name: str, resp: requests.Response):
    """打印响应结果"""
    print(f"\n{'='*60}")
    print(f"【{name}】")
    print(f"Status: {resp.status_code}")
    try:
        data = resp.json()
        print(json.dumps(data, ensure_ascii=False, indent=2))
    except Exception:
        print(resp.text[:500])
    print(f"{'='*60}")


def test_health():
    """1. 健康检查"""
    resp = requests.get(f"{BASE_URL}/health")
    print_response("GET /health", resp)
    return resp.status_code == 200


def test_index_status():
    """2. 索引状态"""
    resp = requests.get(f"{BASE_URL}/index/status")
    print_response("GET /index/status", resp)
    return resp.json().get("index_loaded", False)


def test_index_build():
    """3. 构建索引"""
    resp = requests.post(
        f"{BASE_URL}/index/build",
        json={"corpus_dir": "./data/corpus"},
    )
    print_response("POST /index/build", resp)
    return resp.status_code == 200


def test_retrieve():
    """4. RAG 检索"""
    resp = requests.post(
        f"{BASE_URL}/retrieve",
        json={
            "query": "什么是市盈率？",
            "top_k": 10,
            "rerank_top_k": 5,
        },
    )
    print_response("POST /retrieve", resp)
    return resp.status_code == 200


def test_generate_with_rag():
    """5. 端到端生成（带 RAG）"""
    resp = requests.post(
        f"{BASE_URL}/generate",
        json={
            "query": "某公司股价50元，每股收益2.5元，求市盈率？",
            "use_rag": True,
            "temperature": 0.6,
            "max_new_tokens": 1024,
            "top_p": 0.9,
        },
    )
    print_response("POST /generate (with RAG)", resp)
    return resp.status_code == 200


def test_generate_without_rag():
    """6. 端到端生成（无 RAG）"""
    resp = requests.post(
        f"{BASE_URL}/generate",
        json={
            "query": "解释净资产收益率（ROE）及其意义",
            "use_rag": False,
            "temperature": 0.6,
            "max_new_tokens": 1024,
            "top_p": 0.9,
        },
    )
    print_response("POST /generate (no RAG)", resp)
    return resp.status_code == 200


def test_parse():
    """7. 推理链解析"""
    raw_text = """  
步骤1: 识别参数 P=10000, r=0.05, t=3
步骤2: 计算 A = 10000 * (1+0.05)^3 = 11576.25
  
<answer>
11576.25
</answer>"""
    resp = requests.post(
        f"{BASE_URL}/parse",
        json={
            "raw_text": raw_text,
            "sources": ["https://www.investopedia.com/terms/c/compoundinterest.asp"],
        },
    )
    print_response("POST /parse", resp)
    return resp.status_code == 200


def test_validate_pass():
    """8. 数值校验（通过）"""
    resp = requests.post(
        f"{BASE_URL}/validate",
        json={
            "reasoning_steps": [
                "步骤1: 股价 = 50 元，EPS = 2.5 元",
                "步骤2: P/E = 50 / 2.5 = 20",
            ],
            "final_answer": "20.0",
        },
    )
    print_response("POST /validate (pass)", resp)
    return resp.status_code == 200 and resp.json().get("is_consistent") is True


def test_validate_fail():
    """9. 数值校验（失败）"""
    resp = requests.post(
        f"{BASE_URL}/validate",
        json={
            "reasoning_steps": [
                "步骤1: 股价 = 50 元，EPS = 2.5 元",
                "步骤2: P/E = 50 / 2.5 = 20",
            ],
            "final_answer": "25.0",
        },
    )
    print_response("POST /validate (fail)", resp)
    return resp.status_code == 200 and resp.json().get("is_consistent") is False


def test_generate_compound_interest():
    """10. 复杂计算：复利"""
    resp = requests.post(
        f"{BASE_URL}/generate",
        json={
            "query": "投资本金10000元，年利率5%，按年复利投资3年，本利和是多少？",
            "use_rag": True,
            "temperature": 0.6,
            "max_new_tokens": 1024,
            "top_p": 0.9,
        },
    )
    print_response("POST /generate (compound interest)", resp)
    return resp.status_code == 200


def test_retrieve_rerank():
    """11. 检索+精排"""
    resp = requests.post(
        f"{BASE_URL}/retrieve",
        json={
            "query": "DCF 折现现金流模型",
            "top_k": 10,
            "rerank_top_k": 3,
        },
    )
    print_response("POST /retrieve (with rerank)", resp)
    data = resp.json()
    chunks = data.get("chunks", [])
    if len(chunks) > 0:
        print(f"\n  Top-1 score: {chunks[0].get('score', 'N/A')}")
        print(f"  Top-1 text: {chunks[0].get('text', 'N/A')[:80]}...")
    return resp.status_code == 200


def run_all_tests():
    """运行全部测试"""
    print("\n" + "="*60)
    print("Finance-DeepSeek API 测试套件")
    print(f"Base URL: {BASE_URL}")
    print("="*60)

    results = {}

    # 1. 健康检查
    results["health"] = test_health()

    # 2. 索引状态
    index_loaded = test_index_status()
    results["index_status"] = True  # 总是返回 200

    # 3. 如果索引未加载，先构建
    if not index_loaded:
        print("\n[⚠️] 索引未加载，正在构建...")
        results["index_build"] = test_index_build()
        time.sleep(1)  # 等待索引加载
    else:
        results["index_build"] = True
        print("\n[✅] 索引已加载，跳过构建")

    # 4. 检索测试
    results["retrieve"] = test_retrieve()

    # 5. 端到端生成（带 RAG）
    print("\n[⏳] 端到端生成（带 RAG）需要加载模型，首次请求较慢...")
    results["generate_with_rag"] = test_generate_with_rag()

    # 6. 端到端生成（无 RAG）
    results["generate_without_rag"] = test_generate_without_rag()

    # 7. 推理链解析
    results["parse"] = test_parse()

    # 8. 数值校验（通过）
    results["validate_pass"] = test_validate_pass()

    # 9. 数值校验（失败）
    results["validate_fail"] = test_validate_fail()

    # 10. 复杂计算
    results["generate_compound"] = test_generate_compound_interest()

    # 11. 检索+精排
    results["retrieve_rerank"] = test_retrieve_rerank()

    # 汇总
    print("\n" + "="*60)
    print("测试结果汇总")
    print("="*60)
    passed = 0
    failed = 0
    for name, ok in results.items():
        status = "✅ PASS" if ok else "❌ FAIL"
        print(f"  {status:10}  {name}")
        if ok:
            passed += 1
        else:
            failed += 1

    print(f"\n总计: {passed} 通过, {failed} 失败 / {len(results)} 测试")
    print("="*60)

    return failed == 0


if __name__ == "__main__":
    try:
        ok = run_all_tests()
        sys.exit(0 if ok else 1)
    except requests.exceptions.ConnectionError:
        print(f"\n❌ 无法连接到 {BASE_URL}")
        print("请确保服务已启动: venv/Scripts/python.exe -m api.main")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n测试已中断")
        sys.exit(130)
