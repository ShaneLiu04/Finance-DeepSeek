# Finance-DeepSeek API 使用文档

> 版本：0.1.0 | 基础 URL：`http://localhost:8000`

---

## 目录

1. [快速开始](#快速开始)
2. [端点概览](#端点概览)
3. [端点详解](#端点详解)
   - [GET /health](#get-health)
   - [POST /retrieve](#post-retrieve)
   - [POST /generate](#post-generate)
   - [POST /parse](#post-parse)
   - [POST /validate](#post-validate)
   - [GET /index/status](#get-indexstatus)
   - [POST /index/build](#post-indexbuild)
4. [测试脚本](#测试脚本)
5. [常见错误码](#常见错误码)

---

## 快速开始

### 1. 启动服务

```bash
# 使用 venv 中的 Python
venv/Scripts/python.exe -m api.main

# 或使用 uvicorn 直接启动
venv/Scripts/python.exe -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```

### 2. 查看自动生成的 API 文档

- **Swagger UI**：`http://localhost:8000/docs`
- **ReDoc**：`http://localhost:8000/redoc`

### 3. 健康检查

```bash
curl http://localhost:8000/health
```

预期响应：
```json
{
  "status": "ok",
  "version": "0.1.0",
  "model_loaded": false,
  "index_loaded": false
}
```

> **注意**：首次启动时 `model_loaded` 和 `index_loaded` 均为 `false`，因为模型和索引采用**懒加载**策略，仅在首次请求时初始化。首次请求 `/generate` 或 `/retrieve` 时会自动加载。

---

## 端点概览

| 方法 | 路径 | 功能 | 依赖 |
|:---|:---|:---|:---|
| `GET` | `/health` | 健康检查 | 无 |
| `POST` | `/retrieve` | RAG 稠密检索 | 需 FAISS 索引 |
| `POST` | `/generate` | 端到端推理生成 | 需模型 + 索引（可选） |
| `POST` | `/parse` | 推理链解析 | 无（轻量） |
| `POST` | `/validate` | 数值一致性校验 | 无（轻量） |
| `GET` | `/index/status` | 索引状态查询 | 无 |
| `POST` | `/index/build` | 从语料构建索引 | 需语料目录 |

---