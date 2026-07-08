<div align="center">


<!-- Project Logo / Banner -->
<img src="https://img.shields.io/badge/🧠-Finance--DeepSeek-2ea44f?style=for-the-badge&logoColor=white" alt="Project Logo" />

<h3>面向金融垂直领域的可解释问答系统</h3>
<p>不仅给出答案，更展示完整推理链，并自动验证数值一致性</p>

<!-- Dynamic Shields.io Badges -->
[![Python](https://img.shields.io/badge/python-3.11%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4.1-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.136.1-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Docker](https://img.shields.io/badge/Docker-Supported-2496ED?logo=docker&logoColor=white)](DOCKER.md)

<!-- GitHub Dynamic Badges -->
[![GitHub Stars](https://img.shields.io/github/stars/ShaneLiu04/Finance-DeepSeek?style=social)](https://github.com/ShaneLiu04/Finance-DeepSeek/stargazers)
[![GitHub Forks](https://img.shields.io/github/forks/ShaneLiu04/Finance-DeepSeek?style=social)](https://github.com/ShaneLiu04/Finance-DeepSeek/network)
[![GitHub Issues](https://img.shields.io/github/issues/ShaneLiu04/Finance-DeepSeek)](https://github.com/ShaneLiu04/Finance-DeepSeek/issues)
[![GitHub Last Commit](https://img.shields.io/github/last-commit/ShaneLiu04/Finance-DeepSeek)](https://github.com/ShaneLiu04/Finance-DeepSeek/commits/main)

<!-- Build & Quality Badges -->
[![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Type Hints](https://img.shields.io/badge/type%20hints-mypy-blue.svg)](http://mypy-lang.org/)
[![Documentation](https://img.shields.io/badge/docs-API%20Usage%20%7C%20Docker-blue.svg)](API_USAGE.md)

</div>

---

## 目录

- [项目简介](#项目简介)
- [核心特性](#核心特性)
- [快速开始](#快速开始)
- [项目结构](#项目结构)
- [API 文档](#api-文档)
- [部署指南](#部署指南)
- [技术架构](#技术架构)
- [配置文件](#配置文件)
- [测试](#测试)
- [技术债务与路线图](#技术债务与路线图)
- [贡献指南](#贡献指南)
- [许可证](#许可证)

---

## 项目简介

Finance-DeepSeek 是一个基于 **DeepSeek-R1-Distill-Qwen-1.5B** 的金融垂直领域问答系统。系统通过 **RAG（检索增强生成）** 注入结构化金融知识，通过 **QLoRA 量化微调** 对齐金融推理风格，通过**结构化推理链解析**与**数值一致性校验**确保输出可追溯、可验证。

### 一句话定位

> 通用大模型在金融领域存在"幻觉"风险（数值计算错误、推理过程不可见），Finance-DeepSeek 通过强制输出结构化推理链（`think`/`answer` 标签）并自动校验数值一致性，让金融问答变得**可解释、可验证、可审计**。

### 适用场景

- 金融概念解释（市盈率、ROE、DCF 等）
- 财务指标计算与验证
- 金融公式推导与逐步分析
- 基于知识库的专业问答（非实时行情）

---

## 核心特性

| 特性                   | 说明                                                         |
| :--------------------- | :----------------------------------------------------------- |
| **结构化推理链**       | 强制模型输出 `think`...`think` 推理过程 + `<answer>`...`</answer>` 最终结论 |
| **数值一致性校验**     | 自动验证推理链中的数值与最终答案是否一致（1% 相对误差容忍）  |
| **三阶 Fallback 解析** | 标签缺失时自动降级解析，确保系统不崩溃                       |
| **RAG 知识增强**       | FAISS 向量检索 + Cross-Encoder 精排，注入结构化金融知识      |
| **消费级 GPU 可运行**  | 4-bit NF4 量化 + QLoRA 微调，峰值显存 ~5.5GB（RTX 3060 8GB 可训） |
| **FastAPI 服务层**     | 7 个 REST 端点，支持端到端生成、独立解析/校验、索引管理      |
| **Docker 容器化**      | 多阶段构建，支持 API / 训练 / 数据生成 / 索引构建 4 种服务形态 |

---

## 快速开始

### 环境要求

- Python 3.11+
- CUDA 11.8+（GPU 模式，可选）
- 8GB+ 显存（训练）或 4GB+ 显存（推理）
- 模型文件预先下载至 `models/base/deepseek-ai/DeepSeek-R1-Distill-Qwen-1___5B/`

### 1. 克隆与安装

```bash
# 创建虚拟环境
python -m venv venv

# Windows
venv\Scripts\python.exe -m pip install -r requirements.txt

# Linux/macOS
source venv/bin/activate
pip install -r requirements.txt
```

### 2. 准备模型

```bash
# 方式一：从 Hugging Face 下载（需要网络）
venv\Scripts\python.exe -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
model = 'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B'
AutoTokenizer.from_pretrained(model, trust_remote_code=True)
AutoModelForCausalLM.from_pretrained(model, trust_remote_code=True)
"

# 方式二：手动下载后放入 models/base/deepseek-ai/DeepSeek-R1-Distill-Qwen-1___5B/
# 需要文件：config.json, tokenizer.json, model.safetensors, 等
```

### 3. 启动 API 服务

```bash
# 启动 FastAPI 服务
venv\Scripts\python.exe -m api.main

# 或使用 uvicorn
venv\Scripts\python.exe -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```

服务启动后访问：

- **测试控制台**：`http://localhost:8000/`（自动重定向）
- **Swagger UI**：`http://localhost:8000/docs`
- **ReDoc**：`http://localhost:8000/redoc`

### 4. 首次使用流程

```bash
# 1. 健康检查
curl http://localhost:8000/health

# 2. 构建索引（从语料目录）
curl -X POST http://localhost:8000/index/build \
  -H "Content-Type: application/json" \
  -d '{"corpus_dir": "./data/corpus"}'

# 3. 测试检索
curl -X POST http://localhost:8000/retrieve \
  -H "Content-Type: application/json" \
  -d '{"query": "什么是市盈率？", "top_k": 10}'

# 4. 端到端生成（首次较慢，模型懒加载）
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "query": "某公司股价50元，每股收益2.5元，求市盈率？",
    "use_rag": true
  }'
```

### 5. 使用交互式测试页面

浏览器打开 `http://localhost:8000/`，点击按钮即可测试所有端点：

- 填写参数 → 点击发送 → 查看彩色高亮的 JSON 响应
- 支持一键填充示例、一键运行全部测试

---

## 项目结构

```
finance_deepseek/
├── api/                          # FastAPI 服务层
│   ├── __init__.py
│   ├── main.py                   # FastAPI 应用入口（7个端点）
│   ├── schemas.py                # Pydantic 请求/响应模型
│   └── service.py                # 业务逻辑单例（懒加载）
├── rag/                          # 检索增强生成（RAG）
│   ├── __init__.py
│   ├── embeddings.py             # Embedding 编码（双模型回退）
│   ├── indexer.py                # FAISS 索引构建与管理
│   ├── retriever.py              # 稠密检索 + Cross-Encoder 精排
│   └── reranker.py               # Cross-Encoder 重排序
├── reasoning/                    # 推理解析与校验
│   ├── __init__.py
│   ├── chain_parser.py           # 推理链解析（三阶 Fallback）
│   ├── validator.py              # 数值一致性校验（Decimal 精度）
│   └── test_chain_parser.py      # 单元测试
├── training/                     # 数据生成与模型微调
│   ├── __init__.py
│   ├── data_generator.py         # SFT 数据生成（教师模型蒸馏）
│   └── qlora_trainer.py          # QLoRA 4-bit 量化微调
├── models/                       # 模型文件目录
│   └── base/
│       └── deepseek-ai/
│           └── DeepSeek-R1-Distill-Qwen-1___5B/   # 基础模型
├── data/                         # 数据目录
│   ├── corpus/                   # 语料（JSONL）
│   ├── index/                    # FAISS 索引
│   └── sft/                      # SFT 训练数据
├── assets/                       # 项目资源
├── config.yaml                   # 集中配置文件 ⭐
├── requirements.txt              # 依赖清单 ⭐
├── Dockerfile                    # 多阶段构建 ⭐
├── docker-compose.yml            # 4 服务编排 ⭐
├── test_api.html                 # 交互式 Web 测试控制台 ⭐
├── test_api.py                   # Python 自动化测试脚本
├── test_api.http                 # VS Code REST Client 测试文件
├── API_USAGE.md                  # 详细 API 使用文档
├── DOCKER.md                     # Docker 部署指南
└── README.md                     # 本文件
```

---

## API 文档

### 端点速查

| 方法   | 路径            | 功能               | 是否需要模型 |
| :----- | :-------------- | :----------------- | :----------- |
| `GET`  | `/`             | 重定向到测试控制台 | 否           |
| `GET`  | `/health`       | 健康检查           | 否           |
| `GET`  | `/index/status` | 索引状态查询       | 否           |
| `POST` | `/index/build`  | 构建 FAISS 索引    | 否           |
| `POST` | `/retrieve`     | RAG 稠密检索       | 否（需索引） |
| `POST` | `/generate`     | 端到端推理生成     | 是           |
| `POST` | `/parse`        | 推理链解析         | 否           |
| `POST` | `/validate`     | 数值一致性校验     | 否           |

### 核心端点详解

#### POST /generate — 端到端推理生成

完整流程：**检索上下文 → 模型生成 → 推理链解析 → 数值校验**

**请求体**：

```json
{
  "query": "某公司股价50元，每股收益2.5元，求市盈率？",
  "use_rag": true,
  "temperature": 0.6,
  "max_new_tokens": 1024,
  "top_p": 0.9
}
```

**响应**：

```json
{
  "query": "某公司股价50元，每股收益2.5元，求市盈率？",
  "reasoning_steps": [
    "步骤1: 识别已知条件：股价 = 50 元，EPS = 2.5 元。",
    "步骤2: 应用市盈率公式：P/E = 50 / 2.5 = 20。",
    "步骤3: 验证：20 倍属于合理估值区间，计算无误。"
  ],
  "final_answer": "20.0",
  "confidence": 0.95,
  "fallback_triggered": false,
  "fallback_reason": "",
  "sources": ["https://www.investopedia.com/terms/p/price-earningsratio.asp"],
  "raw_output": "<think>...<think>\n<answer>20.0</answer>",
  "validated": true,
  "validation_message": "Consistent"
}
```

**质量信号字段**：

- `confidence` — 置信度 (0.0-1.0)，< 0.70 建议人工复核
- `fallback_triggered` — 是否触发降级解析（标签缺失）
- `validated` — 数值一致性校验是否通过

#### POST /retrieve — RAG 检索

```json
// 请求
{
  "query": "什么是市盈率？",
  "top_k": 10,
  "rerank_top_k": 5
}

// 响应
{
  "query": "什么是市盈率？",
  "chunks": [
    {
      "text": "市盈率（P/E Ratio）是股价与每股收益的比率...",
      "source_url": "https://www.investopedia.com/terms/p/price-earningsratio.asp",
      "category": "估值指标",
      "title": "市盈率 P/E Ratio",
      "score": 0.9234
    }
  ],
  "context": "[1] 市盈率（P/E Ratio）...\n（来源: https://...）"
}
```

#### POST /parse — 推理链解析

将模型原始输出解析为结构化步骤和答案，支持三阶 Fallback：

- 标签完整 → 正则提取
- 缺失 `think` → 全文作为答案
- 缺失 `<answer>` → 从 `think` 后提取最后一行

#### POST /validate — 数值校验

验证推理链中的数值与最终答案是否一致（1% 相对误差容忍，使用 `Decimal` 避免浮点精度问题）。

**更多 API 细节请参阅** [`API_USAGE.md`](API_USAGE.md)

---

## 部署指南

### 方式一：本地开发（推荐初学者）

```bash
# 1. 安装依赖
venv\Scripts\python.exe -m pip install -r requirements.txt

# 2. 启动服务
venv\Scripts\python.exe -m api.main

# 3. 浏览器打开 http://localhost:8000/
```

### 方式二：Docker 容器化（推荐生产环境）

```bash
# 构建镜像
docker compose build

# 启动 API 服务（常驻）
docker compose up api

# 运行训练任务（一次性）
docker compose --profile training run --rm trainer

# 运行数据生成（一次性）
docker compose --profile data-gen run --rm data-gen

# 构建索引（一次性）
docker compose --profile indexer run --rm indexer
```

**多阶段构建优势**：

- Builder 阶段编译 heavy deps（gcc/cmake），不进入最终镜像
- Production 阶段仅含运行时，体积最小化
- 模型权重通过 Volume 挂载，不打包进镜像
- 非 root 用户运行，安全加固

**更多 Docker 细节请参阅** [`DOCKER.md`](DOCKER.md)

### 方式三：Docker 快速测试（无需本地 Python）

```bash
# 仅启动 API 服务（自动挂载模型、数据、配置）
docker compose up api

# 访问测试页面
open http://localhost:8000/
```

---

## 技术架构

```
┌─────────────────────────────────────────────────────────────┐
│  数据层 (Data Layer)                                         │
│  FinQA 金融语料 → 教师模型蒸馏 → Alpaca 格式 SFT 训练数据       │
├─────────────────────────────────────────────────────────────┤
│  模型层 (Model Layer)                                        │
│  DeepSeek-R1-Distill-Qwen-1.5B + 4-bit NF4 量化 + QLoRA 微调  │
├─────────────────────────────────────────────────────────────┤
│  推理层 (Inference Layer)                                    │
│  用户查询 → Embedding → FAISS 检索 → 上下文组装 → 模型生成      │
│              ↓                                              │
│         Cross-Encoder 精排                                    │
│              ↓                                              │
│         ChainParser 解析 → ChainValidator 校验 → 结构化输出  │
└─────────────────────────────────────────────────────────────┘
```

### 三大核心模块

| 模块         | 文件                                                         | 职责                                               |
| :----------- | :----------------------------------------------------------- | :------------------------------------------------- |
| `rag/`       | `embeddings.py`, `indexer.py`, `retriever.py`, `reranker.py` | 向量编码、FAISS 索引、稠密检索、Cross-Encoder 精排 |
| `reasoning/` | `chain_parser.py`, `validator.py`                            | 推理链结构化解析、数值一致性校验                   |
| `training/`  | `data_generator.py`, `qlora_trainer.py`                      | SFT 数据生成（教师蒸馏）、QLoRA 量化微调           |

### 关键技术选型

| 选型     | 方案                          | 决策依据                                               |
| :------- | :---------------------------- | :----------------------------------------------------- |
| 基础模型 | DeepSeek-R1-Distill-Qwen-1.5B | MIT 许可证、中文原生、R1 推理链蒸馏、消费级 GPU 可运行 |
| 量化     | 4-bit NF4 + 双重量化          | 1.5B 模型显存从 3GB → 0.65GB                           |
| 微调     | QLoRA (r=16, α=32)            | 可训练参数 < 0.5%，单卡 8GB 可训                       |
| 检索     | FAISS + Cross-Encoder         | 零外部依赖、自动选型（FlatIP/HNSW）、精排提升准确率    |
| 服务     | FastAPI + Uvicorn             | 异步高性能、自动 OpenAPI 文档、Pydantic 类型校验       |

---

## 配置文件

所有配置集中管理于 `config.yaml`：

```yaml
# 模型配置
model:
  base_model: "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
  local_path: "./models/base/deepseek-ai/DeepSeek-R1-Distill-Qwen-1___5B"
  max_seq_length: 2048

# RAG 配置
rag:
  embedding_model: "BAAI/bge-large-zh-v1.5"
  fallback_embedding_model: "yiyanghkust/finbert-tone"
  top_k: 10
  rerank_top_k: 5
  use_reranker: true
  reranker:
    model_name: "cross-encoder/ms-marco-MiniLM-L-6-v2"
  index_path: "./data/index/faiss.index"
  corpus_dir: "./data/corpus"

# 量化配置
quantization:
  load_in_4bit: true
  bnb_4bit_quant_type: "nf4"
  bnb_4bit_use_double_quant: true
  bnb_4bit_compute_dtype: "bfloat16"

# LoRA 配置
lora:
  r: 16
  lora_alpha: 32
  target_modules: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
  lora_dropout: 0.05

# 训练配置
training:
  output_dir: "./models/adapters/finance-qlora"
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 4
  num_train_epochs: 3
  learning_rate: 2.0e-4

# API 配置
api:
  host: "0.0.0.0"
  port: 8000

# 推理配置
reasoning:
  min_steps: 2
  numeric_tolerance: 0.01
```

---

## 测试

### 交互式 Web 测试

浏览器访问 `http://localhost:8000/`，点击按钮测试所有端点。

### Python 自动化测试

```bash
venv\Scripts\python.exe test_api.py
```

运行 11 个测试用例：健康检查、索引构建、检索、生成（带/不带 RAG）、解析、校验（通过/失败）、复杂计算、精排。

### VS Code REST Client

安装 [REST Client](https://marketplace.visualstudio.com/items?itemName=humao.rest-client) 扩展，打开 `test_api.http`，点击 `Send Request`。

### Bash 快速测试

```bash
bash test_api.sh
```

---

## 技术债务与路线图

### 已解决的 P0 技术债务 ✅

| 债务项                  | 状态     | 解决方案                                  |
| :---------------------- | :------- | :---------------------------------------- |
| 缺少 `config.yaml`      | ✅ 已解决 | 创建集中配置文件                          |
| 缺少 `requirements.txt` | ✅ 已解决 | 锁定 16 个核心依赖版本                    |
| 无 API 服务层           | ✅ 已解决 | FastAPI + 7 个端点 + 交互式测试页面       |
| 无 Docker 容器化        | ✅ 已解决 | 多阶段 Dockerfile + 4 服务 docker-compose |
| 精排模块占位            | ✅ 已解决 | Cross-Encoder 真实实现                    |

### 已知限制（MVP 阶段）

- 无实时行情数据接入
- 无用户认证授权
- 无缓存层（Redis）
- 无混合检索（BM25 + 向量）
- 无多模型路由
- 无监控告警（Prometheus/Grafana）

### 路线图

| 阶段   | 目标                                      | 时间    |
| :----- | :---------------------------------------- | :------ |
| **Q1** | 服务化封装、CI/CD、测试覆盖、Redis 缓存   | 1-3 月  |
| **Q2** | 混合检索、PDF 解析、7B 模型升级、监控体系 | 4-6 月  |
| **H2** | 多租户、实时行情、MoE 架构探索            | 7-12 月 |

---

## 贡献指南

1. Fork 本仓库
2. 创建特性分支：`git checkout -b feature/xxx`
3. 提交变更：`git commit -m "feat: xxx"`
4. 推送分支：`git push origin feature/xxx`
5. 创建 Pull Request

**代码规范**：

- 使用 `ruff` 进行代码风格检查
- 使用 `black` 进行格式化
- 新增功能需补充单元测试

---

## 许可证

本项目采用 MIT 许可证。基础模型 DeepSeek-R1-Distill-Qwen-1.5B 采用 MIT 许可证，允许商用。

---

## 相关文档

| 文档                                                         | 说明                                     |
| :----------------------------------------------------------- | :--------------------------------------- |
| [`API_USAGE.md`](API_USAGE.md)                               | 详细 API 端点文档、请求/响应示例、错误码 |
| [`DOCKER.md`](DOCKER.md)                                     | Docker 多阶段构建、服务编排、部署指南    |
| [`Finance-DeepSeek-技术白皮书.docx`](Finance-DeepSeek-技术白皮书.docx) | 完整技术架构白皮书（9 章）               |
| `finance_deepseek_tech_report.agent.final.md`                | 技术报告 Markdown 版本                   |

---

<p align="center">
  <sub>Built with ❤️ using DeepSeek-R1, FastAPI, FAISS, and QLoRA</sub>
</p>

