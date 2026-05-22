<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/PyTorch-2.4%2B-ee4c2c?logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/CUDA-12.1-green?logo=nvidia&logoColor=white" alt="CUDA 12.1">
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT">
  <img src="https://img.shields.io/badge/FastAPI-0.110%2B-009688?logo=fastapi&logoColor=white" alt="FastAPI">
</p>

<h1 align="center">Finance-DeepSeek</h1>

<p align="center">
  <b>基于 DeepSeek-R1 蒸馏的金融领域推理增强问答系统</b><br>
  <i>RAG + QLoRA + 结构化推理链解析 —— 单张 RTX 4060 (8GB) 可完成训练与部署</i>
</p>

<p align="center">
  <a href="#-快速开始">🚀 快速开始</a> •
  <a href="#-架构概览">🏗️ 架构</a> •
  <a href="#-api-文档">📡 API</a> •
  <a href="#-部署方式">🐳 部署</a> •
  <a href="#-评测结果">📊 评测</a>
</p>

---

## 📌 项目简介

Finance-DeepSeek 是一款面向**金融领域**的生产级问答系统，基于 `DeepSeek-R1-Distill-Qwen-1.5B` 构建。该模型由 DeepSeek-R1 671B 满血版经知识蒸馏得到，原生具备多步数值推理与自我验证能力。

系统通过三大核心模块实现金融场景下的高精度问答：

1. **QLoRA 领域对齐微调** —— 4-bit NF4 量化 + LoRA rank 128，在单卡 RTX 4060 上完成金融领域适配
2. **FAISS 稠密检索增强 (RAG)** —— `finance-embeddings-investopedia` 专业金融 Embedding + Top-8 上下文注入，抑制幻觉
3. **结构化推理链解析** —— 自动提取 `<think>` 推理过程与 `<answer>` 最终答案，支持 SSE 流式"边想边答"

> 💡 **核心设计哲学**：不再"教模型推理"，而是"规范模型已有的推理能力并注入金融领域知识"。

---

## ✨ 项目亮点

### 1. 轻量级推理蒸馏适配
- 基座模型：`DeepSeek-R1-Distill-Qwen-1.5B`（Qwen2.5-Math-1.5B 架构，经 671B 蒸馏）
- 量化方案：4-bit NF4 + double quant + bf16 混合精度计算
- **基座仅占约 1GB 显存**，为训练与推理留出充裕空间
- 训练峰值 ≤ 6.5GB，推理峰值 ≤ 4GB，完美适配消费级显卡

### 2. 金融专用 RAG 检索
- **Embedding**：`FinLang/finance-embeddings-investopedia`（金融领域专用，dim=768）
- **备选回退**：`yiyanghkust/finbert-tone`（FinBERT 金融情感模型）
- **索引引擎**：FAISS 自动选择 `IndexFlatIP`（<5万条）或 `IndexHNSWFlat`（≥5万条）
- **增量更新**：支持 `add()` 实时添加新文档，无需全量重建
- Prompt 严格约束："若材料不足以回答，请明确说明'根据现有资料无法确定'"

### 3. 原生多步数值推理可视化
- 模型输出自带 `<think>...</think>` 结构化推理链
- 系统自动解析为 `reasoning_steps: List[str]`
- SSE 流式输出：**先传输思考过程，再传输最终答案**
- 前端可实现"打字机效果"展示模型思维过程

### 4. 教师模型蒸馏流水线
- 支持调用 **DeepSeek-R1 API**（`deepseek-reasoner`）生成高质量推理链
- 自动回退到本地 **DeepSeek-R1-Distill-Qwen-7B**
- 训练数据内置数值一致性校验（1% 容差），过滤推理链与答案矛盾的样本

### 5. 三种推理模式动态切换

| 模式 | 说明 | 适用场景 |
|------|------|---------|
| `closed-book` | 纯模型自身知识 | 通用金融概念解释 |
| `rag` | RAG 检索增强 | 需要引用外部资料的 factual 问答 |
| `rag+reasoning` | RAG + 结构化推理链解析 | 数值计算、比率分析、需要可解释性的场景 |

### 6. NLI 级 Faithfulness 评测
- 使用 `mDeBERTa-v3-base-xnli-multilingual-nli-2mil7` 判断生成内容与上下文的**蕴含关系**
- 替代粗糙的"数字重叠"启发式，评测结果更具可信度
- 同时支持 Recall@K、MRR、EM、Numeric EM、BLEU-4、ROUGE-L、Chain Completeness

### 7. 生产级安全与稳定性
- ✅ 输入敏感词过滤（色情、赌博、洗钱、内幕交易等）
- ✅ temperature 硬封顶 0.7，防止金融场景幻觉
- ✅ Token 级精确截断（`max_input_length=1024`）
- ✅ 单并发 semaphore 保护，适配 RTX 4060 单卡

---

## 🏗️ 架构概览

```
┌─────────────────────────────────────────────────────────────────┐
│                         用户请求层                               │
│   ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│   │  Closed-Book │  │     RAG      │  │   RAG + Reasoning    │  │
│   └──────┬───────┘  └──────┬───────┘  └──────────┬───────────┘  │
└──────────┼────────────────┼─────────────────────┼──────────────┘
           │                │                     │
           ▼                ▼                     ▼
┌─────────────────────────────────────────────────────────────────┐
│                      FastAPI 服务层                              │
│  /v1/chat/completions  │  /v1/ingest  │  /v1/health            │
│  • OpenAI-compatible   │  • 增量索引   │  • GPU/模型/索引状态    │
│  • SSE 流式输出         │  • 安全过滤   │                        │
└─────────────────────────────────────────────────────────────────┘
           │                │                     │
           ▼                ▼                     ▼
┌─────────────────────────────────────────────────────────────────┐
│                      推理引擎 (InferenceEngine)                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐   │
│  │  Prompt 拼接  │  │  model.gen   │  │ ChainParser + Validator│ │
│  │  • Top-8 检索 │  │  • 4-bit QL  │  │  • <think> 提取     │    │
│  │  • 模板注入   │  │  • bf16 计算 │  │  • 数值一致性校验     │    │
│  └──────────────┘  └──────────────┘  └──────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────────────┐
│                      数据与模型层                                │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │ DeepSeek-R1  │  │  LoRA Adapter│  │  FAISS Index         │  │
│  │ Distill-1.5B │  │  (QLoRA r=128)│  │  • finance-embedding │  │
│  │ 4-bit 量化   │  │  金融领域对齐 │  │  • 768 dim           │  │
│  └──────────────┘  └──────────────┘  └──────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📦 未包含的文件与获取方式

由于 GitHub 单文件 100MB 限制及环境独立性原则，以下文件**未包含在仓库中**，但你只需运行对应脚本即可自动生成或下载：

| 缺失文件/目录 | 原因 | 大小 | 获取方式 |
|-------------|------|------|---------|
| `models/base/**/*.safetensors` | 基座模型权重过大 | ~3.5 GB | **首次运行时自动从 HuggingFace 下载**（`deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`） |
| `models/adapters/**/adapter_model.safetensors` | LoRA 适配器权重 | ~590 MB | 方式①：运行 `python run_training.py` 自行训练  <br>方式②：从 HuggingFace 下载预训练 adapter（如有发布） |
| `data/corpus/faiss_index.index` | FAISS 向量索引 | ~15 KB（随语料增长） | 运行 **`python run_data_gen.py`** 自动重建 |
| `data/corpus/chunk_metadata.json` | Chunk 元数据 | ~1 KB | 同上，随索引一起生成 |
| `data/finetune/*.jsonl` | SFT 训练数据 | ~500 KB | 同上，`run_data_gen.py` 自动生成 Alpaca + FinQA 数据 |
| `venv/` | Python 虚拟环境 | ~5 GB | 自行创建：`python -m venv venv` |
| `evaluation/results/` | 评测输出图表 | 运行时生成 | 运行 `python run_benchmark.py` 自动生成 |

### 一键准备所有缺失文件

```bash
# 1. 创建虚拟环境
python -m venv venv
# Linux/macOS: source venv/bin/activate
# Windows:     venv\Scripts\activate

# 2. 安装依赖
pip install -r requirements.txt

# 3. 一键生成数据 + 索引（自动下载 Embedding 模型和 tokenizer）
python run_data_gen.py

# 4. （可选）训练 LoRA Adapter
python run_training.py
```

完成以上步骤后，所有缺失文件均已就位，可直接启动 API 服务。

---

## 🚀 快速开始

### 环境要求

| 项目 | 要求 |
|---|---|
| GPU | NVIDIA RTX 4060 8GB 或同等级（支持 CUDA 12.x） |
| 显存峰值（训练） | ≤ 6.5 GB |
| 显存峰值（推理） | ≤ 4 GB |
| 系统内存 | ≥ 16 GB 推荐 |
| 磁盘空间 | ≥ 20 GB（含模型、索引与依赖） |

### 1. 克隆与安装

```bash
git clone https://github.com/ShaneLiu04/Finance-DeepSeek.git
cd Finance-DeepSeek/finance_deepseek

python -m venv venv
# Linux/macOS
source venv/bin/activate
# Windows
venv\Scripts\activate

pip install -r requirements.txt
```

### 2. 一键数据准备

```bash
# 生成 SFT 训练数据 + 构建 FAISS 索引
python run_data_gen.py
```

> 首次运行会自动：
> - 下载 `FinLang/finance-embeddings-investopedia`（约 500MB）
> - 生成 500 条 Alpaca 基础数据 + 3 条 FinQA Demo 数据
> - 构建 5 条演示语料的 FAISS 索引

**（可选）接入教师模型蒸馏**：
```bash
export DEEPSEEK_API_KEY="sk-your-api-key-here"  # Linux/macOS
set DEEPSEEK_API_KEY=sk-your-api-key-here         # Windows
python run_data_gen.py
```

### 3. 启动 API 服务

```bash
# Linux/macOS
PYTHONPATH="..:." python run_api.py

# Windows
set PYTHONPATH=..;.
python run_api.py
```

服务将在 `http://localhost:8000` 启动。

### 4. 验证部署

```bash
curl http://localhost:8000/v1/health
```

期望返回：
```json
{
  "status": "ok",
  "model_loaded": true,
  "adapter_loaded": true,
  "index_loaded": true,
  "gpu_memory_used_gb": 2.5,
  "gpu_memory_total_gb": 8.0,
  "index_doc_count": 5
}
```

---

## 📡 API 文档

### 核心端点

| 端点 | 方法 | 说明 |
|------|------|------|
| `/` | GET | 系统信息与端点列表 |
| `/v1/health` | GET | 健康检查（GPU 内存、模型/adapter/索引状态） |
| `/v1/chat/completions` | POST | OpenAI-compatible 对话接口 |
| `/v1/ingest` | POST | 实时文档摄入（分块、Embedding、增量索引） |
| `/docs` | GET | FastAPI 自动 Swagger UI |

### `POST /v1/chat/completions`

**Request Body**：

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `messages` | `List[ChatMessage]` | ✅ | 对话历史，`[{role:"user", content:"..."}]` |
| `mode` | `string` | ❌ | `"closed-book"` / `"rag"` / `"rag+reasoning"`（默认 `rag+reasoning`） |
| `stream` | `bool` | ❌ | 是否 SSE 流式返回（默认 `false`） |
| `temperature` | `float` | ❌ | 采样温度，范围 `[0.0, 0.7]`（默认 `0.6`） |
| `top_p` | `float` | ❌ | nucleus sampling（默认 `0.9`） |
| `max_tokens` | `int` | ❌ | 最大生成 token 数 `[1, 2048]`（默认 `1024`） |

**cURL 示例**：

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "某公司股票价格 80 元，EPS 为 4 元，市盈率是多少？"}],
    "mode": "rag+reasoning",
    "stream": false,
    "temperature": 0.6
  }'
```

**Response 示例**（`rag+reasoning` 模式）：

```json
{
  "id": "chatcmpl-a1b2c3d4",
  "object": "chat.completion",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "<think>\n步骤1: 识别已知条件：股价 = 80 元，EPS = 4 元。\n步骤2: 应用市盈率公式：P/E = 80 / 4 = 20。\n步骤3: 验证：20 倍属于合理估值区间。\n</think>\n<answer>\n20\n</answer>"
      },
      "finish_reason": "stop"
    }
  ]
}
```

### SSE 流式调用（推理链可视化）

```python
import requests

resp = requests.post(
    "http://localhost:8000/v1/chat/completions",
    json={
        "messages": [{"role": "user", "content": "投资 10000 元，年利率 5%，3 年复利本利和？"}],
        "mode": "rag+reasoning",
        "stream": True,
        "temperature": 0.6,
    },
    stream=True,
)

for line in resp.iter_lines():
    if line:
        print(line.decode("utf-8"))
```

输出格式：
```
data: 步骤1: 识别参数...
data: 步骤2: 代入公式...
...
event: parsed
data: {"reasoning_steps": [...], "final_answer": "11576.25", ...}
data: [DONE]
```

### `POST /v1/ingest`（文档摄入）

```bash
curl -X POST http://localhost:8000/v1/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "documents": [
      {
        "text": "央行降准 0.5 个百分点，释放长期资金约 1 万亿元...",
        "source_url": "https://example.com/news/1",
        "category": "货币政策",
        "title": "央行降准"
      }
    ]
  }'
```

返回：
```json
{"status": "success", "chunks_indexed": 2, "total_indexed": 7}
```

---

## 🐳 部署方式

### 方式一：本地部署（推荐开发调试）

```bash
cd finance_deepseek
python run_api.py
```

### 方式二：Docker 部署

```bash
docker-compose up --build
```

`docker-compose.yml` 已配置：
- NVIDIA CUDA 12.1 Runtime 基础镜像
- GPU 资源预留
- 健康检查自动重启
- 端口映射 `8000:8000`

### 方式三：QLoRA 微调训练

```bash
# 1. 准备数据
python run_data_gen.py

# 2. 启动训练
python run_training.py
```

训练产物：
- `models/adapters/finance_qlora/final_adapter/adapter_model.safetensors`
- `models/adapters/finance_qlora/final_adapter/adapter_config.json`

> 训练配置（`config.yaml`）：
> - LoRA: r=128, alpha=32, target_modules=[q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]
> - Batch: per_device=2, gradient_accumulation=2（有效 batch=4）
> - Epochs: 3, LR: 1e-4, Scheduler: cosine, Optimizer: paged_adamw_8bit

---

## 📊 评测结果

一键执行消融实验：

```bash
python run_benchmark.py
```

输出：
- 终端 Markdown 表格
- `evaluation/results/ablation_comparison.png`（柱状图可视化）
- `evaluation/results/ablation_results.json`

### 核心指标

| 指标 | 目标值 | 说明 |
|------|--------|------|
| **Faithfulness@8** | ≥ 72% | 事实性问答严格保真度（NLI 模型判定） |
| **Exact Match (EM)** | ≥ 15% | 字符串完全匹配率 |
| **Numeric EM** | ≥ 65% | 数值答案相对误差 1% 内匹配率 |
| **推理延迟** | < 3s | 非流式单请求，RTX 4060 |
| **训练显存峰值** | ≤ 6.5GB | 单卡 RTX 4060 安全余量 |
| **推理显存峰值** | ≤ 4GB | 单卡 RTX 4060 安全余量 |

---

## 📁 项目结构

```
Finance-DeepSeek/
├── finance_deepseek/
│   ├── api/
│   │   ├── main.py              # FastAPI 服务入口（lifespan、端点、安全过滤）
│   │   ├── schemas.py           # Pydantic 模型（OpenAI-compatible）
│   │   ├── inference.py         # 推理引擎（3 模式路由、SSE 流式、Token 截断）
│   │   └── dependencies.py      # 全局单例（AppState）
│   ├── rag/
│   │   ├── indexer.py           # FAISS 索引构建 + 增量 add()
│   │   ├── retriever.py         # DenseRetriever（Top-K + 可选重排）
│   │   └── embeddings.py        # EmbeddingProvider（金融模型 + 自动回退）
│   ├── reasoning/
│   │   ├── chain_parser.py      # <think>/<answer> 解析引擎
│   │   ├── validator.py         # 推理链-答案数值一致性校验
│   │   └── test_chain_parser.py # 单元测试
│   ├── training/
│   │   ├── qlora_trainer.py     # QLoRA SFT 训练入口
│   │   └── data_generator.py    # 教师模型蒸馏数据生成（DeepSeek-R1 API）
│   ├── evaluation/
│   │   ├── metrics.py           # EM / Numeric EM / Faithfulness@K / BLEU / ROUGE
│   │   └── benchmark.py         # 消融实验 + matplotlib 可视化
│   ├── data/
│   │   ├── corpus/              # 语料分块 + FAISS 索引 + chunk_metadata
│   │   └── finetune/            # SFT 训练数据（Alpaca + FinQA）
│   ├── models/
│   │   ├── base/                # DeepSeek-R1-Distill-Qwen-1.5B（自动下载）
│   │   └── adapters/            # LoRA Adapter 输出
│   ├── config.yaml              # 全局 YAML 配置
│   ├── requirements.txt         # Python 依赖
│   ├── run_api.py               # 一键启动 API
│   ├── run_training.py          # 一键启动训练
│   ├── run_data_gen.py          # 一键生成数据 + 索引
│   ├── run_benchmark.py         # 一键评测
│   ├── test_deploy.py           # 部署验证脚本
│   ├── Dockerfile               # CUDA 12.1 Runtime 镜像
│   └── docker-compose.yml       # Docker Compose 编排
└── README.md                    # 本文档
```

---

## 🛠️ 技术栈

| 层级 | 技术选型 |
|------|---------|
| **基座模型** | `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` |
| **微调框架** | `transformers` + `peft` + `trl` + `bitsandbytes` (QLoRA) |
| **训练精度** | 4-bit NF4 + bfloat16 |
| **向量检索** | `faiss-gpu` + `sentence-transformers` |
| **Embedding** | `FinLang/finance-embeddings-investopedia`（主选）/ `finbert-tone`（备选） |
| **NLI 评测** | `MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7` |
| **API 服务** | `FastAPI` + `Uvicorn` + `SSE` 流式输出 |
| **配置管理** | YAML 驱动 |
| **容器化** | `Docker` + `docker-compose` |

---

## 📌 路线图

- [x] QLoRA 领域微调框架
- [x] FAISS 稠密检索 + 增量更新
- [x] 结构化推理链解析 + SSE 流式
- [x] 教师模型蒸馏流水线（DeepSeek-R1 API）
- [x] NLI 级 Faithfulness 评测
- [x] 生产安全过滤 + Token 截断
- [ ] 接入真实 SEC 10-K / 央行政策语料库
- [ ] cross-encoder 二阶段精排（rerank）
- [ ] 多轮对话记忆与上下文压缩
- [ ] vLLM / TGI 推理加速
- [ ] Gradio / Streamlit 前端演示界面

---

## 🤝 贡献指南

欢迎提交 Issue 和 PR！

1. Fork 本仓库
2. 创建功能分支：`git checkout -b feature/amazing-feature`
3. 提交更改：`git commit -m 'Add amazing feature'`
4. 推送分支：`git push origin feature/amazing-feature`
5. 提交 Pull Request

请确保代码遵循 PEP8，关键函数包含 Docstring，并补充相应单元测试。

---

## 📄 许可证

本项目采用 [MIT License](LICENSE) 开源协议。

---

## 🙏 致谢

- [DeepSeek-AI / DeepSeek-R1-Distill-Qwen](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B) — 强大的推理蒸馏基座
- [Qwen2.5-Math](https://huggingface.co/Qwen) — 优秀的数学推理能力
- [Hugging Face PEFT / TRL](https://github.com/huggingface) — 高效微调生态
- [FAISS](https://github.com/facebookresearch/faiss) — 高性能向量检索
- [FinLang / finance-embeddings-investopedia](https://huggingface.co/FinLang) — 金融领域专用 Embedding

---

<p align="center">
  <i>如果本项目对您有帮助，请点亮 ⭐ Star，让更多人发现它！</i>
</p>
