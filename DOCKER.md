# Finance-DeepSeek Docker Deployment Guide

## Quick Start

### 1. Build the Docker image
```bash
docker compose build
```

### 2. Start the API service
```bash
docker compose up api
```
The FastAPI service will be available at `http://localhost:8000`.

### 3. Run one-off jobs

**Generate SFT training data:**
```bash
docker compose run --rm data-gen
```

**Build FAISS index from corpus:**
```bash
docker compose run --rm indexer
```

**Run QLoRA fine-tuning:**
```bash
docker compose run --rm trainer
```

## Prerequisites

- Docker Engine >= 24.0 with BuildKit enabled
- Docker Compose >= 2.20
- NVIDIA Container Toolkit (for GPU support)
- NVIDIA driver >= 525 (CUDA 12.1 compatible)

## Environment Variables

Create a `.env` file in the project root:

```env
# DeepSeek API (optional, for teacher model data generation)
DEEPSEEK_API_KEY=your_api_key_here
DEEPSEEK_API_BASE=https://api.deepseek.com

# HuggingFace (optional, for downloading models)
HF_TOKEN=your_hf_token_here
```

## Volume Mounts

| Host Path | Container Path | Purpose |
|-----------|----------------|---------|
| `./models` | `/app/models` | Model weights (read-only) |
| `./data` | `/app/data` | Training data & corpus |
| `./outputs` | `/app/outputs` | Fine-tuned adapters & indices |
| `./logs` | `/app/logs` | Application logs |
| `./config.yaml` | `/app/config.yaml` | Project configuration |

## Architecture

- **Multi-stage build**: Separates build dependencies from runtime to minimize image size
- **Non-root user**: `appuser` runs the application for security
- **GPU runtime**: NVIDIA Container Toolkit provides CUDA 12.1 inside containers
- **Health check**: API service includes HTTP health probe on `/health`

## Image Size Optimization

- Model weights (`.safetensors`, `.bin`) are excluded from the build context via `.dockerignore`
- Heavy compilation dependencies (`build-essential`, `cmake`) are isolated in the `builder` stage
- Only compiled Python packages are copied to the final `production` stage

## Troubleshooting

**GPU not detected inside container:**
```bash
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```
If this fails, reinstall the NVIDIA Container Toolkit.

**OOM during training:**
Reduce `per_device_train_batch_size` in `config.yaml` or limit GPU memory fraction:
```env
CUDA_VISIBLE_DEVICES=0
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
```

**Model path not found:**
Ensure the local model directory is mounted:
```bash
ls models/base/deepseek-ai/DeepSeek-R1-Distill-Qwen-1___5B/
```
