# Finance-DeepSeek Docker 镜像
# 基于 NVIDIA CUDA 运行时，适配 RTX 4060 等消费级显卡

FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

# 安装 Python 与基础依赖
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 复制依赖并安装
COPY requirements.txt .
RUN pip3 install --upgrade pip setuptools wheel && \
    pip3 install -r requirements.txt

# 复制项目代码
COPY . .

# 暴露 API 端口
EXPOSE 8000

# 默认启动命令（可根据需要覆盖为训练或评测）
CMD ["python3", "-m", "uvicorn", "finance_deepseek.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
