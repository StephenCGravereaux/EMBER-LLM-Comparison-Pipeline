# EMBER LLM Pipeline - Docker Image
FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Install ember package
COPY ember/ ./ember/
RUN pip install --no-cache-dir ./ember/

# Copy application files
COPY *.py ./
COPY *.txt ./
COPY *.jsonl ./
COPY ember2018/ ./ember2018/
COPY 20251029_214241_LoRA_r96_optimal_ember_llama/ ./20251029_214241_LoRA_r96_optimal_ember_llama/
COPY 20251020_FullParam_100pct_ember_llama/ ./20251020_FullParam_100pct_ember_llama/

CMD ["python", "model_comparison_framework.py"]

