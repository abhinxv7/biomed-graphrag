FROM mcr.microsoft.com/devcontainers/python:3.11

# Install system dependencies for audio and vision
RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install --no-install-recommends ffmpeg libsm6 libxext6  \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip

# 1. Pre-install CPU-only PyTorch to prevent massive 3GB CUDA downloads
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# 2. Install the GraphRAG and Local AI stack (memory optimized)
RUN pip install --no-cache-dir langchain langgraph neo4j qdrant-client openai tiktoken \
    beautifulsoup4 pandas numpy pydantic python-dotenv \
    langchain-huggingface sentence-transformers langchain-community \
    langchain-ollama