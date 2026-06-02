FROM python:3.10-slim

# Install system utilities needed for compile-dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables for hugging face python caching
ENV PYTHONUNBUFFERED=1 \
    TRANSFORMERS_CACHE=/tmp/huggingface \
    HF_HOME=/tmp/huggingface

WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download the required NLP model
RUN python -m spacy download en_core_web_sm

# Copy the rest of the application files
COPY . .

# Pre-create writable folders and set permissions for Hugging Face user (UID 1000)
RUN mkdir -p /app/logs /app/data/chroma_db && \
    chmod -R 777 /app/logs /app/data

# Hugging Face Spaces default port is 7860
EXPOSE 7860

# Start Uvicorn server
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
