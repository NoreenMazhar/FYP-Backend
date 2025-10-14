FROM python:3.12-slim

WORKDIR /app

# System deps for MySQL connector
RUN apt-get update && apt-get install -y --no-install-recommends \
    default-libmysqlclient-dev build-essential curl \
    && rm -rf /var/lib/apt/lists/*

COPY Backend/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY Backend /app

# Environment for Ollama (OpenAI-compatible)
ENV OLLAMA_BASE_URL=http://ollama:11434/v1 \
    OLLAMA_ANSWER_MODEL=phi3.5:mini-instruct-q4_K_M \
    SUMMARY_PREVIEW_ROWS=30 \
    SUMMARY_MAX_TOKENS=400

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]


