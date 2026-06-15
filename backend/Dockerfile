# ─────────────────────────────────────────────────────
#  Maintenance Wizard — Backend (FastAPI)
#  Build context: repo ROOT
# ─────────────────────────────────────────────────────
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc g++ libffi-dev libssl-dev curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY backend/requirements.txt .

RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir \
        chromadb>=0.4.22 \
        sentence-transformers>=2.3.0 \
        scikit-learn>=1.3.0 \
        xgboost>=2.0.0 \
        numpy>=1.24.0 \
        pandas>=2.0.0 \
        joblib>=1.3.0 \
        pypdf>=4.0.0 \
        groq>=0.9.0 \
        openai>=1.10.0 \
        langchain>=0.1.0 \
        langchain-community>=0.0.20 \
        langchain-core>=0.1.0 \
        tiktoken>=0.5.0 \
        python-dateutil>=2.8.0 \
        aiofiles>=0.8.0

COPY backend/app/ ./app/
COPY backend/start.sh /start.sh
RUN chmod +x /start.sh

RUN mkdir -p /app/data /app/chroma_db /app/models

ENV PORT=8000
ENV DATABASE_URL=sqlite+aiosqlite:////app/data/maintenance_wizard.db
ENV CHROMA_DB_DIR=/app/chroma_db
ENV UPLOAD_DIR=/app/data
ENV LLM_PROVIDER=groq
ENV LLM_MODEL=llama-3.3-70b-versatile

EXPOSE 8000

CMD ["/start.sh"]
