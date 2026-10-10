# Container image deployed to Google Cloud Run by .github/workflows/deploy.yml.
# Also works on any container host (Render, a Docker Space, a VM): it listens on $PORT.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    GRADIO_ANALYTICS_ENABLED=False \
    MODEL_CACHE_DIR=/app/.model_cache \
    DATA_DIR=/app/data \
    PORT=7860

RUN useradd -m -u 1000 user
WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
# Bake the ONNX embedding + reranker models into the image so cold starts don't download them.
RUN python scripts/download_models.py && mkdir -p /app/data && chown -R user:user /app

USER user
EXPOSE 7860
CMD ["python", "app.py"]
