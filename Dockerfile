# Serving image. CPU base by default so the image builds anywhere; pass
# --build-arg BASE=nvidia/cuda:12.4.1-runtime-ubuntu22.04 for a GPU host.
ARG BASE=python:3.11-slim
FROM ${BASE}

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY scripts/ ./scripts/

ENV PYTHONPATH=/app/src \
    GBS_DEVICE=cpu \
    GBS_MAX_BATCH_SIZE=32 \
    GBS_MAX_QUEUE_DELAY_MS=5

EXPOSE 8000

# The model is downloaded on first start rather than baked in, so one image
# serves any encoder the GBS_MODEL_ID variable names.
HEALTHCHECK --interval=10s --timeout=5s --start-period=180s --retries=5 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/healthz',timeout=3).status==200 else 1)"

CMD ["python", "-m", "uvicorn", "gpu_batch_serve.api:get_app", "--factory", \
     "--host", "0.0.0.0", "--port", "8000"]
