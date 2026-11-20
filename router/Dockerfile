FROM python:3.12-slim-bookworm
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 ROUTER_MODEL_CACHE=/models
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py examples.json ./
# Bake the model into the image, so the pod needs no internet access or writable disk.
RUN python -c "import app; app.TextEmbedding(app.MODEL, cache_dir='/models')" && chmod -R a+rX /models
ENV HF_HUB_OFFLINE=1
USER 65532:65532
EXPOSE 8080
ENTRYPOINT ["python", "app.py"]
