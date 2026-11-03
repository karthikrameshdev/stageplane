FROM python:3.13-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HOME=/tmp HF_HOME=/tmp/hf
COPY pyproject.toml ./
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    python -c "import tomllib; open('/tmp/requirements.txt', 'w').write('\n'.join(tomllib.load(open('pyproject.toml', 'rb'))['project']['dependencies']))" && \
    pip install --no-cache-dir -r /tmp/requirements.txt
COPY README.md LICENSE NOTICE ./
COPY stageplane ./stageplane
RUN pip install --no-cache-dir --no-deps . && \
    useradd --system --uid 10001 --home-dir /home/stageplane --create-home stageplane && \
    # Empty named volumes inherit these owners, so the unprivileged user can write caches and state.
    mkdir -p /cache /data /state /models && chown stageplane:stageplane /cache /data /state /models
USER 10001:10001
EXPOSE 8080 8101 9101
# Executor (8080) and worker (8101) both serve unauthenticated /livez.
HEALTHCHECK --interval=10s --timeout=3s --start-period=30s --retries=6 CMD ["stageplane", "probe"]
ENTRYPOINT ["stageplane"]
CMD ["executor", "--host", "0.0.0.0"]
