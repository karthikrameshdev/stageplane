FROM python:3.13-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
COPY pyproject.toml ./
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    python -c "import tomllib; open('/tmp/requirements.txt', 'w').write('\n'.join(tomllib.load(open('pyproject.toml', 'rb'))['project']['dependencies']))" && \
    pip install --no-cache-dir -r /tmp/requirements.txt
COPY README.md LICENSE NOTICE ./
COPY stageplane ./stageplane
RUN pip install --no-cache-dir --no-deps .
EXPOSE 8080 8101 9101
ENTRYPOINT ["stageplane"]
CMD ["executor", "--host", "0.0.0.0"]
