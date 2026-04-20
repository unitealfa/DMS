FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_DEFAULT_TIMEOUT=1000
ENV LANG_PIPE_OFFLINE=1
ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1

WORKDIR /app

RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-fra \
    tesseract-ocr-eng \
    tesseract-ocr-ara \
    poppler-utils \
    libzbar0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt
RUN python -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch \
    && python - <<'PY'
from pathlib import Path
src = Path('/tmp/requirements.txt').read_text(encoding='utf-8').splitlines()
filtered = [line for line in src if line.strip() and line.strip() != 'torch']
Path('/tmp/requirements.docker.txt').write_text('\n'.join(filtered) + '\n', encoding='utf-8')
PY
RUN python -m pip install --no-cache-dir -r /tmp/requirements.docker.txt

COPY core /app
RUN python -m pip install --no-cache-dir -e /app

EXPOSE 8765

CMD ["python", "local_api.py", "--host", "0.0.0.0", "--port", "8765"]
