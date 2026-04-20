FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y     tesseract-ocr     tesseract-ocr-fra     tesseract-ocr-eng     tesseract-ocr-ara     poppler-utils     libzbar0     curl     && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

COPY core /app
RUN pip install --no-cache-dir -e /app

EXPOSE 8765

CMD ["python", "local_api.py", "--host", "0.0.0.0", "--port", "8765"]
