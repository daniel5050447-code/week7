# Cloud Run 배포용 이미지
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

# faiss-cpu 가 OpenMP 런타임(libgomp)을 필요로 함
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY main.py .

# Cloud Run 이 PORT 를 주입한다. 인메모리 상태를 쓰므로 워커는 1개로 고정.
CMD exec uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080} --workers 1
