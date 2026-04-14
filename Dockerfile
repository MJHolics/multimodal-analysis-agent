# ── Build stage ──────────────────────────────────────────────
FROM python:3.13-slim AS builder

WORKDIR /app

# 시스템 의존성 (chromadb 빌드에 필요)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt


# ── Runtime stage ─────────────────────────────────────────────
FROM python:3.13-slim

WORKDIR /app

# 빌드 스테이지에서 설치된 패키지만 복사
COPY --from=builder /usr/local/lib/python3.13/site-packages /usr/local/lib/python3.13/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# 앱 소스 복사
COPY api/        ./api/
COPY data/       ./data/
COPY run_server.py .

# .env는 런타임에 주입 (docker-compose 또는 -e 플래그 사용)
# COPY .env .  ← 절대 이미지에 포함하지 않음

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
