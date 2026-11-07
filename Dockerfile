# ==============================================================================
# Vietnamese AI Framework - Ultra-Lightweight Production Dockerfile
# Khởi chạy 1-Click Vietnamese AI Web Studio Dashboard (Zero-Cost Deployment)
# ==============================================================================

FROM python:3.11-slim AS builder

WORKDIR /app

# Cài đặt công cụ build tối thiểu
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Cài đặt dependencies (Micro-Core siêu nhẹ)
COPY pyproject.toml setup.py README.md ./
COPY vietnamese_ai/ ./vietnamese_ai/

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir .

# ==============================================================================
# Runtime Stage (Giảm dung lượng image xuống mức tối thiểu ~120MB)
# ==============================================================================
FROM python:3.11-slim AS runner

WORKDIR /app

# Sao chép các gói thư viện đã cài đặt từ builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin/vai /usr/local/bin/vai
COPY --from=builder /app /app

EXPOSE 8080

# Chẩn đoán sức khỏe container
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/api/overview')" || exit 1

# Khởi động Web Studio Dashboard trực quan
ENTRYPOINT ["vai", "studio", "--host", "0.0.0.0", "--port", "8080"]
