FROM python:3.11-slim AS builder

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

FROM python:3.11-slim

RUN groupadd -r vai && useradd -r -g vai -d /app vai

WORKDIR /app

COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY vietnamese_ai/ vietnamese_ai/
COPY setup.py pyproject.toml README.md LICENSE ./
RUN pip install --no-cache-dir -e .

RUN mkdir -p /app/models /app/data && chown -R vai:vai /app

USER vai

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import vietnamese_ai; print('ok')" || exit 1

ENTRYPOINT ["python", "-m", "vietnamese_ai.cli.main"]
CMD ["info"]
