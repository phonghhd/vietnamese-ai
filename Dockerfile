FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN pip install -e .

EXPOSE 8080

ENTRYPOINT ["python", "-m", "vietnamese_ai.cli.main"]
CMD ["info"]
