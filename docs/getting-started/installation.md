# Cài đặt

## Yêu cầu hệ thống

- Python >= 3.8
- pip hoặc conda

## Cài đặt cơ bản

```bash
pip install vietnamese-ai
```

## Cài đặt từ source

```bash
git clone https://github.com/phonghhd/vietnamese-ai.git
cd vietnamese-ai
python -m venv venv
source venv/bin/activate
pip install -e ".[all]"
```

## Optional dependencies

```bash
# NLP tiếng Việt (underthesea)
pip install "vietnamese-ai[nlp]"

# Development tools
pip install "vietnamese-ai[dev]"

# Documentation
pip install "vietnamese-ai[docs]"

# Tất cả
pip install "vietnamese-ai[all]"
```

## Docker

```bash
docker build -t vietnamese-ai .
docker run -p 8080:8080 vietnamese-ai serve --model model.pkl --port 8080
```
