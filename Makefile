.PHONY: help install test lint format docs clean build publish

help:
	@echo "Vietnamese AI Framework - Makefile"
	@echo ""
	@echo "  make install    Cài đặt dependencies"
	@echo "  make test       Chạy test suite"
	@echo "  make lint       Kiểm tra code quality"
	@echo "  make format     Format code (black + ruff)"
	@echo "  make docs       Build documentation"
	@echo "  make serve-docs Serve docs locally"
	@echo "  make clean      Xóa build artifacts"
	@echo "  make build      Build package"
	@echo "  make publish    Publish to PyPI"

install:
	pip install -e ".[all]"

test:
	pytest tests/ -v --tb=short

test-cov:
	pytest tests/ -v --tb=short --cov=vietnamese_ai --cov-report=html

lint:
	ruff check vietnamese_ai/ tests/
	mypy vietnamese_ai/ --ignore-missing-imports

format:
	black vietnamese_ai/ tests/ examples/
	ruff check vietnamese_ai/ tests/ --fix

docs:
	mkdocs build

serve-docs:
	mkdocs serve

clean:
	rm -rf build/ dist/ *.egg-info .pytest_cache .mypy_cache htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null
	find . -type f -name "*.pyc" -delete 2>/dev/null

build: clean
	python -m build

publish: build
	twine upload dist/*
