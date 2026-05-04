from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="vietnamese-ai",
    version="10.0.0",
    author="EvoNet AI Team",
    author_email="huynhduongphong9@gmail.com",
    description="Framework AI thuần tiếng Việt cho Python - Học máy đơn giản, API tiếng Việt, Production-ready",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/phonghhd/vietnamese-ai",
    project_urls={
        "Documentation": "https://phonghhd.github.io/vietnamese-ai",
        "Bug Tracker": "https://github.com/phonghhd/vietnamese-ai/issues",
        "Changelog": "https://github.com/phonghhd/vietnamese-ai/blob/main/CHANGELOG.md",
        "Source": "https://github.com/phonghhd/vietnamese-ai",
    },
    packages=find_packages(exclude=["tests", "tests.*", "examples", "examples.*", "docs"]),
    entry_points={
        "console_scripts": [
            "vai=vietnamese_ai.cli.main:main",
        ],
    },
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Natural Language :: Vietnamese",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.19.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "matplotlib>=3.4.0",
    ],
    extras_require={
        "nlp": [
            "underthesea>=6.0",
        ],
        "torch": [
            "torch>=2.0",
        ],
        "transformers": [
            "transformers>=4.30",
            "torch>=2.0",
        ],
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=4.0",
            "black>=23.0",
            "ruff>=0.1.0",
            "mypy>=1.0",
        ],
        "docs": [
            "mkdocs>=1.5",
            "mkdocs-material>=9.0",
            "mkdocstrings[python]>=0.24",
        ],
        "all": [
            "underthesea>=6.0",
            "torch>=2.0",
            "transformers>=4.30",
            "pytest>=7.0",
            "pytest-cov>=4.0",
            "black>=23.0",
            "ruff>=0.1.0",
            "mypy>=1.0",
            "mkdocs>=1.5",
            "mkdocs-material>=9.0",
            "mkdocstrings[python]>=0.24",
        ],
    },
)
