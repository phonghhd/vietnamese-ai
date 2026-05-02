from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="vietnamese-ai",
    version="1.0.0",
    author="EvoNet AI Team",
    author_email="huynhduongphong9@gmail.com",
    description="Framework AI thuần tiếng Việt cho các kỹ sư Việt Nam",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/phonghhd/vietnamese-ai",
    packages=find_packages(),
    entry_points={
        "console_scripts": [
            "vai=vietnamese_ai.cli.main:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Natural Language :: Vietnamese",
    ],
    python_requires=">=3.7",
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
        "dev": [
            "pytest>=6.0",
            "black>=21.0",
            "flake8>=3.8",
        ],
        "all": [
            "underthesea>=6.0",
            "pytest>=6.0",
            "black>=21.0",
            "flake8>=3.8",
        ],
    },
)
