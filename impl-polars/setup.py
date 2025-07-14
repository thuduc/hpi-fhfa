"""Setup script for RSAI model implementation using Polars."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="rsai-polars",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Refined Stratified-Adjusted Index (RSAI) model implementation using Polars",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/rsai-polars",
    packages=find_packages(exclude=["tests", "tests.*", "docs", "scripts", "notebooks"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "pytest-xdist>=3.3.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.5.0",
            "isort>=5.12.0",
        ],
        "docs": [
            "sphinx>=7.0.0",
            "sphinx-rtd-theme>=1.3.0",
            "sphinx-autodoc-typehints>=1.24.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "rsai-train=rsai.cli.train:main",
            "rsai-predict=rsai.cli.predict:main",
            "rsai-evaluate=rsai.cli.evaluate:main",
        ],
    },
)