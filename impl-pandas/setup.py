"""Setup configuration for HPI-FHFA package."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements-minimal.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="hpi-fhfa",
    version="0.1.0",
    author="HPI-FHFA Implementation Team",
    description="Federal Housing Finance Agency Repeat-Sales Aggregation Index Implementation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourorg/hpi-fhfa",
    packages=find_packages(exclude=["tests", "tests.*"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.2.0",
            "pytest-cov>=4.0.0",
            "hypothesis>=6.60.0",
            "black>=22.10.0",
            "flake8>=5.0.0",
            "mypy>=0.990",
            "isort>=5.10.0",
        ],
        "docs": [
            "sphinx>=5.3.0",
            "sphinx-rtd-theme>=1.1.0",
            "nbsphinx>=0.8.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "hpi-fhfa=hpi_fhfa.cli:main",
        ],
    },
)