from setuptools import setup, find_packages

setup(
    name="rsai",
    version="1.0.0",
    description="Repeat-Sales Aggregation Index (RSAI) Model Implementation",
    author="RSAI Implementation Team",
    packages=find_packages(where="rsai"),
    package_dir={"": "rsai"},
    install_requires=[
        "pandas>=1.5.0",
        "numpy>=1.23.0",
        "statsmodels>=0.13.0",
        "scikit-learn>=1.1.0",
        "scipy>=1.9.0",
        "pydantic>=2.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
        ]
    },
    python_requires=">=3.8",
)