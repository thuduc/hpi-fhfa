from setuptools import setup, find_packages

setup(
    name="hpi-fhfa-pyspark",
    version="0.1.0",
    author="Data Team",
    description="PySpark implementation of FHFA House Price Index calculation",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "pyspark>=3.5.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "pandas>=2.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-spark>=0.6.0",
            "pytest-cov>=4.1.0",
            "great-expectations>=0.17.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "hpi-pipeline=hpi_fhfa.pipeline.main_pipeline:main",
        ],
    },
)