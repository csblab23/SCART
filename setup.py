from setuptools import setup, find_packages

setup(
    name="scT-CAR_Designer",
    version="0.1.0",
    description="Single-cell tumor CAR target design pipeline",
    author="Your Name",
    packages=find_packages(),
    install_requires=[
        "GEOparse>=1.34.0",
        "pandas>=1.3.0"
    ],
    python_requires=">=3.8",
)

