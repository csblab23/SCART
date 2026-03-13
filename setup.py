from setuptools import setup, find_packages
from pathlib import Path

req_file = Path(__file__).parent / "requirements.txt"

requirements = []
if req_file.exists():
    with open(req_file) as f:
        requirements = f.read().splitlines()

setup(
    name="SCART",
    version="0.1.0",
    description="Single-cell Antigen Ranking Tool",
    author="Vinaya S",
    packages=find_packages(),
    install_requires=requirements,
    python_requires=">=3.8",
    include_package_data=True,
    zip_safe=False,
)
