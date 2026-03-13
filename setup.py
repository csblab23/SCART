from setuptools import setup, find_packages

# read requirements from requirements.txt
with open("requirements.txt") as f:
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


