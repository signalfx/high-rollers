from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = f.readlines()

# Package (minimal) configuration
setup(
    name="approximate_grouped_quantiles",
    version="0.0.1",
    description="Streaming algorithms for quantiles by key",
    packages=find_packages(),
    install_requires=requirements,
)
