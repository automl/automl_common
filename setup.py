import setuptools

with open("README.md", "r") as f:
    long_description = f.read()

with open("requirements.txt", "r") as f:
    requirements = [line.strip() for line in f]


setuptools.setup(
    name="automl_common",
    version="0.0.1",
    author="AutoML Freiburg",
    author_email="feurerm@informatik.uni-freiburg.de",
    description="Shared utilities that AutoML frameworks may benefit from.",
    long_description=long_description,
    url="https://github.com/automl/automl_common",
    license="Apache License 2.0",
    keywords=" ".join(
        [
            "machine learning",
            "algorithm configuration",
            "hyperparameter optimization",
            "tuning",
            "neural architecture",
            "deep learning",
        ]
    ),
    packages=setuptools.find_packages(),
    python_requires=">=3.7",
    platforms=["Linux"],
    install_requires=requirements,
    extra_reqs={
        "dev": [
            "pytest",
            "pytest-cov",
            "pydocstyle",
            "flake8",
            "black",
            "isort",
            "mypy",
            "pre-commit",
        ]
    },
    include_package_data=True,
)
