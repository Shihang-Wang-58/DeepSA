from setuptools import setup, find_packages
import os

# Read README.md as long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Define dependencies
requirements = [
    "torch>=2.0.0",
    "torchvision",
    "torchaudio",
    "autogluon>=1.0.0",
    "rdkit>=2023.3.1",
    "pandas>=2.0.0",
    "numpy>=1.24.0",
    "scikit-learn>=1.2.0",
]

setup(
    name="deepsa",
    version="0.1.2",
    author="Shihang Wang",
    author_email="wangshh12022@shanghaitech.edu.cn",
    description="A Deep-learning Driven Predictor of Compound Synthesis Accessibility",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Shihang-Wang-58/DeepSA",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    include_package_data=True,
    package_data={
        "deepsa": ["model/*", "model/hf_text/*"],
    },
    entry_points={
        "console_scripts": [
            "deepsa-predict=deepsa.cli:predict_cli",
            "deepsa-train=deepsa.cli:train_cli",
        ],
    },
)