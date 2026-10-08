# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 coccinella-labs
from setuptools import find_packages, setup

setup(
    name="bitinfer",
    version="0.1.0",
    description="Hugging Face encoder inference for Apple Silicon with float16 weights",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.20.0",
        "accelerate>=0.20.0",
        "numpy>=1.21.0",
    ],
    python_requires=">=3.9",
    author="coccinella-labs",
    url="https://github.com/coccinella-labs/bitinfer",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3 :: Only",
    ],
)
