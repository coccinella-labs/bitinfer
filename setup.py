# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 harpertoken
from setuptools import find_packages, setup

setup(
    name="bitinfer",
    version="0.1.0",
    description="Fast quantized inference framework for Apple Silicon",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.20.0",
        "accelerate>=0.20.0",
        "numpy>=1.21.0",
    ],
    python_requires=">=3.9",
    author="harpertoken",
    author_email="harpertoken@icloud.com",
    url="https://github.com/harpertoken/bitinfer",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.9+",
    ],
)
