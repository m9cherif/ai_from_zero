from setuptools import setup, find_packages

setup(
    name="myai",
    version="0.2.0",
    description="A completely original language model built from scratch",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.24.0",
        "torch>=2.0.0",
        "tqdm>=4.65.0",
        "pyyaml>=6.0",
    ],
    extras_require={
        "dev": ["pytest>=7.0"],
    },
    entry_points={
        "console_scripts": [
            "myai-train=myai.scripts.train:main",
            "myai-generate=myai.scripts.generate:main",
            "myai-tokenize=myai.scripts.tokenize:main",
            "myai-evaluate=myai.scripts.evaluate:main",
        ],
    },
)
