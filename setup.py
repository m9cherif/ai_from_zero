from setuptools import setup, find_packages

setup(
    name="myai",
    version="0.1.0",
    description="A completely original language model built from scratch",
    packages=find_packages(),
    python_requires=">=3.10",
    entry_points={
        "console_scripts": [
            "myai-train=myai.scripts.train:main",
            "myai-generate=myai.scripts.generate:main",
            "myai-tokenize=myai.scripts.tokenize:main",
            "myai-evaluate=myai.scripts.evaluate:main",
        ],
    },
)
