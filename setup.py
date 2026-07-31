from setuptools import setup, find_packages

setup(
    name="sera",
    version="0.1.0",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=["torch>=2.0.0"],
    entry_points={
        "console_scripts": [
            "sera=sera.generate:main",
            "sera-train=sera.train:main",
        ],
    },
)
