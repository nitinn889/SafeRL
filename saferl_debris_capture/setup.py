"""
setup.py — saferl_debris_capture package

Installs the project as an editable package so that all sub-modules
can be imported without manually setting PYTHONPATH.

Usage
-----
# Editable install (recommended for development)
pip install -e .

# Standard install
pip install .
"""

from setuptools import setup, find_packages

setup(
    name="saferl_debris_capture",
    version="0.1.0",
    description=(
        "Probabilistic Shield-Augmented Reinforcement Learning "
        "for Autonomous Space Debris Capture"
    ),
    author="[Author Name]",
    python_requires=">=3.10",
    packages=find_packages(
        include=["saferl_debris_capture*"]
    ),
    install_requires=[
        "torch>=2.1",
        "stable-baselines3>=2.2",
        "gymnasium>=0.29",
        "numpy>=1.24",
        "scipy>=1.11",
        "matplotlib>=3.7",
        "pandas>=2.0",
        "pyyaml>=6.0",
        # Optional but recommended:
        # "wandb",
        # "isaacsim==4.2.0.2",
    ],
    extras_require={
        "dev": [
            "pytest",
            "pytest-cov",
            "black",
            "isort",
            "mypy",
        ]
    },
    entry_points={
        "console_scripts": [
            "saferl-train=saferl_debris_capture.training.train:main",
            "saferl-benchmark=saferl_debris_capture.evaluation.benchmark:main",
        ]
    },
)
