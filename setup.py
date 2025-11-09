"""
Setup script for Atlas: Map. Decide. Optimize.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

setup(
    name="atlas-iot",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Atlas: Map. Decide. Optimize. - Hybrid DQN-PPO-GNN resource allocator for IoT edge computing",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/ai-edge-allocator",
    project_urls={
        "Bug Tracker": "https://github.com/yourusername/ai-edge-allocator/issues",
        "Documentation": "https://github.com/yourusername/ai-edge-allocator/wiki",
        "Source Code": "https://github.com/yourusername/ai-edge-allocator",
    },
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.9",
    install_requires=[
        "gymnasium>=0.29.0",
        "stable-baselines3>=2.1.0",
        "torch>=2.0.0",
        "torch-geometric>=2.4.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "pyyaml>=6.0",
        "tensorboard>=2.14.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "networkx>=3.1",
        "tqdm>=4.65.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.7.0",
            "flake8>=6.1.0",
            "mypy>=1.5.0",
            "jupyter>=1.0.0",
            "jupyterlab>=4.0.0",
        ],
        "docs": [
            "sphinx>=7.1.0",
            "sphinx-rtd-theme>=1.3.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "atlas=src.main:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
