"""Setup configuration for fitanalyzer package."""
from setuptools import setup, find_packages
from pathlib import Path

# Read README
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="fitanalyzer",
    version="0.1.0",
    author="FIT Analyzer Contributors",
    description="A Python library for analyzing Garmin FIT files and calculating training metrics",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/fitanalyzer",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.8",
    install_requires=[
        "fitparse>=1.2.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "garth>=0.5.0",
        "python-dateutil>=2.8.0",
    ],
    extras_require={
        "dev": [
            "pytest>=8.0.0",
            "pytest-cov>=4.0.0",
            "pytest-mock>=3.0.0",
            "flake8>=7.0.0",
            "black>=24.0.0",
            "mypy>=1.8.0",
            "isort>=5.13.0",
            "pylint>=3.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "fitanalyzer-parse=fitanalyzer.parser:main",
            "fitanalyzer-sync=fitanalyzer.sync:main",
            "fitanalyzer-setup=fitanalyzer.credentials:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Healthcare Industry",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
    ],
    keywords="garmin fit fitness training analysis metrics",
)
