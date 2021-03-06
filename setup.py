import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="cognibench",
    version="0.1.0",
    author="",
    author_email="",
    description="A benchmarking framework for cognitive models using behavioral data built on top of sciunit and gym",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/fmelinscak/cognibench",
    packages=setuptools.find_packages(),
    install_requires=[
        "numpy>=1.16.2",
        "scipy>=1.2.1",
        "pandas>=0.24.2",
        "gym",
        "sciunit",
        "overrides",
        "oct2py",
        "rpy2",
    ],
    python_requires=">=3.6.8",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Natural Language :: English",
        "Topic :: Scientific/Engineering",
    ],
)
