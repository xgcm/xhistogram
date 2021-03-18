#!/usr/bin/env python
import versioneer
from setuptools import setup, find_packages

DISTNAME = "xhistogram"
LICENSE = "MIT"
AUTHOR = "xhistogram Developers"
AUTHOR_EMAIL = "rpa@ldeo.columbia.edu"
URL = "https://github.com/xgcm/xhistogram"
CLASSIFIERS = [
    "Development Status :: 4 - Beta",
    "License :: OSI Approved :: Apache Software License",
    "Operating System :: OS Independent",
    "Intended Audience :: Science/Research",
    "Programming Language :: Python",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.6",
    "Topic :: Scientific/Engineering",
]

INSTALL_REQUIRES = ["xarray>=0.12.0", "dask", "numpy>=1.16"]
PYTHON_REQUIRES = ">=3.6"

DESCRIPTION = "Fast, flexible, label-aware histograms for numpy and xarray"


def readme():
    with open("README.rst") as f:
        return f.read()


setup(
    name=DISTNAME,
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    license=LICENSE,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    classifiers=CLASSIFIERS,
    description=DESCRIPTION,
    long_description=readme(),
    install_requires=INSTALL_REQUIRES,
    python_requires=PYTHON_REQUIRES,
    url=URL,
    packages=find_packages(),
)
