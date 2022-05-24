#!/usr/bin/env python3

import sys
import setuptools

with open("README.md", "r") as fd:
    long_description = fd.read()

setuptools.setup(
    name="mh_python",
    version="0.0.11",
    author="Chaoqing Wang",
    author_email="chaoqingwang.nick@gmail.com",
    description="Matlab to Python/Numpy translator",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://translate.mat2py.org",
    license="GNU Affero General Public License v3",
    packages=["mh_python"],
    install_requires=["miss-hit-core>=0.9.30"],
    python_requires=">=3.6, <4",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: GNU Affero General Public License v3 or later (AGPLv3+)",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering",
        "Topic :: Software Development :: Compilers",
        "Topic :: Software Development :: Quality Assurance",
        "Topic :: Utilities"
    ],
    entry_points={
        "console_scripts": [
            "mh_python = mh_python:main",
        ],
    },
)
