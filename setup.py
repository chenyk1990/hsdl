#!/usr/bin/env python
# -*- encoding: utf8 -*-
import io
import os

#from setuptools import find_packages
from setuptools import setup
from distutils.core import Extension
import numpy


long_description = """
Source code: https://github.com/chenyk1990/HSDL""".strip() 


def read(*names, **kwargs):
    return io.open(
        os.path.join(os.path.dirname(__file__), *names),
        encoding=kwargs.get("encoding", "utf8")).read()

from distutils.core import Extension
                                                
setup(
    name="HSDL",
    version="0.0.1",
    license='MIT License',
    description="HSDL: A novel and practical method to refine automatic earthquake catalog using hybrid shallow and deep learning",
    long_description=long_description,
    author="HSDL developing team",
    author_email="chenyk2016@gmail.com",
    url="https://github.com/chenyk1990/HSDL",
    packages=['HSDL'],
    include_package_data=True,
    zip_safe=False,
    classifiers=[
        # complete classifier list:
        # http://pypi.python.org/pypi?%3Aaction=list_classifiers
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Operating System :: Unix",
        "Operating System :: POSIX",
        "Operating System :: Microsoft :: Windows",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: Implementation :: CPython",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Physics"
    ],
    keywords=[
        "seismology", "earthquake seismology", "exploration seismology", "array seismology", "denoising", "science", "engineering", "structure", "local slope", "filtering"
    ],
    install_requires=[
        "numpy", "scipy", "matplotlib"
    ],
    extras_require={
        "docs": ["sphinx", "ipython", "runipy"]
    }
)
