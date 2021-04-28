#!/usr/bin/env python

"""
Ref: https://github.com/argoai/argoverse-api/blob/master/setup.py
A setuptools based setup module.
See:
https://packaging.python.org/en/latest/distributing.html
https://github.com/pypa/sampleproject
"""

import platform
import sys
from codecs import open  # To use a consistent encoding
from os import path
from setuptools import setup, find_packages

here = path.abspath(path.dirname(__file__))

setup(
    name='WolfSim',
    version='1.0.1',
    url='https://github.gatech.edu/charris92/Wolfsim',
    license='',
    author='Caleb Harris (charris92)',
    author_email='caleb.harris94@gatech.edu',
    description='',
    python_requires=">= 3.5",
    packages=find_packages(),
    install_requires=['matplotlib', 'mesa', 'numpy', 'networkx', 'scipy']
)
