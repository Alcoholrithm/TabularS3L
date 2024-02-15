# Copyright Alcoholrithm
# Licensed under the MIT License.

""" Setup
"""
from setuptools import setup, find_packages
from codecs import open
from os import path
import pathlib

import pkg_resources

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# requirements
with pathlib.Path('requirements.txt').open() as requirements_txt:
    install_requires = [
        str(requirement)
        for requirement
        in pkg_resources.parse_requirements(requirements_txt)
    ]

setup(
    name='ts3l',
    version='v0.10',
    description='A PyTorch-based library for self- and semi-supervised learning tabular models.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/Alcoholrithm/TabularS3L',
    author='Minwook Kim',
    author_email='kmiiiaa@pusan.ac.kr',

    # Note that this is a string of words separated by whitespace, not a list.
    keywords='tabular-data semi-supervised-learning self-supervised-learning VIME SubTab SCARF',
    packages=find_packages(),
    # The `include_package_data` parameter in the `setup()` function is used to specify whether to
    # include non-Python files (such as data files, configuration files, etc.) that are part of the
    # package when it is installed.
    include_package_data=False,
    install_requires=install_requires,
    python_requires='>=3.7',
)