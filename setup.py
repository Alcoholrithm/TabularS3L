# Copyright (c) Alcoholrithm
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
    name='tabs3l',
    version='0.1',
    description='test',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/Alcoholrithm/TabularS3L/tree/dev',
    author='Minwook Kim',
    author_email='kmiiiaa@pusan.ac.kr',

    # Note that this is a string of words separated by whitespace, not a list.
    keywords='tabular',
    packages=find_packages(exclude=["pl_modules"]),
    # packages=find_packages(),
    # packages=['tabs3l'],
    # package_dir={'': 'tabs3l'},
    # The `include_package_data` parameter in the `setup()` function is used to specify whether to
    # include non-Python files (such as data files, configuration files, etc.) that are part of the
    # package when it is installed.
    include_package_data=False,
    # install_requires=['torch >= 1.8', 'torchvision', 'torchaudio', 'transformers', 'timm', 'progress', 'ruamel.yaml', 'scikit-image', 'scikit-learn', 'tensorflow', ''],
    install_requires=install_requires,
    extras_require={
        "PL": ["pytorch_lightning"],
    },
    python_requires='>=3.7',
)