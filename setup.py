#!/usr/bin/env python

"""
The setup script for pip. Allows for `pip install -e .` installation.
"""

from setuptools import setup, find_packages

requirements = ['numpy', 'matplotlib', 'torch']
setup_requirements = []
tests_requirements = ['pytest']
with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='fdmoperators',
    version='0.1',
    author='G. Bogopolsky, L. Cheng, E. Ajuria',
    author_email='bogopolsky@cerfacs.fr',
    url='https://github.com/gbogopolsky/fdm-operators',
    description='PyTorch and NumPy implementation of the physical operators (gradient, divergence, curl, laplacian) '
                'in Finite Difference formulation over rectangular cartesian grids.',
    long_description=long_description,
    keywords='plasma poisson deep learning pytorch',
    license='GNU General Public License v3',

    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.8'
    ],
    packages=find_packages(include=['PlasmaNet']),

    install_requires=requirements,
    include_package_data=True,
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=tests_requirements,
    zip_safe=False,
)
