#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages
from opencv_engine import __version__

tests_require = [
    'mock',
    'nose',
    'coverage',
    'yanc',
    'colorama',
    'preggy',
    'ipdb',
    'coveralls',
    'numpy',
    'colour',
]

setup(
    name='opencv_engine',
    version=__version__,
    description='OpenCV imaging engine for thumbor.',
    long_description='''
OpenCV imaging engine for thumbor.
''',
    keywords='thumbor imaging opencv',
    author='Globo.com',
    author_email='timehome@corp.globo.com',
    url='',
    license='MIT',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Operating System :: MacOS',
        'Operating System :: POSIX',
        'Operating System :: Unix',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 2.7',
    ],
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'colour',
        'numpy',
        'tornado>=4.1.0,<5.0.0',
    ],
    extras_require={
        'tests': tests_require,
    }
)
