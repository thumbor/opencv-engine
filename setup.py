#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import find_packages, setup

from opencv_engine import __version__

tests_require = [
    "black",
    "colorama",
    "colour",
    "ipdb",
    "isort",
    "mock",
    "numpy",
    "preggy",
    "pytest",
    "pytest-cov",
]

setup(
    name="opencv_engine",
    version=__version__,
    description="OpenCV imaging engine for thumbor.",
    long_description="""
OpenCV imaging engine for thumbor.
""",
    keywords="thumbor imaging opencv",
    author="Globo.com",
    author_email="timehome@corp.globo.com",
    url="",
    license="MIT",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Operating System :: MacOS",
        "Operating System :: POSIX",
        "Operating System :: Unix",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 2.7",
    ],
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "colour",
        "numpy",
        "opencv-python",
        "thumbor",
    ],
    extras_require={
        "tests": tests_require,
    },
)
