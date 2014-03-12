#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging

__version__ = '0.1.0'

try:
    from opencv_engine.engine import Engine  # NOQA
except ImportError:
    logging.warning('Could not import opencv_engine. Probably due to setup.py installing it.')
