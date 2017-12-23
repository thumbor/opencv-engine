#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging

__version__ = '1.0.1'

try:
    from opencv_engine.engine_cv3 import Engine  # NOQA
except ImportError:
    logging.exception('Could not import opencv_engine. Probably due to setup.py installing it.')
