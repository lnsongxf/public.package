#!/usr/bin/env python
""" This module serves as the interface for manual tests of the GRMPY package.
"""

# standard library
import sys
import os

# GRMPY import
sys.path.insert(0, os.environ['GRMPY'])
from grmpy import *

simulate('test.grmpy.ini')

estimate(use_simulation=True, init='test.grmpy.ini')