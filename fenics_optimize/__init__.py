#! /usr/bin/python3
# -*- coding: utf-8 -*-
"""
.. include:: ../README.md
"""

__version__ = "0.1.1.alpha"

from .Core import with_derivative, without_derivative, with_minmax_derivative
from .Utilities import Recorder, Logger
from .Solvers import AMGsolver
from .Optimizer import HSLoptimize, MMAoptimize
from .Filters import helmholtzFilter, hevisideFilter, isoparametric2Dfilter