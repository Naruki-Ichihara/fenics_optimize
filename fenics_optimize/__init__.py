#! /usr/bin/python3
# -*- coding: utf-8 -*-
"""
.. include:: ../README.md
"""

__version__ = "0.0.8.alpha"

from .Core import with_derivative, without_derivative, max_derivative_approximation
from .Utilities import Recorder, Logger
from .Solvers import AMGsolver
from .Optimizer import HSLoptimize, MMAoptimize
from .Filters import helmholtzFilter, hevisideFilter, isoparametric2Dfilter