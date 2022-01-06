#! /usr/bin/python3
# -*- coding: utf-8 -*-
"""
.. include:: ../README.md
"""

__version__ = "0.0.5.alpha"

from .Core import with_derivative, without_derivative
from .Utilities import Recorder, Logger, evalGradient
from .Solvers import AMGsolver, AMG2Dsolver, AMG3Dsolver, SLUDsolver, MUMPSsolver
from .Optimizer import HSLoptimize, MMAoptimize
from .Filters import helmholtzFilter, hevisideFilter, isoparametric2Dfilter

__all__ = [
    'with_derivative',
    'without_derivative',
    'Recorder',
    'Logger',
    'HSLoptimize',
    'MMAoptimize',
    'helmholtzFilter',
    'isoparametric2Dfilter'
]