#! /usr/bin/python3
# -*- coding: utf-8 -*-

"""
.. include:: ../README.md
"""

__version__ = "0.5.0.alpha"

from dolfin import *
from dolfin_adjoint import *
from .core import Module
from .utils import to_numpy, from_numpy
from .filters import helmholtzFilter, hevisideFilter, b2c, helmholtzFilter_duplicated
from .optimizer import Optimizer, optimize_duplicated