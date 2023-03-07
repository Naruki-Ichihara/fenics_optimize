#! /usr/bin/python3
# -*- coding: utf-8 -*-

__version__ = "0.3.0.alpha"

from dolfin import *
from dolfin_adjoint import *
from .core import Module
from .utils import to_numpy, from_numpy
from .filters import helmholtzFilter, hevisideFilter, box2circleConstraint
from .optimizer import optimize