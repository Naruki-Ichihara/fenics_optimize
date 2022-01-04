#! /usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (C) 2021 Naruki Ichihara
#
# This file is part of Morphogenesis
#
# SPDX-License-Identifier:    MIT
"""Main module for Morphogenesis"""

__version__ = "0.0.4.alpha"

from .Core import *
from .Utilities import *
from .Solvers import AMGsolver, AMG2Dsolver, AMG3Dsolver, SLUDsolver, MUMPSsolver
from .Optimizer import HSLoptimize, MMAoptimize
from .Filters import *