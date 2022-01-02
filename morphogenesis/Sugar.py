#! /usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (C) 2021 Naruki Ichihara
#
# This file is part of Morphogenesis
#
# SPDX-License-Identifier:    MIT
"""Syntax sugar"""
from dolfin import *
from dolfin_adjoint import *
from fecr import from_numpy, to_numpy

def morphogen(temp):
    def _morphogen(func):
        def wrapper(*args, **kwargs):
            x = from_numpy(args[0], Function(temp))
            res = func(x, **kwargs)
            control = Control(x)
            J = to_numpy(compute_gradient(res, control))
            return res, J
        return wrapper
    return _morphogen