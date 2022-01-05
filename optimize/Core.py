#! /usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (C) 2021 Naruki Ichihara
#
# This file is part of Morphogenesis
#
# SPDX-License-Identifier:    MIT
"""Core"""
from dolfin import *
from dolfin_adjoint import *
import numpy as np
from .fecr import from_numpy, to_numpy

def with_derivative(temp, wrt=None):
    """# Core
    ## forward
    Decorator to wrap the fenics chain. 

    Examples:
    ```
    @forward([X1, X2, ..., Xn], wrt=[0, 1, ..., m])
    def function(xs):
        Processes with xs[n]
        return cost
    ```
    """
    def _with_derivative(func):
        def wrapper(*args, **kwargs):
            x = np.split(args[0], len(temp))
            xs = []
            for pack in zip(x, temp):
                x, X = pack
                xs.append(from_numpy(x, Function(X)))
            res = func(xs, **kwargs)
            Js = []
            Hs = []
            if wrt is None:
                for x_ in xs:
                    control = Control(x_)
                    Js.append(to_numpy(compute_gradient(res, control)))
                J = np.concatenate(Js)
                return res, J
            else:
                for i in range(len(xs)):
                    if i not in wrt:
                        Js.append(to_numpy(xs[i])*0)
                    else:
                        control = Control(xs[i])
                        Js.append(to_numpy(compute_gradient(res, control)))
                J = np.concatenate(Js)
                return res, J
        return wrapper
    return _with_derivative

def without_derivative(temp):
    """# Core
    ## without_derivative
    Decorator to wrap the fenics chain. 

    Examples:
    ```
    @forward([X1, X2, ..., Xn])
    def function(xs):
        Processes with xs[n]
        return cost
    ```
    """
    def _without_derivative(func):
        def wrapper(*args, **kwargs):
            x = np.split(args[0], len(temp))
            xs = []
            for pack in zip(x, temp):
                x, X = pack
                xs.append(from_numpy(x, Function(X)))
            res = func(xs, **kwargs)
            return res
        return wrapper
    return _without_derivative
