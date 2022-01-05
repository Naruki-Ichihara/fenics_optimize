#! /usr/bin/python3
# -*- coding: utf-8 -*-
''' Core source file for fenics-optimize.

This source contains some decorators that wrrap the fenics function to the numpy function
and calculate the Jacobian vector using the dolfin-adjoint.
'''

from dolfin import *
from dolfin_adjoint import *
import numpy as np
from .fecr import from_numpy, to_numpy

def with_derivative(temp, wrt=None):
    '''
    Decorator to wrap the fenics chain.

    Args:
        temp (list[dolfin_adjoint.FunctionSpace]): Function spaces that contain each control variables.
        wrt (list[int], optional): Automatic derivative of cost w.r.t. wrt index. Defaults to None.

    Returns:
    '''
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
    '''
    Decorator to wrap the fenics chain without derivative.

    Args:
        temp (list[dolfin_adjoint.FunctionSpace]): Function spaces that contain each control variables.

    Returns:
    '''
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