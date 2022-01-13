#! /usr/bin/python3
# -*- coding: utf-8 -*-
''' Core source file for fenics-optimize.

This source contains some decorators that wrrap the fenics function to the numpy function
and calculate the Jacobian vector using the dolfin-adjoint.
'''

from dolfin import *
from dolfin_adjoint import *
import numpy as np
from fecr import from_numpy, to_numpy

def with_derivative(temp, wrt=None):
    '''
    Decorator to wrap the fenics chain.

    Args:
        temp (list[dolfin_adjoint.FunctionSpace]): Function spaces that contain each control variables.
        wrt (list[int], optional): Automatic derivative of cost w.r.t. wrt index. 
                                   Defaults to None to calculate Jacobians for all controls.

    Returns:
    '''
    def _with_derivative(func):
        def wrapper(*args, **kwargs):
            try:
                split_size = len(temp)
            except TypeError:
                raise TypeError('Wrong type argments is detected. The temp argment must be list of the FunctionSpace.')

            split_index = []
            index = 0
            for template in temp:
                index += Function(template).vector().size()
                split_index.append(index)
            x = np.split(args[0], indices_or_sections=split_index)

            xs = []
            for pack in zip(x, temp):
                x, X = pack
                xs.append(from_numpy(x, Function(X)))

            res = func(xs, **kwargs)

            Js = []
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
            try:
                split_size = len(temp)
            except TypeError:
                raise TypeError('Wrong type argments is detected. The temp argment must be list of the FunctionSpace.')

            split_index = []
            index = 0
            for template in temp:
                index += Function(template).vector().size()
                split_index.append(index)
            x = np.split(args[0], indices_or_sections=split_index)

            xs = []
            for pack in zip(x, temp):
                x, X = pack
                xs.append(from_numpy(x, Function(X)))

            res = func(xs, **kwargs)
            return res
        return wrapper
    return _without_derivative

def max_derivative_approximation(temp, wrt=None, rho=50.0):
    '''
    Return max value and Jacobian of a given field. 
    KS function approximates the Jacobian for the max type cost function.
    Notably, this function returns strict max value.

    Args:
        temp (list): Function spaces that contain each control variables.
        wrt (list, optional): Automatic derivative of cost w.r.t. wrt index. 
                              Defaults to None to calculate Jacobians for all controls.
        rho (float, optional): Parameter. Lager rho provides great approximation but numerical instability.
                                Defaults to 50.0.
    '''    
    def _max_derivative_approximation(func):
        def wrapper(*args, **kwargs):
            try:
                split_size = len(temp)
            except TypeError:
                raise TypeError('Wrong type argments is detected. The temp argment must be list of the FunctionSpace.')

            split_index = []
            index = 0
            for template in temp:
                index += Function(template).vector().size()
                split_index.append(index)
            x = np.split(args[0], indices_or_sections=split_index)

            xs = []
            for pack in zip(x, temp):
                x, X = pack
                xs.append(from_numpy(x, Function(X)))

            res = func(xs, **kwargs)
            cost = 1/rho*ln(assemble(exp(rho*res)*dx))

            Js = []
            if wrt is None:
                for x_ in xs:
                    control = Control(x_)
                    F = assemble(exp(rho*res)*dx)
                    dF = compute_gradient(F, control)
                    Js.append(to_numpy(dF)/(F*rho))
                J = np.concatenate(Js)
                return cost, J
            else:
                for i in range(len(xs)):
                    if i not in wrt:
                        Js.append(to_numpy(xs[i])*0)
                    else:
                        control = Control(xs[i])
                        Js.append(to_numpy(compute_gradient(res, control)))
                J = np.concatenate(Js)
                return cost, J
        return wrapper
    return _max_derivative_approximation