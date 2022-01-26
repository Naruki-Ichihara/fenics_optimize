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

            res, J = func(xs, **kwargs)
            return res, J
        return wrapper
    return _without_derivative

def with_minmax_derivative(temp, wrt=None, method='P-norm', **params):
    '''
    Return max value and Jacobian of a given field. 
    P-norm of KS function approximates the Jacobian for the max type cost function.
    Notably, this function returns strict max value.

    Args:
        temp (list): Function spaces that contain each control variables.
        wrt (list, optional): Automatic derivative of cost w.r.t. wrt index. 
                              Defaults to None to calculate Jacobians for all controls.
        method (str, optional): Set the approximation method. 'P-norm' or 'KS' is available. Defaults to 'p-norm'.
        P (float, optional): P value for the p-norm.
        k (float, optional): k value for the KS approximation.
    '''    
    def _with_minmax_derivative(func):
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
            # cost = res.vector().max()
            # cost = 1/rho*ln(assemble(exp(rho*res)*dx))

            Js = []
            
            if method == 'KS':
                try:
                    k = params['k']
                except KeyError:
                    raise  KeyError('Invalid key is detected. Please give "k" value bacause "KS" method is selected.')
                cost = ln(assemble(exp(k*res)*dx))/k
                if wrt is None:
                    for x_ in xs:
                        control = Control(x_)
                        F = assemble(exp(k*res)*dx)
                        dF = compute_gradient(F, control)
                        Js.append(to_numpy(dF)/(F*k))
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

            elif method == 'P-norm':
                try:
                    P = params['P']
                except KeyError:
                    raise  KeyError('Invalid key is detected. Please give "P" value bacause "P-norm" method is selected.')
                cost = assemble(res**P*dx)**(1/P)
                if wrt is None:
                    for x_ in xs:
                        control = Control(x_)
                        F = assemble(res**P*dx)**(1/P)
                        dF = compute_gradient(F, control)
                        Js.append(to_numpy(dF))
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
            else:
                raise KeyError('Unknown method "{}" is ordered. Only supported "P-norm" and "KS".'.format(method))

        return wrapper
    return _with_minmax_derivative