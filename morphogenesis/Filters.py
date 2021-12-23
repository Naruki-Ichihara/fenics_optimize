#! /usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (C) 2021 Naruki Ichihara
#
# This file is part of Morphogenesis
#
# SPDX-License-Identifier:    MIT
"""Filters"""

from dolfin import *
from dolfin_adjoint import *
import numpy as np

def helmholtzFilter(u, U, R=0.025):
    """# helmholtzFilter
    Apply the helmholtz filter.
    Args:
        u (fenics.function): target function
        U (fenics.FunctionSpace): Functionspace of target function
        r (float): filter radius
    Return:
        v (fenics.function): filtered function
    """
    v = TrialFunction(U)
    dv = TestFunction(U)
    vh = Function(U)
    a = R*inner(grad(v), grad(dv))*dx + dot(v, dv)*dx
    L = inner(u, dv)*dx
    solve(a==L, vh, solver_parameters={'linear_solver': "superlu_dist"})
    return vh

def hevisideFilter(u, a=10, offset=0.01):
    """# hevisideFilter

    Apply the heviside function (approximate with sigmoid function)
    Args:
        u (fenics.function): target function
        U (fenics.FunctionSpace): Functionspace of target function
        a (float): coefficient. a>50 -> Step. a=3 -> S-shape
    Returns:
        v (fenics.function): filterd function
    Note:
    """
    return (1 / (1 + exp(-a*u)))*(1-offset) + offset

def isoparametric2Dfilter(z, e):
    """# isoparametric2Dfilter

    Apply 2D isoparametric projection onto orientation vector.

    Args:
        z: 0-component of the orientation vector (on natural setting)
        e: 1-component of the orientation vector (on natural setting)

    Returns:
        [Nx, Ny] (fenics.vector): Orientation vector with unit circle boundary condition on real setting.
    """
    u = as_vector([-1/np.sqrt(2), 0, 1/np.sqrt(2), 1, 1/np.sqrt(2), 0, -1/np.sqrt(2), -1])
    v = as_vector([-1/np.sqrt(2), -1, -1/np.sqrt(2), 0, 1/np.sqrt(2), 1, 1/np.sqrt(2), 0])
    N1 = -(1-z)*(1-e)*(1+z+e)/4
    N2 =  (1-z**2)*(1-e)/2
    N3 = -(1+z)*(1-e)*(1-z+e)/4
    N4 =  (1+z)*(1-e**2)/2
    N5 = -(1+z)*(1+e)*(1-z-e)/4
    N6 =  (1-z**2)*(1+e)/2
    N7 = -(1-z)*(1+e)*(1+z-e)/4
    N8 =  (1-z)*(1-e**2)/2
    N = as_vector([N1, N2, N3, N4, N5, N6, N7, N8])
    Nx = inner(u, N)
    Ny = inner(v, N)
    return as_vector((Nx, Ny))