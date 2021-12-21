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