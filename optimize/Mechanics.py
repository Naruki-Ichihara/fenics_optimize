#! /usr/bin/python3
# -*- coding: utf-8 -*-
''' Helper module for machanical engineering problems.

This source contains useful methods for the elasticity problem.
'''

from dolfin import *
from dolfin_adjoint import *
from .shells import *

def sigma(v, E, nu):
    '''
    Compute stress tensor form.

    Args:
        v (dolfin_adjoint.Function): Displacement vector
        E (float): Yound's modulus
        nu (float): Poisson's ratio

    Returns:
        form (dolfin_adjoint.Form): Stress tensor
    '''
    mu = E/(2.0*(1.0 + nu))
    lmbda = E*nu/((1.0 + nu)*(1.0 - 2.0*nu))
    return 2.0*mu*sym(grad(v)) + lmbda*tr(sym(grad(v)))*Identity(len(v))

def epsilon(v):
    '''
    Compute strain tensor form.

    Args:
        v (dolfin_adjoint.Function): Displacement vector

    Returns:
        form (dolfin_adjoint.Form): Strain tensor
    '''
    return sym(grad(v))