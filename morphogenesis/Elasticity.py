#! /usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (C) 2021 Naruki Ichihara
#
# This file is part of Morphogenesis
#
# SPDX-License-Identifier:    MIT
"""elasicity"""

from dolfin import *
from dolfin_adjoint import *

def sigma(v, E, nu):
    """# Helpers
    ## elasticity
    ### sigma
    Compute stress tensor form.
    Args:
        v (fenics.function): displacement vector
        E (float): Younds modulus
        nu (float): Poissons ratio
    Return:
        form
    """
    mu = E/(2.0*(1.0 + nu))
    lmbda = E*nu/((1.0 + nu)*(1.0 - 2.0*nu))
    return 2.0*mu*sym(grad(v)) + lmbda*tr(sym(grad(v)))*Identity(len(v))

def reducedSigma(rho, v, E, nu, p=3):
    """# Helpers
    ## elasticity
    ### reducedSigma
    Compute stress tensor form.
    Args:
        rho : density
        v (fenics.function): displacement vector
        E (float): Younds modulus
        nu (float): Poissons ratio
        p : penalty
    Return:
        form
    """
    E_red = E*rho**p
    mu = E_red/(2.0*(1.0 + nu))
    lmbda = E_red*nu/((1.0 + nu)*(1.0 - 2.0*nu))
    return 2.0*mu*sym(grad(v)) + lmbda*tr(sym(grad(v)))*Identity(len(v))

def epsilon(v):
    """# Helpers
    ## elasticity
    ### epsilon
    Compute strain tensor form.
    Args:
        v (fenics.function): displacement vector
    Return:
        form
    """
    return sym(grad(v))