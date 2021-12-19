#! /usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (C) 2021 Naruki Ichihara
#
# This file is part of Morphogenesis
#
# SPDX-License-Identifier:    MIT
"""Chain"""

from dolfin import *
from dolfin_adjoint import *
from fecr import from_numpy, to_numpy

def evalGradient(target, control):
    """# Helpers
    ## Chain
    ### evalGradient

    Evaluation gradient using the descrete-adjoint method.

    Args:
        target (assemble): target function
        control (fenics.Function): control
    
    Return:
        dJdx (numpy.ndarray): gradient
    """
    cont = Control(control)
    return to_numpy(compute_gradient(target, cont))

def numpy2fenics(u, U):
    """# Helpers
    ## Chain
    ### numpy2fenics
    Convert from ndarray to fenics function w.r.t. function space U.

    Args:
        u (ndarray): 
        U (fenics.FunctionSpace):
        
    Return:
        u (fenics.function):
    """
    return from_numpy(u, Function(U))

def fenics2numpy(u):
    """# Helpers
    ## Chain
    ### fennics2numpy
    Convert from fenics function to ndarray.
    Args:
        u (finics.function): 
    Return:
        u (ndarray)
    """
    return to_numpy(u)
