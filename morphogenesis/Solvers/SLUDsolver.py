#! /usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (C) 2021 Naruki Ichihara
#
# This file is part of Morphogenesis
#
# SPDX-License-Identifier:    MIT
"""SLUD solver for the elasticity problems"""

from dolfin import *
from dolfin_adjoint import *
import numpy as np

# Test for PETSc
if not has_linear_algebra_backend("PETSc"):
    print("DOLFIN has not been configured with PETSc. Exiting.")
    exit()

# Set backend to PETSC
parameters["linear_algebra_backend"] = "PETSc"

class SLUDsolver():
    """# SLUDsolver
    ## SLUDsolver
    Class of the solver for the large static linear elasticity problem.
    The solver uses Super LU dist method.

    ### Args:
        a : bilinear form
        L : linear form
        bcs (list) : boundary conditions

    ### Examples:

    ### Note:

    """
    def __init__(self, a, L, bcs):
        self.a = a
        self.L = L
        self.bcs = bcs

    def forwardSolve(self, u, V, monitor_convergence=True):
        """# AMGsolver
        ## solver
        ### forwardSolve
        solve the problem.
        ### Args:
            u : function
            V : function space
            monitor_convergence (bool) : igonored
        ### Returns:

        ### Examples:

        ### Note:

        """
        A, b = assemble_system(self.a, self.L, self.bcs)

        PETSC_solver = PETScLUSolver("superlu_dist")
        PETSC_solver.set_operator(A)
        PETSC_solver.solve(u.vector(), b)
        return u
    
class MUMPSsolver():
    """# SLUDsolver
    ## SLUDsolver
    Class of the solver for the large static linear elasticity problem.
    The solver uses Super LU dist method.

    ### Args:
        a : bilinear form
        L : linear form
        bcs (list) : boundary conditions

    ### Examples:

    ### Note:

    """
    def __init__(self, a, L, bcs):
        self.a = a
        self.L = L
        self.bcs = bcs

    def forwardSolve(self, u, V, monitor_convergence=True):
        """# AMGsolver
        ## solver
        ### forwardSolve
        solve the problem.
        ### Args:
            u : function
            V : function space
            monitor_convergence (bool) : igonored
        ### Returns:

        ### Examples:

        ### Note:

        """
        A, b = assemble_system(self.a, self.L, self.bcs)

        PETSC_solver = PETScLUSolver("mumps")
        PETSC_solver.set_operator(A)
        PETSC_solver.solve(u.vector(), b)
        return u



        
