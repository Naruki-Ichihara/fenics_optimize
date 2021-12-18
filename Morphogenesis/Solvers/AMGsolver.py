#! /usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (C) 2021 Naruki Ichihara
#
# This file is part of Morphogenesis
#
# SPDX-License-Identifier:    MIT
"""AMG solver for the elasticity problems"""

from dolfin import *
from dolfin_adjoint import *
import numpy as np

# Test for PETSc
if not has_linear_algebra_backend("PETSc"):
    print("DOLFIN has not been configured with PETSc. Exiting.")
    exit()

# Set backend to PETSC
parameters["linear_algebra_backend"] = "PETSc"

def build_nullspace(V, x):
    """Function to build null space for 3D elasticity"""

    # Create list of vectors for null space
    nullspace_basis = [x.copy() for i in range(6)]

    # Build translational null space basis
    V.sub(0).dofmap().set(nullspace_basis[0], 1.0);
    V.sub(1).dofmap().set(nullspace_basis[1], 1.0);
    V.sub(2).dofmap().set(nullspace_basis[2], 1.0);

    # Build rotational null space basis
    V.sub(0).set_x(nullspace_basis[3], -1.0, 1);
    V.sub(1).set_x(nullspace_basis[3],  1.0, 0);
    V.sub(0).set_x(nullspace_basis[4],  1.0, 2);
    V.sub(2).set_x(nullspace_basis[4], -1.0, 0);
    V.sub(2).set_x(nullspace_basis[5],  1.0, 1);
    V.sub(1).set_x(nullspace_basis[5], -1.0, 2);

    for x in nullspace_basis:
        x.apply("insert")

    # Create vector space basis and orthogonalize
    basis = VectorSpaceBasis(nullspace_basis)
    basis.orthonormalize()

    return basis

class AMGsolver():
    """# AMGsolver
    ## AMGsolver
    Class of the solver for the large static linear elasticity problem.
    The solver uses smoothed aggregation algerbaric multigrig method.

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
            monitor_convergence (bool) :
        ### Returns:

        ### Examples:

        ### Note:

        """
        A, b = assemble_system(self.a, self.L, self.bcs)
        null_space = build_nullspace(V, u.vector())
        as_backend_type(A).set_near_nullspace(null_space)

        pc = PETScPreconditioner("petsc_amg")
        PETScOptions.set("mg_levels_ksp_type", "chebyshev")
        PETScOptions.set("mg_levels_pc_type", "jacobi")
        PETScOptions.set("mg_levels_esteig_ksp_type", "cg")
        PETScOptions.set("mg_levels_ksp_chebyshev_esteig_steps", 50)
        PETSC_solver = PETScKrylovSolver("cg", pc)
        PETSC_solver.parameters["monitor_convergence"] = monitor_convergence
        PETSC_solver.set_operator(A)
        PETSC_solver.solve(u.vector(), b)
        return u
        



        