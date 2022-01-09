#! /usr/bin/python3
# -*- coding: utf-8 -*-
''' Solvers.
[detail]
'''

from dolfin import *
from dolfin_adjoint import *
import numpy as np

if not has_linear_algebra_backend("PETSc"):
    print("DOLFIN has not been configured with PETSc. Exiting.")
    exit()

parameters["linear_algebra_backend"] = "PETSc"

def _build_nullspace_3D(V, x):
    nullspace_basis = [x.copy() for i in range(6)]

    V.sub(0).dofmap().set(nullspace_basis[0], 1.0);
    V.sub(1).dofmap().set(nullspace_basis[1], 1.0);
    V.sub(2).dofmap().set(nullspace_basis[2], 1.0);

    V.sub(0).set_x(nullspace_basis[3], -1.0, 1);
    V.sub(1).set_x(nullspace_basis[3],  1.0, 0);
    V.sub(0).set_x(nullspace_basis[4],  1.0, 2);
    V.sub(2).set_x(nullspace_basis[4], -1.0, 0);
    V.sub(2).set_x(nullspace_basis[5],  1.0, 1);
    V.sub(1).set_x(nullspace_basis[5], -1.0, 2);

    for x in nullspace_basis:
        x.apply("insert")

    basis = VectorSpaceBasis(nullspace_basis)
    basis.orthonormalize()

    return basis

def _build_nullspace_2D(V, x):
    nullspace_basis = [x.copy() for i in range(2)]

    V.sub(0).dofmap().set(nullspace_basis[0], 1.0);
    V.sub(1).dofmap().set(nullspace_basis[1], 1.0);

    for x in nullspace_basis:
        x.apply("insert")

    basis = VectorSpaceBasis(nullspace_basis)
    basis.orthonormalize()

    return basis

class AMGsolver():
    '''
    Class of the solver for the large static linear elasticity problem.
    The solver uses smoothed aggregation algerbaric multigrig method.
    
    Attributes:
    '''    
    def __init__(self, A, b):
        '''
        Class of the solver for the large static linear elasticity problem.
        The solver uses smoothed aggregation algerbaric multigrig method.

        Args:
            A (dolfin_adjoint.Form): Bilinear form
            b (dolfin_adjoint.Form): Linear form
        '''        
        self.A = A
        self.b = b

    def solve(self, u, V, monitor_convergence=True, build_null_space=None, mg_levels_ksp_type='chebyshev',
              mg_levels_pc_type='jacobi', mg_levels_esteig_ksp_type='cg', mg_levels_ksp_chebyshev_esteig_steps=50):
        '''
        Solving the problem with unknown vector `u`.

        Args:
            u (dolfin_adjoint.Function): Unknown function vector u.
            V (dolfin_adjoint.FunctionSpace): Function space of u.
            monitor_convergence (bool, optional): Monitor_convergence. Defaults to True.
            build_null_space (str, optional): The bilinear form `A` will be set to near null space. Defaults to None without null_space. Set to be 'x-D' when x-demension.
            mg_levels_ksp_type (str, optional): Defaults to 'chebyshev'.
            mg_levels_pc_type (str, optional): Defaults to 'jacobi'.
            mg_levels_esteig_ksp_type (str, optional): Defaults to 'cg'.
            mg_levels_ksp_chebyshev_esteig_steps (int, optional): Defaults to 50.

        Returns:
            dolfin_adjoint.vector: Solution
        '''
        if build_null_space == '2-D':
            try:
                null_space = _build_nullspace_2D(V, u.vector())
            except ValueError as v:
                raise ValueError('Invalid demension is detected. The number of subspaces is assumed {}.'.format(V.num_sub_spaces()))
            as_backend_type(self.A).set_near_nullspace(null_space)

        elif build_null_space == '3-D':
            try:
                null_space = _build_nullspace_3D(V, u.vector())
            except ValueError as v:
                raise ValueError('Invalid demension is detected. The number of subspaces is assumed {}.'.format(V.num_sub_spaces()))
            as_backend_type(self.A).set_near_nullspace(null_space)

        elif build_null_space is None:
            pass
        else:
            raise ValueError("Invalid key is detected: Set to be 'x-D' when x-demension.")

        pc = PETScPreconditioner("petsc_amg")
        PETScOptions.set("mg_levels_ksp_type", mg_levels_ksp_type)
        PETScOptions.set("mg_levels_pc_type", mg_levels_pc_type)
        PETScOptions.set("mg_levels_esteig_ksp_type", mg_levels_esteig_ksp_type)
        PETScOptions.set("mg_levels_ksp_chebyshev_esteig_steps", mg_levels_ksp_chebyshev_esteig_steps)
        PETSC_solver = PETScKrylovSolver("cg", pc)
        PETSC_solver.parameters["monitor_convergence"] = monitor_convergence
        PETSC_solver.set_operator(self.A)
        PETSC_solver.solve(u.vector(), self.b)
        return u