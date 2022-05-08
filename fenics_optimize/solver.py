#! /usr/bin/python3
# -*- coding: utf-8 -*-
''' Built-in solvers.
'''

from dolfin import *
from dolfin_adjoint import *

def amg(A, b, u, monitor_convergence=True, mg_levels_ksp_type='chebyshev', mg_levels_pc_type='jacobi', mg_levels_esteig_ksp_type='cg', mg_levels_ksp_chebyshev_esteig_steps=50):
    '''
    Solving the problem `Au=b` with unknown vector `u`.
    Args:
        A (dolfin_adjoint.Form): Bilinear form
        b (dolfin_adjoint.Form): Linear form
        u (dolfin_adjoint.Function): Unknown function vector u.
        monitor_convergence (bool, optional): Monitor_convergence. Defaults to True.
        mg_levels_ksp_type (str, optional): Defaults to 'chebyshev'.
        mg_levels_pc_type (str, optional): Defaults to 'jacobi'.
        mg_levels_esteig_ksp_type (str, optional): Defaults to 'cg'.
        mg_levels_ksp_chebyshev_esteig_steps (int, optional): Defaults to 50.
    Returns:
        dolfin_adjoint.vector: Solution
    '''
    pc = PETScPreconditioner("petsc_amg")
    PETScOptions.set("mg_levels_ksp_type", mg_levels_ksp_type)
    PETScOptions.set("mg_levels_pc_type", mg_levels_pc_type)
    PETScOptions.set("mg_levels_esteig_ksp_type", mg_levels_esteig_ksp_type)
    PETScOptions.set("mg_levels_ksp_chebyshev_esteig_steps", mg_levels_ksp_chebyshev_esteig_steps)
    PETSC_solver = PETScKrylovSolver("cg", pc)
    PETSC_solver.parameters["monitor_convergence"] = monitor_convergence
    PETSC_solver.set_operator(A)
    PETSC_solver.solve(u.vector(), b)
    return u