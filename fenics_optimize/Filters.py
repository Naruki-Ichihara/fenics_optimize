#! /usr/bin/python3
# -*- coding: utf-8 -*-
''' Filters for homogenization based topology optimization problems.

[Detail]
'''

from dolfin import *
from dolfin_adjoint import *
import numpy as np

def helmholtzFilter(u, U, R=0.025):
    '''
    Apply the helmholtz filter.

    Args:
        u (dolfin_adjoint.Function): Target function
        U (dolfin_adjoint.FunctionSpace): Functionspace of target function
        R (float, optional): Filter radius. Defaults to 0.025.

    Returns:
        (dolfin_adjoint.Function): Filtered function
    '''
    v = TrialFunction(U)
    dv = TestFunction(U)
    vh = Function(U)
    a = R**2*inner(grad(v), grad(dv))*dx + dot(v, dv)*dx
    L = inner(u, dv)*dx
    A, b = assemble_system(a, L)
    pc = PETScPreconditioner("petsc_amg")
    PETScOptions.set("mg_levels_ksp_type", "chebyshev")
    PETScOptions.set("mg_levels_pc_type", "jacobi")
    PETScOptions.set("mg_levels_esteig_ksp_type", "cg")
    PETScOptions.set("mg_levels_ksp_chebyshev_esteig_steps", 50)
    PETSC_solver = PETScKrylovSolver("cg", pc)
    PETSC_solver.parameters["monitor_convergence"] = False
    PETSC_solver.set_operator(A)
    PETSC_solver.solve(vh.vector(), b)
    return vh

def hevisideFilter(u, V, a=10.0, offset=0.001):
    '''
    Apply the heviside function (approximate with sigmoid function).

    Args:
        u (dolfin_adjoint.Function): Target function.
        V (dolfin_adjoint.FunctionSpace): Function space for function.
        a (float, optional): Coefficient for the smoothness. Defaults to 50.
        offset (float, optional): Minimize bias. Defaults to 0.001.

    Returns:
        dolfin_adjoint.Form: Filtered function form.
    '''
    return project((1 + offset*exp(-a*u))/(1 + exp(-a*u)), V)

def isoparametric2Dfilter(z, e):
    '''
    Apply 2D isoparametric projection onto orientation vector.

    Args:
        z (dolfin_adjoint.Function): 0-component of the orientation vector (on natural setting).
        e (dolfin_adjoint.Function): 1-component of the orientation vector (on natural setting)

    Returns:
        dolfin_adjoint.Vector: Orientation vector with unit circle boundary condition on real setting.
    '''    
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