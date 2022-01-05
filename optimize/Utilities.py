#! /usr/bin/python3
# -*- coding: utf-8 -*-
''' Solvers.
[detail]
'''

from dolfin import *
from dolfin_adjoint import *
from .fecr import from_numpy, to_numpy

def evalGradient(target, control):
    """# Utilities
    ## evalGradient

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
    """# Utilities
    ## numpy2fenics
    Convert from ndarray to fenics function w.r.t. function space U.

    Args:
        u (ndarray): 
        U (fenics.FunctionSpace):
        
    Return:
        u (fenics.function):
    """
    return from_numpy(u, Function(U))

def fenics2numpy(u):
    """# Utilities
    ## fennics2numpy
    Convert from fenics function to ndarray.
    Args:
        u (finics.function): 
    Return:
        u (ndarray)
    """
    return to_numpy(u)

def export_result(u, filepath):
    """# Utilities
    ## export_result
    Export static result as the xdmf format.

    ### Args:
        u : function
        filepath : file name

    ### Examples:

    ### Note:

    """
    file = XDMFFile(filepath)
    file.write(u)
    pass

def read_xdmf(filepath, comm):
    """# Utilities
    ## read_xdmf
    Read mesh with the MPI comminication world.

    ### Args:
        filepath : file name

    ### Examples:

    ### Note:

    """
    mesh = Mesh()
    XDMFFile(comm, filepath).read(mesh)
    return mesh