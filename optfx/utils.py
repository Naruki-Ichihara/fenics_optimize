#! /usr/bin/python3
# -*- coding: utf-8 -*-
'''Utiltes for optfx.'''
from dolfin import *
import dolfin
from dolfin_adjoint import *
import numpy as np

def to_numpy(fenics_var: Constant | Function) -> np.ndarray:
    """Converting from fenicsx variables to numpy ndarray.
    Args:
        fenics_var (fem.Constant | fem.Function): fenicsx variable
    Raises:
        ValueError
    Returns:
        np.ndarray: Return the numpy.ndarray
    """
    if isinstance(fenics_var, dolfin.Constant):
        return np.asarray(fenics_var.values())

    if isinstance(fenics_var, dolfin.GenericVector):
        if fenics_var.mpi_comm().size > 1:
            data = fenics_var.gather(np.arange(fenics_var.size(), dtype="I"))
        else:
            data = fenics_var.get_local()
        return np.asarray(data)

    if isinstance(fenics_var, dolfin.Function):
        fenics_vec = fenics_var.vector()
        if fenics_vec.mpi_comm().size > 1:
            data = fenics_vec.gather(np.arange(fenics_vec.size(), dtype="I"))
        else:
            data = fenics_vec.get_local()        
        return np.asarray(data)

    raise ValueError("Cannot convert " + str(type(fenics_var)))

def from_numpy(numpy_array: np.ndarray, func_temp: Function) -> Function:
    """Converting from numpy array to fenicsx variables based on the function space.
    Args:
        numpy_array (np.ndarray): Array to convert into fenicsx
        func_space (fem.FunctionSpace): Base-function space
    Raises:
        ValueError
    Returns:
        fem.Function: Return the fenicsx.fem.Function
    """    

    if isinstance(func_temp, dolfin.Constant):
        if numpy_array.shape == (1,):
            return type(func_temp)(numpy_array[0])
        else:
            return type(func_temp)(numpy_array)

    if isinstance(func_temp, dolfin.Function):
        function_space = func_temp.function_space()

        u = type(func_temp)(function_space)

        fenics_size = u.vector().size()
        np_size = numpy_array.size

        if np_size != fenics_size:
            err_msg = ("Cannot convert numpy array to Function: Wrong size {} vs {}".format(np_size, fenics_size))
            raise ValueError(err_msg)

        if numpy_array.dtype != np.float_:
            err_msg = ("The numpy array must be of type {}, but got {}".format(np.float_, numpy_array.dtype))
            raise ValueError(err_msg)


        range_begin, range_end = u.vector().local_range()
        numpy_array = np.asarray(numpy_array)
        local_array = numpy_array.reshape(fenics_size)[range_begin:range_end]
        u.vector().set_local(local_array)
        u.vector().apply("insert")
        return u