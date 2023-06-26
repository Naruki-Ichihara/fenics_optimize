#! /usr/bin/python3
# -*- coding: utf-8 -*-
''' The code is a Python module that defines an abstract class called Module.
This class is the core of an optimization problem and defines the basic methods and attributes
that are required for optimization. The Module class is defined using the ABCMeta metaclass, 
which makes it an abstract class that cannot be instantiated directly.

The Module class is inherited by the OptimizationProblem class, which is the main class of the
fenics-optimize package. The OptimizationProblem class is used to define an optimization problem
and to solve it using the optimize function. 
The problem method is overloaded in the OptimizationProblem class to define the objective function.

The objective argument is the objective function that is being optimized,
the controls argument is a list of the control variables, and the wrt argument is a list of the indices
of the controls for which sensitivities need to be computed.

If the wrt argument is an iterable, the method computes the sensitivities for each control variable 
whose index is in the wrt list. For each control variable, 
the method creates a Control object and computes the gradient of the objective function with respect to 
that control variable using the compute_gradient function from the fenics_adjoint module. 
The sensitivities are then converted to a NumPy array using the to_numpy function from the utils module.

'''
from abc import ABCMeta, abstractmethod
from collections.abc import Iterable
from fenics import *
from fenics_adjoint import *
import numpy as np
from .utils import to_numpy

class Module(metaclass=ABCMeta):
    index = int(0)
    log_obj = [0]
    '''
    Core module of the fenics-optimize. The Module class is an abstract class that defines 
    the basic methods and attributes of the optimization problem.
    '''     
    def __compute_sensitivities(self, objective, controls, wrt) -> list[np.ndarray]:
        '''Computes the sensitivities of the optimization problem with respect to the control variables.

        Args:
            objective: The objective function being optimized.
            controls: A list of the control variables.
            wrt: Either an integer index or a list of integer indices indicating which control variables to compute sensitivities for.

        Returns:
            A list of NumPy arrays containing the sensitivities of the objective function with respect to the specified control variables.
        '''
        if isinstance(wrt, Iterable):
            sensitivities_numpy = []
            for i in range(len(controls)):
                if i not in wrt:
                    sensitivities_numpy.append(to_numpy(controls[i])*0.0)
                else:
                    control = Control(controls[i])
                    sensitivities_numpy.append(to_numpy(compute_gradient(objective, control)))
            return sensitivities_numpy
        elif wrt is None:
            controls_fenics = [Control(i) for i in controls]
        sensitivities_numpy = [to_numpy(compute_gradient(objective, i)) for i in controls_fenics]
        return sensitivities_numpy

    @abstractmethod
    def problem(self, controls):
        ''' Define the optimization problem. This method should be overloaded in the child class.
        Args:
            controls (list): list of fenics.Functions that will be used as control.
            templates (list): list of function spaces of the controls.
        Returns:
            AdjFroat: objective float value.
        '''
        raise NotImplementedError('problem method is not implemented.')

    def forward(self, controls):
        ''' Compute the objective value based on the problem method.
        Args:
            controls (list): list of fenics.Functions that will be used as control.
            templates (list): list of function spaces of the controls.
        Returns:
            AdjFroat: objective float value.
        '''        
        self.controls_fenics = controls
        self.objective = self.problem(self.controls_fenics)
        self.log_obj.append(self.objective)
        self.index += 1
        return self.objective

    def backward(self, wrt=None):
        ''' Compute the sensitivities of the objective value w.r.t. `wrt` indices.
        Args:
            wrt (list, optional): Automatic derivative of objective w.r.t. wrt index. Defaults to None to calculate sensitivities for all controls.
        Returns:
            list: list of numpy array.
        '''        
        sensitivities = self.__compute_sensitivities(self.objective, self.controls_fenics, wrt)
        return sensitivities

    def backward_constraint(self, target, wrt=None):
        ''' Compute the sensitivities of the constraint value w.r.t. `wrt` indices.
        Args:
            target (str): name of the constraint function.
            wrt (list, optional): Automatic derivative of objective w.r.t. wrt index. Defaults to None to calculate sensitivities for all controls.
        Returns:
            list: list of numpy array.
        '''
        sensitivities = self.__compute_sensitivities(getattr(self, target)(), self.controls_fenics, wrt)
        return sensitivities
