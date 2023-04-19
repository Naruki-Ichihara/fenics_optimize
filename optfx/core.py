#! /usr/bin/python3
# -*- coding: utf-8 -*-
''' Abstract class for field optimization problems.
'''
from abc import ABCMeta, abstractmethod
from collections.abc import Iterable
from fenics import *
from fenics_adjoint import *
import numpy as np
from .utils import to_numpy

class Module(metaclass=ABCMeta):
    index = int(0)
    snes_obj = None
    sens_cns = None
    log_obj = [0]
    '''
    Core module of the fenics-optimize. 
    '''     
    def __compute_sensitivities(self, objective, controls, wrt):
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
        '''
        This is abstract method. You must override this method.
        Raises:
            NotImplementedError
        '''        
        raise NotImplementedError('')

    def forward(self, controls):
        '''
        Solving the problem that defined in the `problem` method.
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
        '''
        Compute the sensitivities of the objective value w.r.t. `wrt` indices.
        Args:
            wrt (list, optional): Automatic derivative of objective w.r.t. wrt index. Defaults to None to calculate sensitivities for all controls.
        Returns:
            list: list of numpy array.
        '''        
        sensitivities = self.__compute_sensitivities(self.objective, self.controls_fenics, wrt)
        self.snes_obj = sensitivities
        return sensitivities

    def backward_constraint(self, target, wrt=None):
        '''
        Compute the sensitivities of the constraint value w.r.t. `wrt` indices.
        Args:
            target (str): Name of the target constraint.
            wrt (list, optional): Automatic derivative of objective w.r.t. wrt index. Defaults to None to calculate sensitivities for all controls.
        Returns:
            list: list of numpy array.
        '''
        sensitivities = self.__compute_sensitivities(getattr(self, target)(), self.controls_fenics, wrt)
        self.sens_cns = sensitivities
        return sensitivities
