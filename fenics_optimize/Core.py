#! /usr/bin/python3
# -*- coding: utf-8 -*-
''' Abstract class for field optimization problems.
'''
from abc import ABCMeta, abstractmethod
from collections.abc import Iterable
from fenics import *
from fenics_adjoint import *
import numpy as np
from fecr import from_numpy, to_numpy

class Module(metaclass=ABCMeta):
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
        elif np.isscalar(wrt):
            controls_fenics = [Control(controls[wrt])]
        sensitivities_numpy = [to_numpy(compute_gradient(objective, i)) for i in controls_fenics]
        return sensitivities_numpy

    @abstractmethod
    def problem(self, *args):
        '''
        This is abstract method. You must override this method.

        Raises:
            NotImplementedError
        '''        
        raise NotImplementedError('')

    def forward(self, controls, templates):
        '''
        Solving the problem that defined in the `problem` method.

        Args:
            controls (list): list of numpy.arrays that will be used as control.
            templates (list): list of function spaces of the controls.

        Returns:
            AdjFroat: objective float value.
        '''        
        templates_function = [Function(i) for i in templates]
        self.controls_fenics = [from_numpy(i, 