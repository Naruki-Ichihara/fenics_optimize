#! /usr/bin/python3
# -*- coding: utf-8 -*-
''' Optimizer interfaces for nlopt.
'''
from dolfin import *
from dolfin_adjoint import *
import numpy as np
try:
    import nlopt as nl
except ImportError:
    raise ImportError('Optimizer depends on Nlopt.')

def interface_nlopt(problem, initial, templates, constraints, wrt, setting, params, algorithm='LD_MMA'):
    problem_size = 0
    for template in templates:
        problem_size += Function(template).vector().size()
    optimizer = nl.opt(getattr(nl, algorithm), problem_size)
    def eval(x, grad):
        xs = np.split(x, len(templates))
        cost = problem.forward(xs, templates)
        grad[:] = np.concatenate(problem.backward())
        return cost
    if constraints is not None:
        for constraint in constraints:
            def const(x, grad):
                measure = getattr(problem, constraint)()
                grad[:] = np.concatenate(problem.backward_constraint(constraint, wrt))
                return measure
            optimizer.add_inequality_constraint(const, 1e-8)
    optimizer.set_min_objective(eval)
    for set in setting:
        getattr(optimizer, set)(setting[set])
    for param in params:
        optimizer.set_param(param, params[param])
    solution = optimizer.optimize(initial)
    return solution