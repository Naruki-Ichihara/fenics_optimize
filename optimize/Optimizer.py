#! /usr/bin/python3
# -*- coding: utf-8 -*-
''' Optimizers.

Todo:
    * Changed interface: constraint to constraints
'''
from dolfin import *
from dolfin_adjoint import *
import nlopt as nl
from cyipopt import minimize_ipopt

def MMAoptimize(problemSize, initial, forward, constraint, maxeval=100, bounds=[-1, 1], rel=1e-8, verbosity=1):
    def eval(x, grad):
        cost, J = forward(x)
        grad[:] = J
        return cost
    def const(x, grad):
        cost, J = constraint(x)
        grad[:] = J
        return cost
    opt = nl.opt(nl.LD_MMA, problemSize)
    opt.set_min_objective(eval)
    opt.add_inequality_constraint(const, 1e-8)
    opt.set_lower_bounds(bounds[0])
    opt.set_upper_bounds(bounds[1])
    opt.set_xtol_rel(rel)
    opt.set_param('verbosity', verbosity)
    opt.set_maxeval(maxeval)
    return opt.optimize(initial)

def HSLoptimize(problemSize, initial, forward, constraint, maxeval=100, bounds=[-1, 1], rel=1e-8, verbosity=5):
    options = {'print_level': verbosity,
               'max_iter': maxeval,
               'tol': rel,
               'hessian_constant': 'yes'}
    constraints = [{
            'type': 'ineq',
            'fun': lambda x: -constraint(x)[0],
            'jac': lambda x: -constraint(x)[1]
    }]
    res = minimize_ipopt(forward,
                         initial,
                         jac=True,
                         bounds=(bounds, )*problemSize,
                         constraints=constraints,
                         options=options)
    return res.x