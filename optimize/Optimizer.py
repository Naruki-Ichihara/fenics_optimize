#! /usr/bin/python3
# -*- coding: utf-8 -*-
''' Optimizer interfaces of the ipopt and nlopt.
'''
from dolfin import *
from dolfin_adjoint import *
import nlopt as nl
from cyipopt import minimize_ipopt

def MMAoptimize(problemSize, initial, forward, constraints, bounds, maxeval=100, rel=1e-8, verbosity=1):
    '''
    Method of Moving Asymptotes based on Nlopt. Working with Jacobians.

    Args:
        problemSize (int): Problem size.
        initial (np.ndarray): numpy vector for the initial values.
        forward (function): Forward calculation chain with decorator `with_derivative`.
        constraints (list): Inequality constraints list `gs =< 0`.
        bounds (list): Bounds for control variables.
        maxeval (int, optional): Number of maximum evaluations. Defaults to 100.
        rel (float, optional): Relative tolerance. Defaults to 1e-8.
        verbosity (int, optional): Defaults to 0.

    Returns:
        optimized (np.ndarray): Optimized controls.
    '''    
    opt = nl.opt(nl.LD_MMA, problemSize)
    def eval(x, grad):
        cost, J = forward(x)
        grad[:] = J
        return cost
    for constraint in constraints:
        def const(x, grad):
            cost, J = constraint(x)
            grad[:] = J
            return cost
        opt.add_inequality_constraint(const, 1e-8)
    opt.set_min_objective(eval)
    opt.add_inequality_constraint(const, 1e-8)
    opt.set_lower_bounds(bounds[0])
    opt.set_upper_bounds(bounds[1])
    opt.set_xtol_rel(rel)
    opt.set_param('verbosity', verbosity)
    opt.set_maxeval(maxeval)
    return opt.optimize(initial)

def HSLoptimize(problemSize, initial, forward, constraints, bounds, maxeval=100, rel=1e-8, verbosity=5):
    '''
    HSL (ma-XX) by Ipopt optimizer. Working with Jacobians.

    Args:
        problemSize (int): Problem size.
        initial (np.ndarray): numpy vector for the initial values.
        forward (function): Forward calculation chain with decorator `with_derivative`.
        constraints (list): Inequality constraints list `gs =< 0`.
        bounds (list): Bounds for control variables.
        maxeval (int, optional): Number of maximum evaluations. Defaults to 100.
        rel (float, optional): Relative tolerance. Defaults to 1e-8.
        verbosity (int, optional): Defaults to 0.

    Returns:
        optimized (np.ndarray): Optimized controls.
    '''    
    options = {'print_level': verbosity,
               'max_iter': maxeval,
               'tol': rel}
    constraints_dict = []
    for constraint in constraints:
        elem = {
            'type': 'ineq',
            'fun': lambda x: -constraint(x)[0],
            'jac': lambda x: -constraint(x)[1]
        }
        constraints_dict.append(elem)

    bounds_assy = []
    for i in range(len(bounds[0])):
        bounds_assy.append((bounds[0][i], bounds[1][i]))

    res = minimize_ipopt(forward,
                         initial,
                         jac=True,
                         bounds=bounds_assy,
                         constraints=constraints_dict,
                         options=options)
    return res.x