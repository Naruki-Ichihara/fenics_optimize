#! /usr/bin/python3
# -*- coding: utf-8 -*-
''' Optimizer interfaces of the ipopt and nlopt.
'''
from dolfin import *
from dolfin_adjoint import *
import nlopt as nl
import cyipopt as cp
from cyipopt import minimize_ipopt
import numpy as np

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

def _HSLoptimize(problemSize, initial, forward, constraints, bounds, maxeval=100, rel=1e-8, verbosity=5):
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
               'tol': rel,
               'linear_solver': 'ma97'}
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
                         hess=None,
                         bounds=bounds_assy,
                         constraints=constraints_dict,
                         options=options)
    return res.x

def HSLoptimize(problemSize, initial, forward, constraints, bounds, maxeval=100, rel=1e-8, verbosity=5, solver_type='ma27'):
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
        solver_type (str, optional): HSL solver type. Default to 'ma27'.

    Returns:
        optimized (np.ndarray): Optimized controls.
    '''
    class HS():
        def objective(self, x):
            return forward(x)[0]
        
        def gradient(self, x):
            return forward(x)[1]

        def constraints(self, x):
            constraints_list = []
            for constraint in constraints:
                constraints_list.append(constraint(x)[0])
            return np.array(constraints_list)

        def jacobian(self, x):
            constraints_list = []
            for constraint in constraints:
                constraints_list.append(constraint(x)[1])
            return np.array(constraints_list)

    cl = [-1e19]
    cu = np.zeros(len(constraints))

    nlp = cp.Problem(
    n=problemSize,
    m=len(cl),
    problem_obj=HS(),
    lb=bounds[0],
    ub=bounds[1],
    cl=cl,
    cu=cu,
    )

    nlp.add_option('linear_solver', solver_type)
    nlp.add_option('max_iter', maxeval)
    nlp.add_option('tol', rel)
    nlp.add_option('print_level', verbosity)

    x, info = nlp.solve(initial)
    print(info)
    return x

