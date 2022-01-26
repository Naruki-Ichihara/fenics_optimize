#! /usr/bin/python3
# -*- coding: utf-8 -*-
''' Optimizer interfaces of the ipopt and nlopt.
'''
from dolfin import *
from dolfin_adjoint import *
import nlopt as nl
import cyipopt as cp
import numpy as np

def MMAoptimize(problemSize, initial, forward, constraints=None, constraints_targets=None, bounds=[0, 1], maxeval=100, rel=1e-8, verbosity=1):
    '''
    Method of Moving Asymptotes based on Nlopt. Working with Jacobians.

    Args:
        problemSize (int): Problem size.
        initial (np.ndarray): numpy vector for the initial values.
        forward (function): Forward calculation chain with decorator `with_derivative`.
        constraints (list): Inequality constraints list `gs =< constraints_target`.
        constraints_targets (list): Inequality constraint targets list `gs =< constraints_target`.
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
    if constraints is not None:
        for pack in zip(constraints, constraints_targets):
            constraint, constraints_target = pack
            def const(x, grad):
                cost, J = constraint(x)
                grad[:] = J
                return cost - constraints_target
            opt.add_inequality_constraint(const, 1e-8)
    else:
        print('Optimizer is conctructed without any constraints.')
        
    opt.set_min_objective(eval)
    opt.add_inequality_constraint(const, 1e-8)
    opt.set_lower_bounds(bounds[0])
    opt.set_upper_bounds(bounds[1])
    opt.set_xtol_rel(rel)
    opt.set_param('verbosity', verbosity)
    opt.set_maxeval(maxeval)
    return opt.optimize(initial)

def HSLoptimize(problemSize, initial, forward, constraints=None, constraints_targets=None, bounds=[0, 1], maxeval=100, rel=1e-8, verbosity=5, solver_type='ma27'):
    '''
    HSL (ma-XX) by Ipopt optimizer. Working with Jacobians.

    Args:
        problemSize (int): Problem size.
        initial (np.ndarray): numpy vector for the initial values.
        forward (function): Forward calculation chain with decorator `with_derivative`.
        constraints (list): Inequality constraints list `gs =< constraints_target`.
        constraints_targets (list): Inequality constraint targets list `gs =< constraints_target`.
        bounds (list): Bounds for control variables.
        maxeval (int, optional): Number of maximum evaluations. Defaults to 100.
        rel (float, optional): Relative tolerance. Defaults to 1e-8.
        verbosity (int, optional): Defaults to 0.
        solver_type (str, optional): HSL solver type. Default to 'ma27'.

    Returns:
        optimized (np.ndarray): Optimized controls.
    '''
    if constraints is not None:
        class Problem():
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

        cl = [-1e19, -1e19]
        cu = constraints_targets
        m = len(cl)
    else:
        class Problem():
            def objective(self, x):
                return forward(x)[0]
        
            def gradient(self, x):
                return forward(x)[1]
        print('Optimizer is conctructed without any constraints.')
        m = 0
        cl = None
        cu = None



    nlp = cp.Problem(
    n=problemSize,
    m=m,
    problem_obj=Problem(),
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

