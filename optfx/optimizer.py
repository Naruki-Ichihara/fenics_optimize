#! /usr/bin/python3
# -*- coding: utf-8 -*-
'''The optimize function computes the size of the optimization problem by summing 
the sizes of the initial conditions, and then creates an instance of the nlopt optimizer using 
the specified algorithm. The function then splits the optimization problem into subproblems 
based on the size of the initial conditions, 
and solves each subproblem separately using the minimize method of the nlopt optimizer.
'''
from dolfin import *
from dolfin_adjoint import *
import numpy as np
from .utils import from_numpy, to_numpy
try:
    import nlopt as nl
except ImportError:
    raise ImportError('Optimizer depends on Nlopt.')

class Optimizer(nl.opt):
    '''
    The Optimizer class is a subclass of the nlopt.opt class.
    It creates an instance of the nlopt optimizer using the specified algorithm.
    '''
    def __init__(self, problem, initials, wrt, algorithm='LD_MMA', *args):
        ''' Creates an instance of the nlopt optimizer using the specified algorithm.

    Available algorithms:
        - LD_MMA (default) Method of moving asymptotes. 
            see https://nlopt.readthedocs.io/en/latest/NLopt_Algorithms/#mma-method-of-moving-asymptotes
        - LD_SLSQP Sequential quadratic programming.
            see https://nlopt.readthedocs.io/en/latest/NLopt_Algorithms/#slsqp-sequential-quadratic-programming
        - LD_LBFGS Limited-memory BFGS.
            see https://nlopt.readthedocs.io/en/latest/NLopt_Algorithms/#lbfgs-limited-memory-bfgs
        - LD_TNEWTON Truncated Newton.
            see https://nlopt.readthedocs.io/en/latest/NLopt_Algorithms/#tnewton-truncated-newton
        - LD_TNEWTON_RESTART Truncated Newton with restarting.
            see https://nlopt.readthedocs.io/en/latest/NLopt_Algorithms/#tnewton-restart-truncated-newton-with-restarting
        - LD_TNEWTON_PRECOND Preconditioned truncated Newton.  
            see https://nlopt.readthedocs.io/en/latest/NLopt_Algorithms/#tnewton-precond-preconditioned-truncated-newton
        - LD_TNEWTON_PRECOND_RESTART Preconditioned truncated Newton with restarting.
            see https://nlopt.readthedocs.io/en/latest/NLopt_Algorithms/#tnewton-precond-restart-preconditioned-truncated-newton-with-restarting
        - LD_VAR1 Variably dimensioned solver.
            see https://nlopt.readthedocs.io/en/latest/NLopt_Algorithms/#var1-variably-dimensioned-solver
        - LD_VAR2 Variably dimensioned solver.
            see https://nlopt.readthedocs.io/en/latest/NLopt_Algorithms/#var2-variably-dimensioned-solver
        - LD_AUGLAG Augmented Lagrangian.
            see https://nlopt.readthedocs.io/en/latest/NLopt_Algorithms/#auglag-augmented-lagrangian
        - LD_AUGLAG_EQ Augmented Lagrangian with equality constraints.
            see https://nlopt.readthedocs.io/en/latest/NLopt_Algorithms/#auglag-eq-augmented-lagrangian-with-equality-constraints

        Args:
            algorithm: The name of the nlopt algorithm to use.
            problem: An instance of the OptimizationProblem class.
            initials: A list of the initial conditions.
            wrt: A dict of the indices of the control variables for which sensitivities need to be computed.
            *args: Additional arguments to be passed to the nlopt optimizer.

        Returns:
            An instance of the nlopt optimizer.
            
        '''
        self.initials = initials
        split_index = []
        index = 0
        for initial in self.initials:
            index += initial.vector().size()
            split_index.append(index)
        problemSize = 0
        for initial in self.initials:
            problemSize += initial.vector().size()
        super().__init__(getattr(nl, algorithm), problemSize, *args)
        self.initial_numpy = np.concatenate([to_numpy(i) for i in initials])

        def eval(x, grad):
            xs = np.split(x, split_index)
            xs_fenics = [from_numpy(i, j) for i, j in zip(xs, initials)]
            cost = problem.forward(xs_fenics)
            grad[:] = np.concatenate(problem.backward())
            return cost

        def generate_cost_function(attribute, problem, wrt):
            def cost(x, grad):
                measure = getattr(problem, attribute)()
                grad[:] = np.concatenate(problem.backward_constraint(attribute, wrt[attribute]))
                return measure
            return cost
    
        for attribute in dir(problem):
            if attribute.startswith('constraint'):
                cost_func = generate_cost_function(attribute, problem, wrt)
                print(type(cost_func))
                self.add_inequality_constraint(cost_func, 1e-8)
        
        self.set_min_objective(eval)
        pass

    def run(self):
        ''' Runs the optimization algorithm and returns the solution as a tuple of fenics functions.

        Returns:
            A tuple of fenics functions.
        
        '''
        split_index = []
        index = 0
        for initial in self.initials:
            index += initial.vector().size()
            split_index.append(index)
        solution_numpy = self.optimize(self.initial_numpy)
        solution_fenics = [from_numpy(i, j) for i, j in zip(np.split(solution_numpy, split_index), self.initials)]
        return tuple(solution_fenics)




def optimize(problem, initials, wrt, setting, params, algorithm='LD_MMA'):
    '''Solves an optimization problem using the specified algorithm.
    
    Available algorithms:
        - LD_MMA (default) Method of moving asymptotes. 
            see https://nlopt.readthedocs.io/en/latest/NLopt_Algorithms/#mma-method-of-moving-asymptotes
        - LD_SLSQP Sequential quadratic programming.
            see https://nlopt.readthedocs.io/en/latest/NLopt_Algorithms/#slsqp-sequential-quadratic-programming
        - LD_LBFGS Limited-memory BFGS.
            see https://nlopt.readthedocs.io/en/latest/NLopt_Algorithms/#lbfgs-limited-memory-bfgs
        - LD_TNEWTON Truncated Newton.
            see https://nlopt.readthedocs.io/en/latest/NLopt_Algorithms/#tnewton-truncated-newton
        - LD_TNEWTON_RESTART Truncated Newton with restarting.
            see https://nlopt.readthedocs.io/en/latest/NLopt_Algorithms/#tnewton-restart-truncated-newton-with-restarting
        - LD_TNEWTON_PRECOND Preconditioned truncated Newton.  
            see https://nlopt.readthedocs.io/en/latest/NLopt_Algorithms/#tnewton-precond-preconditioned-truncated-newton
        - LD_TNEWTON_PRECOND_RESTART Preconditioned truncated Newton with restarting.
            see https://nlopt.readthedocs.io/en/latest/NLopt_Algorithms/#tnewton-precond-restart-preconditioned-truncated-newton-with-restarting
        - LD_VAR1 Variably dimensioned solver.
            see https://nlopt.readthedocs.io/en/latest/NLopt_Algorithms/#var1-variably-dimensioned-solver
        - LD_VAR2 Variably dimensioned solver.
            see https://nlopt.readthedocs.io/en/latest/NLopt_Algorithms/#var2-variably-dimensioned-solver
        - LD_AUGLAG Augmented Lagrangian.
            see https://nlopt.readthedocs.io/en/latest/NLopt_Algorithms/#auglag-augmented-lagrangian
        - LD_AUGLAG_EQ Augmented Lagrangian with equality constraints.
            see https://nlopt.readthedocs.io/en/latest/NLopt_Algorithms/#auglag-eq-augmented-lagrangian-with-equality-constraints

    Args:
        problem: An instance of the OptimizationProblem class.
        initials: A list of the initial conditions.
        wrt: A list of the indices of the control variables for which sensitivities need to be computed.
        setting: A dictionary of settings for the nlopt optimizer.
        params: A dictionary of parameters for the nlopt optimizer.
        algorithm: The name of the nlopt algorithm to use.

    Returns:
        A list of NumPy arrays containing the optimized control variables.
    '''
    problem_size = 0

    for initial in initials:
        problem_size += initial.vector().size()

    optimizer = nl.opt(getattr(nl, algorithm), problem_size)

    split_index = []
    index = 0
    for initial in initials:
        index += initial.vector().size()
        split_index.append(index)

    def eval(x, grad):
        xs = np.split(x, split_index)
        xs_fenics = [from_numpy(i, j) for i, j in zip(xs, initials)]
        cost = problem.forward(xs_fenics)
        grad[:] = np.concatenate(problem.backward())
        return cost

    def generate_cost_function(attribute, problem, wrt):
        def cost(x, grad):
            measure = getattr(problem, attribute)()
            grad[:] = np.concatenate(problem.backward_constraint(attribute, wrt[attribute]))
            return measure
        return cost
    
    for attribute in dir(problem):
        if attribute.startswith('constraint'):
            cost_func = generate_cost_function(attribute, problem, wrt)
            print(type(cost_func))
            optimizer.add_inequality_constraint(cost_func, 1e-8)

    optimizer.set_min_objective(eval)
    for set in setting:
        getattr(optimizer, set)(setting[set])
    for param in params:
        optimizer.set_param(param, params[param])
    initial_numpy = np.concatenate([to_numpy(i) for i in initials])
    solution_numpy = optimizer.optimize(initial_numpy)
    solution_fenics = [from_numpy(i, j) for i, j in zip(np.split(solution_numpy, split_index), initials)]
    return tuple(solution_fenics)
