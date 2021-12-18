#! /usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (C) 2021 Naruki Ichihara
#
# This file is part of Morphogenesis
#
# SPDX-License-Identifier:    MIT
"""Utils"""

from dolfin import *
from dolfin_adjoint import *
import numpy as np

class ElasticityProblem():
    def __init__(self, E, nu):
        self.mu = E/(2.0*(1.0 + nu))
        self.lmbda = E*nu/((1.0 + nu)*(1.0 - 2.0*nu))

    def helmholtzFilter(self, u, U, R=0.025):
        """# helmholtz_filter
        Apply the helmholtz filter.
        Args:
            u (fenics.function): target function
            U (fenics.FunctionSpace): Functionspace of target function
            r (float): filter radius
        Return:
            v (fenics.function): filtered function
        """
        v = TrialFunction(U)
        dv = TestFunction(U)
        vh = Function(U)
        a = R*inner(grad(v), grad(dv))*dx + dot(v, dv)*dx
        L = inner(u, dv)*dx
        solve(a==L, vh)
        return project(vh, U)

    def heviside_filter(self, u, U, a=50):
        """# heviside_filter

        Apply the heviside function (approximate with sigmoid function)
        Args:
            u (fenics.function): target function
            U (fenics.FunctionSpace): Functionspace of target function
            a (float): coefficient. a>50 -> Step. a=3 -> S-shape
        Returns:
            v (fenics.function): filterd function
        Note:

        """
        return project(1 / (1 + exp(-a*u)), U)

    def sigma(self, v):
        return 2.0*self.mu*sym(grad(v)) + self.lmbda*tr(sym(grad(v)))*Identity(len(v))

    def epsilon(self, v):
        return sym(grad(v))

    def eval_cost(self, uh):
        """# Optimizer
        ## eval_cost
        Evaluation cost, which is the total strain energy.

        ### Args:
            uh : solved displacement.

        ### Examples:

        ### Note:

        """
        cost = assemble(inner(self.sigma(uh), self.epsilon(uh))*dx)
        return cost
