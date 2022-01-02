#! /usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (C) 2021 Naruki Ichihara
#
# This file is part of Morphogenesis
#
# SPDX-License-Identifier:    MIT
"""MMAoptimizer"""
from dolfin import *
from dolfin_adjoint import *
import nlopt as nl

def MMAoptimize(problemSize, initial, forward, constraint, maxeval=100, bounds=[-1, 1], rel=1e-8, verbosity=0):
    def eval(x, grad):
        cost, J = forward(x)
        grad[:] = J
        return cost

    def const(x, grad):
        cost, J = constraint(x)
        grad[:] = J
        return cost
    topopt = nl.opt(nl.LD_MMA, problemSize)
    topopt.set_min_objective(eval)
    topopt.add_inequality_constraint(const, 1e-8)
    topopt.set_lower_bounds(bounds[0])
    topopt.set_upper_bounds(bounds[1])
    topopt.set_xtol_rel(rel)
    topopt.set_param('verbosity', verbosity)
    topopt.set_maxeval(maxeval)
    return topopt.optimize(initial)