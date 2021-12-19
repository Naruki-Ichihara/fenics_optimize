from dolfin import *
from dolfin_adjoint import *
from fenics_adjoint.blocks.common import FunctionAssignBlock
from numpy.lib.shape_base import column_stack
from Morphogenesis.Solvers.AMGsolver import AMGsolver
from Morphogenesis.Utils.file_io import export_result
from Morphogenesis.Optimizer.Utils import ElasticityProblem
from fecr import from_numpy, to_numpy
import numpy as np
import sys, os
import nlopt as nl

# Elasticity parameters
E = 1.0e9
nu = 0.3

problem = ElasticityProblem(E, nu)

mesh = BoxMesh(
    MPI.comm_world, Point(0.0, 0.0, 0.0),
                     Point(20.0, 10.0, 10.0), 40, 20, 20)

N = mesh.num_vertices()
x0 = np.zeros(N)

X = FunctionSpace(mesh, "CG", 1)
V = VectorFunctionSpace(mesh, "Lagrange", 1)

class Bottom(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and x[2] < 1e-10

def clamped_boundary(x, on_boundary):
    return on_boundary and x[0] < 1e-10 or x[0] > 19.999

def evaluator(x, grad):
    x_ = from_numpy(x, Function(X))
    control = Control(x_)
    rho = problem.hevisideFilter(x_, X)
    export_result(project(rho, FunctionSpace(mesh, 'DG', 0)), 'test.xdmf')
    facets = MeshFunction('size_t', mesh, 1)
    facets.set_all(0)
    bottom = Bottom()
    bottom.mark(facets, 1)
    ds = Measure('ds', subdomain_data=facets)
    f = Constant((0, 0, -1e3))
    u = TrialFunction(V)
    v = TestFunction(V)
    a = rho**3*inner(problem.sigma(u), problem.epsilon(v))*dx
    L = inner(f, v)*ds(1)
    bc = DirichletBC(V, Constant((0, 0, 0)), clamped_boundary)
    u_ = Function(V)
    solver = AMGsolver(a, L, [bc])
    uh = solver.forwardSolve(u_, V, False)
    J = assemble(rho**3*inner(problem.sigma(uh), problem.epsilon(uh))*dx)
    dJdx = to_numpy(compute_gradient(J, control))
    grad[:] = dJdx
    return J

def volumeResponce(x, grad):
    x_ = from_numpy(x, Function(X))
    control = Control(x_)
    rho_bulk = project(Constant(1.0), FunctionSpace(mesh, 'CG', 1))
    rho_0 = assemble(rho_bulk*dx)
    rho_f = assemble(problem.hevisideFilter(x_, X)*dx)
    rel = rho_f/rho_0
    dreldx = to_numpy(compute_gradient(rel, control))
    grad[:] = dreldx
    target = 0.5
    return rel-target

topopt = nl.opt(nl.LD_MMA, N)
topopt.set_min_objective(evaluator)
topopt.add_inequality_constraint(volumeResponce, 1e-8)
topopt.set_lower_bounds(-1.0)
topopt.set_upper_bounds(1.0)
topopt.set_xtol_rel(1e-10)
topopt.set_param('verbosity', 1)
topopt.set_maxeval(100)
x = topopt.optimize(x0)