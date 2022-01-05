from dolfin import *
from dolfin_adjoint import *
import optimize as op
from optimize.Elasticity import sigma, epsilon
import numpy as np

E = 1.0e9
nu = 0.3
f = Constant((0, -1e3))
p = 3
eps = 1.0e-3
target = 0.4
R = 0.1

def k(rho):
    return eps + (1 - eps) * rho ** p

mesh = RectangleMesh(Point(0, 0), Point(20, 10), 200, 100)
problemSize = mesh.num_vertices()
x0 = np.ones(problemSize)*target

X = FunctionSpace(mesh, "CG", 1)
Xs = [X]
V = VectorFunctionSpace(mesh, "CG", 1)

class Bottom(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and x[1] < 1e-10

def clamped_boundary(x, on_boundary):
    return on_boundary and x[0] < 1e-10 or x[0] > 19.999

file = File('result/elasticity/material.pvd')

@op.with_derivative(Xs)
def forward(xs):
    rho = op.helmholtzFilter(xs[0], X, R)
    rho.rename('label', 'material')
    facets = MeshFunction('size_t', mesh, 1)
    facets.set_all(0)
    bottom = Bottom()
    bottom.mark(facets, 1)
    ds = Measure('ds', subdomain_data=facets)
    u = TrialFunction(V)
    v = TestFunction(V)
    a = inner(k(rho)*sigma(u, E, nu), epsilon(v))*dx
    L = inner(f, v)*ds(1)
    bc = DirichletBC(V, Constant((0, 0)), clamped_boundary)
    uh = Function(V)
    A, b = assemble_system(a, L, [bc])
    uh = op.AMG2Dsolver(A, b).solve(uh, V, False)
    file << rho
    return assemble(inner(k(rho)*sigma(uh, E, nu), epsilon(uh))*dx)

@op.with_derivative(Xs)
def constraint(xs):
    rho_bulk = project(Constant(1.0), X)
    rho_0 = assemble(rho_bulk*dx)
    rho_f = assemble(xs[0]*dx)
    rel = rho_f/rho_0
    return rel - target

op.MMAoptimize(problemSize, x0, forward, constraint, maxeval=1000, bounds=[0, 1], rel=1e-20)