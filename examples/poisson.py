from dolfin import *
from dolfin_adjoint import *
import fenics_optimize as op
import numpy as np

V = 0.3  # volume bound on the control
p = 5  # power used in the solid isotropic material
eps = Constant(1.0e-3)  # epsilon used in the solid isotropic material
r = 0.01

def k(rho):
    return eps + (1 - eps) * rho ** p

n = 256
mesh = UnitSquareMesh(n, n)
problemSize = mesh.num_vertices()
x0 = np.ones(problemSize)*V

X = FunctionSpace(mesh, 'CG', 1)

class Left(SubDomain):
    def inside(self, x, on_boundary):
        gamma = 1/250 + 1e-5
        return x[0] == 0.0 and 0.5 - gamma < x[1] < 0.5 + gamma and on_boundary

bcs = [DirichletBC(X, Constant(0.0), Left())]
f = interpolate(Constant(1e-2), X)
file = File('result/poisson/material.pvd')

@op.with_derivative([X])
def forward(xs):
    rho = op.helmholtzFilter(xs[0], X, R=r)
    rho.rename('label', 'material')
    Th = Function(X, name='Temperature')
    u = TrialFunction(X)
    v = TestFunction(X)
    a = inner(grad(u), k(rho)*grad(v))*dx
    L = f*v*dx
    A, b = assemble_system(a, L, bcs)
    Th = op.AMGsolver(A, b).solve(Th, X, False)
    J = assemble(inner(grad(Th), k(rho)*grad(Th))*dx)
    file << rho
    return J

@op.with_derivative([X])
def constraint(xs):
    rho_bulk = project(Constant(1.0), X)
    rho_0 = assemble(rho_bulk*dx)
    rho_f = assemble(xs[0]*dx)
    rel = rho_f/rho_0
    return rel - V

op.MMAoptimize(problemSize, x0, forward, constraint, maxeval=1000, bounds=[0, 1], rel=1e-20)