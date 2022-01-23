from dolfin import *
from dolfin_adjoint import *
import fenics_optimize as op
from fenics_optimize.Mechanics import sigma, epsilon
import numpy as np
from mpi4py import MPI
comm = MPI.COMM_WORLD

rec_sig = op.Recorder('result/stress_constraint', 'sigma')
rec_rho = op.Recorder('result/stress_constraint', 'material')

E = 1.0e9
nu = 0.3
f = Constant((0, -1e3))
p = 3
eps = 1.0e-3
target = 0.5
R = 0.1
EPS = 1e-8

def k(rho):
    return eps + (1 - eps) * rho ** p

def k0(rho):
    return eps + (1 - eps) * rho

def mises(u):
    s = dev(sigma(u, E, nu))
    return sqrt(3/2*inner(s, s))

mesh = Mesh(comm, '/workspace/mesh/notch.xml')
X = FunctionSpace(mesh, "CG", 1)
Xs = [X]
V = VectorFunctionSpace(mesh, "CG", 1)

problemSize = op.catch_problemSize(Xs)

class Upper(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and x[1] > 50 - EPS and -eps < x[0] < 10.0

class Bottom(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and x[1] < -50 + EPS and -eps < x[0] < 10.0

facets = MeshFunction('size_t', mesh, 1)
facets.set_all(0)
upper = Upper()
upper.mark(facets, 1)
ds = Measure('ds', subdomain_data=facets)

bc = DirichletBC(V, Constant((0, 0)), Bottom())

u = TrialFunction(V)
v = TestFunction(V)
uh0 = Function(V)

a0 = inner(sigma(u, E, nu), epsilon(v))*dx
L0 = inner(f, v)*ds(1)
A0, b0 = assemble_system(a0, L0, [bc])
uh0 = op.AMGsolver(A0, b0).solve(uh0, V, monitor_convergence=False, build_null_space='2-D')
e0 = assemble(inner(sigma(uh0, E, nu), epsilon(uh0))*dx)

@op.with_minmax_derivative(Xs, None, 'P-norm', P=6)
def stress_constraint(xs):
    rho = op.helmholtzFilter(xs[0], X, R)
    a = inner(k(rho)*sigma(u, E, nu), epsilon(v))*dx
    L = inner(f, v)*ds(1)
    A, b = assemble_system(a, L, [bc])
    solver = op.AMGsolver(A, b)
    uh = Function(V)
    uh = solver.solve(uh, V, False, '2-D')
    sig_m = project(mises(uh), X)
    rec_sig.rec(sig_m)
    return sig_m

@op.with_derivative(Xs)
def forward(xs):
    rho = op.helmholtzFilter(xs[0], X, R)
    rho_bulk = project(Constant(1.0), X)
    rho_0 = assemble(rho_bulk*dx)
    rho_f = assemble(rho*dx)
    rec_rho.rec(rho)
    a = inner(k(rho)*sigma(u, E, nu), epsilon(v))*dx
    L = inner(f, v)*ds(1)
    A, b = assemble_system(a, L, [bc])
    solver = op.AMGsolver(A, b)
    uh = Function(V)
    uh = solver.solve(uh, V, False, '2-D')
    e = assemble(inner(k0(rho)*sigma(uh, E, nu), epsilon(uh0))*dx)
    theta = 0.5
    cost = theta*(rho_f/rho_0) + (1-theta)*(e/e0)
    return cost

x0 = np.ones(problemSize)*0.5
x_min = np.zeros(problemSize)
x_max = np.ones(problemSize)

op.HSLoptimize(problemSize, x0, forward, [stress_constraint], [5000], maxeval=1000, bounds=[x_min, x_max], rel=1e-20, verbosity=5, solver_type='ma86')