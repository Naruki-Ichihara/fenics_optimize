from fenics import *
from fenics_adjoint import *
import numpy as np
import fenics_optimize as opt
import nlopt as nl
from fecr import from_numpy
import math

m = 0.3    # Target rate of the material amount
p = 5      # Penalty parameter
eps = 1e-3 # Material lower bound
R = 0.1    # Helmholtz filter radius
n = 256    # Resolution

def k(a):
    return eps + (1 - eps) * a ** p

class Left(SubDomain):
    def inside(self, x, on_boundary):
        gamma = 1/n + eps
        return x[0] == 0.0 and 0.5 - gamma < x[1] < 0.5 + gamma and on_boundary

mesh = UnitSquareMesh(n, n)
X = FunctionSpace(mesh, 'CG', 1)
U = FunctionSpace(mesh, 'CG', 1)
f = interpolate(Constant(1e-2), U)
T = Function(U, name='Temperature')
t = TrialFunction(U)
dt = TestFunction(U)
bc = DirichletBC(U, Constant(0.0), Left())

class Module(opt.Module):
    def problem(self, controls):
        rho = opt.helmholtzFilter(controls[0], U, R=R)
        rho.rename('label', 'control')
        a = inner(grad(t), k(rho)*grad(dt))*dx
        L = f*dt*dx
        A, b = assemble_system(a, L, bc)
        T_s = opt.amg(A, b, T)
        J = assemble(inner(grad(T_s), k(rho)*grad(T_s))*dx)
        rho_bulk = project(Constant(1.0), U)
        rho_0 = assemble(rho_bulk*dx)
        rho_f = assemble(controls[0]*dx)
        self.volumeFraction = rho_f/rho_0
        return J

    def volumeConstraint(self):
        return self.volumeFraction - m

problemSize = Function(X).vector().size()
x0 = np.ones(problemSize) * m

optimizer = nl.opt(nl.LD_MMA, problemSize)
testProblem = Module()

def eval(x, grad):
    cost = testProblem.forward([x], [X])
    grad[:] = np.concatenate(testProblem.backward())
    return cost

def const(x, grad):
    measure = testProblem.volumeConstraint()
    grad[:] = np.concatenate(testProblem.backward_constraint('volumeConstraint'))
    return measure

optimizer.set_min_objective(eval)
optimizer.add_inequality_constraint(const, 1e-5)
optimizer.set_lower_bounds(0)
optimizer.set_upper_bounds(1)
optimizer.set_maxeval(10)
optimizer.set_param('verbosity', 5)

def test_optimize():
    res = optimizer.optimize(x0)
    val = testProblem.problem([from_numpy(res, Function(X))])
    print(val)
    assert math.isclose(val, 0.004701790705305114, abs_tol=1e-5)

