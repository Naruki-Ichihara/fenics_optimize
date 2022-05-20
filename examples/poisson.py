from dolfin import *
from dolfin_adjoint import *
import numpy as np
import fenics_optimize as op
from fenics_optimize.optimizer import interface_nlopt
# Some flags for FEniCS
parameters["form_compiler"]["optimize"] = True
parameters["form_compiler"]["cpp_optimize"] = True

# Quadrature degree in FEniCS  (sometimes, the "automatic" determination of the quadrature degree becomes excessively high, meaning that it should be manually reduced)
parameters['form_compiler']['quadrature_degree'] = 5


m = 0.3    # Target rate of the material amount
p = 5      # Penalty parameter
eps = 1e-3 # Material lower bound
R = 0.025    # Helmholtz filter radius
n = 256    # Resolution

def k(a):
    return eps + (1 - eps) * a ** p

mesh = UnitSquareMesh(n, n)
X = FunctionSpace(mesh, 'CG', 1)
f = interpolate(Constant(1e-2), X)

class Initial_density(UserExpression):
    def eval(self, value, x):
        value[0] = m

class Left(SubDomain):
    def inside(self, x, on_boundary):
        gamma = 1/n + eps
        return x[0] == 0.0 and 0.5 - gamma < x[1] < 0.5 + gamma and on_boundary

U = FunctionSpace(mesh, 'CG', 1)
t = TrialFunction(U)
dt = TestFunction(U)
bc = DirichletBC(U, Constant(0.0), Left())

class PoissonProblem(op.Module):
    def problem(self, controls):
        rho = controls[0]
        rho = op.helmholtzFilter(controls[0], X, R=R)
        a = inner(grad(t), k(rho)*grad(dt))*dx
        L = f*dt*dx
        A, b = assemble_system(a, L, bc)
        T = Function(U, name='Temperture')
        T = op.amg(A, b, T, False)
        J = assemble(inner(grad(T), k(rho)*grad(T))*dx)
        rho_bulk = project(Constant(1.0), X)
        rho_0 = assemble(rho_bulk*dx)
        rho_total = assemble(controls[0]*dx)
        rel = rho_total/rho_0
        self.volumeFraction = rel
        return J

    def constraint_volume(self):
        return self.volumeFraction - m
x0 = Function(X)
x0.interpolate(Initial_density())
N = Function(X).vector().size()
min_bounds = np.zeros(N)
max_bounds = np.ones(N)

setting = {'set_lower_bounds': min_bounds,
           'set_upper_bounds': max_bounds,
           'set_maxeval': 100
          }
params = {'verbosity': 5}

problem = PoissonProblem()
solution = interface_nlopt(problem, [x0], [0], setting, params)
file = File('/workspace/examples/results/test.pvd')
file << solution[0]