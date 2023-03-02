import numpy as np
import optfx as opt
# Some flags for FEniCS
opt.parameters["form_compiler"]["optimize"] = True
opt.parameters["form_compiler"]["cpp_optimize"] = True

# Quadrature degree in FEniCS  (sometimes, the "automatic" determination of the quadrature degree becomes excessively high, meaning that it should be manually reduced)
opt.parameters['form_compiler']['quadrature_degree'] = 5

comm = opt.MPI.comm_world
m = 0.10    # Target rate of the material amount
p = 5      # Penalty parameter
eps = 1e-4 # Material lower bound
R = 0.1   # Helmholtz filter radius
n = 256    # Resolution

def k(a):
    return eps + (1 - eps) * a ** p

mesh = opt.UnitSquareMesh(comm, n, n)
X = opt.FunctionSpace(mesh, 'CG', 1)
f = opt.interpolate(opt.Constant(1e-2), X)

class Initial_density(opt.UserExpression):
    def eval(self, value, x):
        value[0] = m

class Left(opt.SubDomain):
    def inside(self, x, on_boundary):
        gamma = 1/n + eps
        return x[0] == 0.0 and 0.5 - gamma < x[1] < 0.5 + gamma and on_boundary

U = opt.FunctionSpace(mesh, 'CG', 1)
t = opt.TrialFunction(U)
dt = opt.TestFunction(U)
bc = opt.DirichletBC(U, opt.Constant(0.0), Left())

class PoissonProblem(opt.Module):
    def problem(self, controls):
        rho = controls[0]
        rho = opt.helmholtzFilter(rho, X, R)
        a = opt.inner(opt.grad(t), k(rho)*opt.grad(dt))*opt.dx
        L = opt.inner(f, dt)*opt.dx
        Th = opt.Function(U, name='Temperture')
        opt.solve(a==L, Th, bc)
        J = opt.assemble(opt.inner(opt.grad(Th), k(rho)*opt.grad(Th))*opt.dx)
        rho_bulk = opt.project(opt.Constant(1.0), X)
        rho_0 = opt.assemble(rho_bulk*opt.dx)
        rho_total = opt.assemble(controls[0]*opt.dx)
        rel = rho_total/rho_0
        self.volumeFraction = rel
        return J

    def constraint_volume(self):
        return self.volumeFraction - m
x0 = opt.Function(X)
x0.interpolate(Initial_density())
N = opt.Function(X).vector().size()
min_bounds = np.zeros(N)
max_bounds = np.ones(N)

setting = {'set_lower_bounds': min_bounds,
           'set_upper_bounds': max_bounds,
           'set_maxeval': 200
          }
params = {'verbosity': 1}

problem = PoissonProblem()
solution = opt.optimize(problem, [x0], [0], setting, params)
file = opt.File('/workspace/examples/results/test.pvd')
file << solution[0]