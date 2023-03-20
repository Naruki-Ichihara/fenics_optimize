import numpy as np
import optfx as opt

opt.parameters["form_compiler"]["optimize"] = True
opt.parameters["form_compiler"]["cpp_optimize"] = True
opt.parameters['form_compiler']['quadrature_degree'] = 5

comm = opt.MPI.comm_world
m = 0.30   # Target rate of the material amount
p = 5      # Penalty parameter
eps = 1e-3 # Material lower bound
R = 0.01   # Helmholtz filter radius
n = 256    # Resolution

def k(a):
    return eps + (1 - eps) * a ** p

mesh = opt.UnitSquareMesh(comm, n, n)
X = opt.FunctionSpace(mesh, 'CG', 1)
f = opt.interpolate(opt.Constant(1e-2), X)

class Initial_density(opt.UserExpression):
    def eval(self, value, x):
        value[0] = 0.4

class Left(opt.SubDomain):
    def inside(self, x, on_boundary):
        gamma = 1/250 + 1e-5
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
        Th = opt.Function(U)
        opt.solve(a==L, Th, bc)
        J = opt.assemble(opt.inner(opt.grad(Th), k(rho)*opt.grad(Th))*opt.dx)
        return J
    def constraint_volume(self, controls):
        rho = controls[0]
        rho = opt.helmholtzFilter(rho, X, R)
        rho_target = opt.project(opt.Constant(m), X)
        cost = opt.assemble((rho - rho_target)*opt.dx)
        return cost

x0 = opt.Function(X)
x0.interpolate(Initial_density())
N = opt.Function(X).vector().size()
min_bounds = np.zeros(N)
max_bounds = np.ones(N)

setting = {'set_lower_bounds': min_bounds,
           'set_upper_bounds': max_bounds,
           'set_maxeval': 50
          }
params = {'verbosity': 1}

problem = PoissonProblem()
solution = opt.optimize(problem, [x0], [0], setting, params)