'''
Example code for poisson problem.
Naruki Ichihara (2023)
'''
import numpy as np
import optfx as of

of.parameters["form_compiler"]["optimize"] = True
of.parameters["form_compiler"]["cpp_optimize"] = True
of.parameters['form_compiler']['quadrature_degree'] = 5

dir = './examples/results/poisson/'
file_rho = of.XDMFFile(dir + 'rho.xdmf')

comm = of.MPI.comm_world
m = 0.30   # Target rate of the material amount
p = 5      # Penalty parameter
eps = 1e-3 # Material lower bound
R = 0.01   # Helmholtz filter radius
n = 256    # Resolution

def k(a):
    return eps + (1 - eps) * a ** p

mesh = of.UnitSquareMesh(comm, n, n)
X = of.FunctionSpace(mesh, 'CG', 1)
f = of.interpolate(of.Constant(1e-2), X)

class Initial_density(of.UserExpression):
    def eval(self, value, x):
        value[0] = m

class Left(of.SubDomain):
    def inside(self, x, on_boundary):
        gamma = 1/250 + 1e-5
        return x[0] == 0.0 and 0.5 - gamma < x[1] < 0.5 + gamma and on_boundary

U = of.FunctionSpace(mesh, 'CG', 1)
t = of.TrialFunction(U)
dt = of.TestFunction(U)
bc = of.DirichletBC(U, of.Constant(0.0), Left())

class PoissonProblem(of.Module):
    def problem(self, controls):
        rho = controls[0]
        rho = of.helmholtzFilter(rho, X, R)
        a = of.inner(of.grad(t), k(rho)*of.grad(dt))*of.dx
        L = of.inner(f, dt)*of.dx
        Th = of.Function(U, name='Temperture')
        of.solve(a==L, Th, bc)
        J = of.assemble(of.inner(of.grad(Th), k(rho)*of.grad(Th))*of.dx)
        one_field = of.project(of.Constant(1.0), X)
        rho_0 = of.assemble(one_field*of.dx)
        rho_t = of.assemble(controls[0]*of.dx)
        self.rel = rho_t/rho_0
        rho.rename('rho', 'rho')
        file_rho.write(rho, self.index)
        return J
    def constraint_volume(self):
        return self.rel - m
    
x0 = of.Function(X)
x0.interpolate(Initial_density())
N = of.Function(X).vector().size()
min_bounds = np.zeros(N)
max_bounds = np.ones(N)

setting = {'set_lower_bounds': min_bounds,
           'set_upper_bounds': max_bounds,
           'set_maxeval': 100
          }
params = {'verbosity': 1}

problem = PoissonProblem()
solution = of.optimize(problem, [x0], [0], setting, params)

with open(dir + 'log_obj.csv', 'wt', encoding='utf-8') as file:
    for value in problem.log_obj:
        file.write(str(float(value)) + '\n')
with open(dir + 'log_cns.csv', 'wt', encoding='utf-8') as file:
    for value in problem.log_cns:
        file.write(str(float(value)) + '\n')