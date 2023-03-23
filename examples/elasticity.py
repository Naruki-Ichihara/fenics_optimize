'''
Example code for elasticity topology optimization problem.
Naruki Ichihara (2023)
'''
import numpy as np
import optfx as of

# Settings
of.parameters["form_compiler"]["optimize"] = True
of.parameters["form_compiler"]["cpp_optimize"] = True
of.parameters['form_compiler']['quadrature_degree'] = 5
comm = of.MPI.comm_world

# I/O paths
dir = './examples/results/elasticity/'
file_rho = of.XDMFFile(dir + 'rho.xdmf')
file_disp = of.XDMFFile(dir + 'disp.xdmf')
file_sig = of.XDMFFile(dir + 'stress.xdmf')

# Hyperparamters
target = 0.30   # Target rate of the material amount
p = 5      # Penalty parameter
eps = 1e-2 # Material lower bound
R = 0.5    # Helmholtz filter radius
T = of.Constant((0.0, -1.0)) # Traction on boundary
beta = 30
eta = 0.3

# Material properties (isotropic plane stress)
E = 1.0e5
nu = 0.3
mu = E/2/(1+nu)
lmbda = E*nu/(1+nu)/(1-2*nu)
lmbda = 2*mu*lmbda/(lmbda+2*mu)
strain = lambda u: of.sym(of.grad(u))
stress = lambda u: lmbda*of.tr(strain(u))*of.Identity(2) + 2.0*mu*strain(u)

# SIMP parametrization
simp = lambda x, p, eps: eps + (1 - eps) * x ** p

# Define problem space
L = 50
H = 25
D = 2.0
n = 4
mesh = of.RectangleMesh(comm, of.Point((0, 0)), of.Point((L, H)), L*n, H*n)
U = of.FunctionSpace(mesh, 'CG', 1)
V = of.VectorFunctionSpace(mesh, 'CG', 1)

class Initial_density(of.UserExpression):
    def eval(self, value, x):
        value[0] = target

EPS = 1e-10
class Left(of.SubDomain):
    def inside(self, x, on_noundary):
        return on_noundary and x[0] < EPS
class Right_load(of.SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and x[0] > L-EPS and  H/2-D/2 < x[1] < H/2+D/2
    
facets = of.MeshFunction('size_t', mesh, 1)
facets.set_all(0)
loading_boundary = Right_load()
loading_boundary.mark(facets, 1)
dS = of.Measure('ds', subdomain_data=facets)
fixed_bc = of.DirichletBC(V, of.Constant((0, 0)), Left())
bcs = [fixed_bc]

us = of.Function(V, name='Displacement')
u = of.TrialFunction(V)
du = of.TestFunction(V)
bc = of.DirichletBC(U, of.Constant(0.0), Left())

# bilinear form
a = lambda u, v, rho: of.inner(stress(u)*simp(rho, p, eps), strain(v))*of.dx
# Linear form
l = lambda v: of.inner(T, v)*dS(1)

class ElasticProblem(of.Module):
    def problem(self, controls):
        rho = controls[0]
        rho = of.hevisideFilter(of.helmholtzFilter(rho, U, R), U, beta=beta, eta=eta)
        of.solve(a(u, du, rho) == l(du), us, bcs)
        cost = of.assemble(l(us))
        one_field = of.project(of.Constant(1.0), U)
        rho_0 = of.assemble(one_field*of.dx)
        rho_t = of.assemble(controls[0]*of.dx)
        self.rel = rho_t/rho_0

        sig = of.project(stress(us), of.TensorFunctionSpace(mesh, 'CG', 1))
        sig.rename('stress', 'stress')
        rho.rename('rho', 'rho')
        file_rho.write(rho, self.index)
        file_disp.write(us, self.index)
        file_sig.write(sig, self.index)
        return cost
    def constraint_volume(self):
        return self.rel - target
    
x0 = of.Function(U)
x0.interpolate(Initial_density())
N = of.Function(U).vector().size()
min_bounds = np.zeros(N)
max_bounds = np.ones(N)

setting = {'set_lower_bounds': min_bounds,
           'set_upper_bounds': max_bounds,
           'set_maxeval': 100
          }
params = {'verbosity': 1}

problem = ElasticProblem()
solution = of.optimize(problem, [x0], [0], setting, params)

with open(dir + 'log_obj.csv', 'wt', encoding='utf-8') as file:
    for value in problem.log_obj:
        file.write(str(float(value)) + '\n')
with open(dir + 'log_cns.csv', 'wt', encoding='utf-8') as file:
    for value in problem.log_cns:
        file.write(str(float(value)) + '\n')
