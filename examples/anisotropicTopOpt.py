from dolfin import *
from dolfin_adjoint import *
import fenics_optimize as op
from fenics_optimize.optimizer import interface_nlopt
import numpy as np
from ufl.operators import atan_2, transpose
# Some flags for FEniCS
parameters["form_compiler"]["optimize"] = True
parameters["form_compiler"]["cpp_optimize"] = True

# Quadrature degree in FEniCS  (sometimes, the "automatic" determination of the quadrature degree becomes excessively high, meaning that it should be manually reduced)
parameters['form_compiler']['quadrature_degree'] = 5

eps = 0.1
target = 0.5
R = 0.5
R_th = 0.5
f = Constant((0, -10))

def SIMP(x, coff, p):
    return eps + (1-eps)*coff*x**p

def POL(x):
    return 0.2733*x**2-0.0292*x+0.0245

def strain_to_voigt(e):
    return as_vector((e[0, 0], e[1, 1], 2*e[0, 1]))

def stress_from_voigt(sigma_voigt):
    return as_matrix(((sigma_voigt[0], sigma_voigt[2]), (sigma_voigt[2], sigma_voigt[1])))

def rotation_matrix(Q, theta):
    c = cos(theta)
    s = sin(theta)
    T = as_matrix([[c**2, s**2, -2*c*s],
                   [s**2, c**2, 2*c*s],
                   [c*s, -c*s, c**2-s**2]])
    return T*Q*transpose(T)

def stress(Q, u):
    return stress_from_voigt(Q*strain_to_voigt(sym(grad(u))))

def strain(u):
    return sym(grad(u))

class Symmetry(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and x[0] < 1e-10

class BottomPoint(SubDomain):
    def inside(self, x, on_boundary):
        return x[1] < 1e-10 and x[0] > 75-1e-10

class Loading(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and x[1] > 25-1e-10 and x[0] < 3

class Initial_alinge_1(UserExpression):
    def eval(self, value, x):
        value[0] = 1

class Initial_alinge_2(UserExpression):
    def eval(self, value, x):
        value[0] = 0

class Initial_density(UserExpression):
    def eval(self, value, x):
        value[0] = target

class Initial_alinge(UserExpression):
    def eval(self, value, x):
        value[0] = 1
        value[1] = 0
    def value_shape(self):
        return (2,)

Q11 = 2334.50
Q12 = 863.59
Q22 = 2353.03
Q66 = 685.62
Q16 = 0.
Q26 = 0.
Q = as_matrix([[Q11, Q12, Q16],
                [Q12, Q22, Q26],
                [Q16, Q26, Q66]])

mesh = RectangleMesh(Point((0, 0)), Point(75, 25), 75*2, 25*2)
X = FunctionSpace(mesh, "CG", 1)
V = VectorFunctionSpace(mesh, "CG", 1)
N = Function(X).vector().size()
problemSize = N*3

facets = MeshFunction('size_t', mesh, 1)
facets.set_all(0)
bottom = Loading()
bottom.mark(facets, 1)
ds = Measure('ds', subdomain_data=facets)
u = TrialFunction(V)
v = TestFunction(V)
bc1 = DirichletBC(V.sub(0), Constant(0), Symmetry())
bc2 = DirichletBC(V.sub(1), Constant(0), BottomPoint(), method='pointwise')
bcs = [bc1, bc2]

class Problem(op.Module):
    def problem(self, controls):
        rho = controls[0]
        zeta = controls[1][0]
        eta = controls[1][1]
        rho = op.helmholtzFilter(rho, X, R)
        phi = op.helmholtzFilter(op.box2circleConstraint(zeta, eta), V, R_th)
        theta = atan_2(phi[1], phi[0])
        Q_reduced_11 = POL(rho)*Q[0, 0]
        Q_reduced_22 = POL(rho)*Q[0, 0]
        Q_reduced_12 = POL(rho)*Q[0, 1]
        Q_reduced_66 = POL(rho)*Q[2, 2]
        Q_reduced = as_matrix([[Q_reduced_11, Q_reduced_12, Q16],
                                [Q_reduced_12, Q_reduced_22, Q26],
                                [Q16, Q26, Q_reduced_66]])
        Q_rotated = rotation_matrix(Q_reduced, theta)
        a = inner(stress(Q_rotated, u), strain(v))*dx
        L = inner(f, v)*ds(1)
        A, b = assemble_system(a, L, bcs)
        us = Function(V)
        us = op.amg(A, b, us, False)
        cost = assemble(inner(stress(Q_rotated, us), strain(us))*dx)
        rho_bulk = project(Constant(1.0), X)
        rho_0 = assemble(rho_bulk*dx)
        rho_f = assemble(controls[0]*dx)
        self.constraint_Volume = rho_f/rho_0 - target
        return cost

x0 = Function(X)
x0.interpolate(Initial_density())
z0 = Function(X)
z0.interpolate(Initial_alinge_1())
e0 = Function(X)
e0.interpolate(Initial_alinge_2())
v0 = Function(V)
v0.interpolate(Initial_alinge())
initials = [x0, v0]
min_bounds = np.concatenate([np.zeros(N), -np.ones(N), -np.ones(N)])
max_bounds = np.concatenate([np.ones(N)*0.9, np.ones(N), np.ones(N)])

anisoTopOpt = Problem()

setting = {'set_lower_bounds': min_bounds,
           'set_upper_bounds': max_bounds,
           'set_maxeval': 10
          }
params = {'verbosity': 5}

solution = interface_nlopt(anisoTopOpt, initials, [0], setting, params)
file = File('/workspace/examples/results/test.pvd')
file << solution[1]