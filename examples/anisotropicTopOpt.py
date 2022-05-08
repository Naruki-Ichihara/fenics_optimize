from dolfin import *
from dolfin_adjoint import *
import fenics_optimize as op
import numpy as np
from ufl.operators import atan_2, transpose
import nlopt as nl

E1 = 1
E2 = 1/15
nu12 = 0.33
G12 = 1/20
eps = 0.01
target = 0.5
R = 0.5
R_th = 0.5
f = Constant((0, -10))

def SIMP(x, p=3):
    return eps + (1-eps)*x**p

def strain_to_voigt(e):
    return as_vector((e[0, 0], e[1, 1], 2*e[0, 1]))

def stress_to_voigt(sigma):
    return as_vector((sigma[0, 0], sigma[1, 1], sigma[0, 1]))

def strain_from_voigt(e_voigt):
    return as_matrix(((e_voigt[0], e_voigt[2]/2.), (e_voigt[2]/2., e_voigt[1])))

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

nu21 = E2/E1*nu12
Q11 = E1/(1-nu12*nu21)
Q12 = nu12*E2/(1-nu12*nu21)
Q22 = E2/(1-nu12*nu21)
Q66 = G12
Q16 = 0.
Q26 = 0.
Q = as_matrix([[Q11, Q12, Q16],
                [Q12, Q22, Q26],
                [Q16, Q26, Q66]])

mesh = RectangleMesh(Point((0, 0)), Point(75, 25), 75, 25)
X = FunctionSpace(mesh, "CG", 1)
V = VectorFunctionSpace(mesh, "CG", 1)
Xs = [X, X, X]
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
        zeta = controls[1]
        eta = controls[2]
        rho = op.helmholtzFilter(rho, X, R)
        phi = op.helmholtzFilter(op.box2circleConstraint(zeta, eta), V, R_th)
        theta = atan_2(phi[1], phi[0])
        Q_reduced = SIMP(rho)*Q
        Q_rotated = rotation_matrix(Q_reduced, theta)
        a = inner(stress(Q_rotated, u), strain(v))*dx
        L = inner(f, v)*ds(1)
        A, b = assemble_system(a, L, bcs)
        us = Function(V)
        us = op.amg(A, b, us)
        cost = assemble(inner(stress(Q_rotated, us), strain(us))*dx)

        rho_bulk = project(Constant(1.0), X)
        rho_0 = assemble(rho_bulk*dx)
        rho_f = assemble(controls[0]*dx)
        self.volumeFraction = rho_f/rho_0

        return cost

    def volume_constraint(self):
        return self.volumeFraction - target

x0 = np.ones(N)*target
z0 = np.ones(N)
e0 = np.zeros(N)
initial = np.concatenate([x0, z0, e0])

x_min = np.zeros(N)
z_min = - np.ones(N)
e_min = - np.ones(N)
min_bounds = np.concatenate([x_min, z_min, e_min])

x_max = np.ones(N)
z_max = np.ones(N)
e_max = np.ones(N)
max_bounds = np.concatenate([x_max, z_max, e_max])

controls = [x0, z0, e0]
templates = [X, X, X]

anisoTopOpt = Problem()
optimizer = nl.opt(nl.LD_MMA, problemSize)

def eval(x, grad):
    xs = np.split(x, 3)
    cost = anisoTopOpt.forward(xs, templates)
    grad[:] = np.concatenate(anisoTopOpt.backward())
    return cost

def const(x, grad):
    measure = anisoTopOpt.volume_constraint()
    grad[:] = np.concatenate(anisoTopOpt.backward_constraint('volume_constraint', wrt=[0]))
    return measure

optimizer.set_min_objective(eval)
optimizer.add_inequality_constraint(const, 1e-5)
optimizer.set_lower_bounds(min_bounds)
optimizer.set_upper_bounds(max_bounds)
optimizer.set_maxeval(10)
optimizer.set_param('verbosity', 5)
solution = optimizer.optimize(initial)

x_opt, z_opt, e_opt = np.split(solution, 3)