from dolfin import *
from dolfin_adjoint import *
from ufl.operators import atan_2, transpose
from morphogenesis import forward
from morphogenesis.Core import forward
from morphogenesis.Solvers import AMG2Dsolver
from morphogenesis.Filters import helmholtzFilter, hevisideFilter, isoparametric2Dfilter
from morphogenesis.FileIO import export_result
from morphogenesis.Elasticity import epsilon
from morphogenesis.Optimizer import MMAoptimize, HSLoptimize
import numpy as np

def stress_from_voigt(sigma_voigt):
    return as_matrix(((sigma_voigt[0], sigma_voigt[2]), (sigma_voigt[2], sigma_voigt[1])))

def strain_to_voigt(e):
    return as_vector((e[0, 0], e[1, 1], 2*e[0, 1]))

def rotated_lamina_stiffness_inplane(E1, E2, G12, nu12, theta):
    c = cos(theta)
    s = sin(theta)
    T = as_matrix([[c**2, s**2, -2*s*c],
                   [s**2, c**2, 2*s*c],
                   [c*s, -c*s, c**2 - s**2]])
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
    Q_theta = T*Q*transpose(T)
    return Q_theta

E1 = 6
E2 = 1
nu12 = 0.3
G12 = 1
p = 3
f = Constant((0, -0.01))
target = 0.3

def stress(Q, u):
    return stress_from_voigt(Q*strain_to_voigt(sym(grad(u))))

class Clamp(SubDomain):
    def inside(self, x, on_boundary):
        return x[1] < -100.0 and on_boundary

class Loading(SubDomain):
    def inside(self, x, on_boundary):
        return x[1] > 18.50 and on_boundary

mesh = Mesh('/workspace/mesh/implant.xml')
N = mesh.num_vertices()

facets = MeshFunction('size_t', mesh, 1)
facets.set_all(0)
Top = Loading()
Top.mark(facets, 1)
ds = Measure('ds', subdomain_data=facets)

z0 = np.zeros(N)
e0 = np.ones(N)
r0 = np.zeros(N)
x0 = np.concatenate([z0, e0, r0])

X = FunctionSpace(mesh, 'CG', 1)
V = VectorFunctionSpace(mesh, "CG", 1)

@forward([X, X, X])
def simulator(xs):
    z = xs[0]
    e = xs[1]
    r = xs[2]
    rho = hevisideFilter(helmholtzFilter(r, X, 20.0), a=5)
    phi = helmholtzFilter(isoparametric2Dfilter(z, e), V)
    theta = atan_2(phi[1], phi[0])
    export_result(project(rho, X), 'result/dens.xdmf')
    export_result(project(phi, V), 'result/theta.xdmf')

    u = TrialFunction(V)
    du = TestFunction(V)
    uh = Function(V)

    Q = rho**p*rotated_lamina_stiffness_inplane(E1, E2, G12, nu12, theta)
    a = inner(stress(Q, u), epsilon(du))*dx
    L = inner(f, du)*ds(1)
    bc = DirichletBC(V, Constant((0, 0)), Clamp())
    A, b = assemble_system(a, L, [bc])
    solver = AMG2Dsolver(A, b)
    uh = solver.forwardSolve(uh, V, False)
    J = assemble(inner(stress(Q, uh), epsilon(uh))*dx)
    export_result(uh, 'result/disp.xdmf')
    return J

@forward([X, X, X], wrt=[2])
def constraints(x):
    r = x[2]
    rho = hevisideFilter(helmholtzFilter(r, X, 20), a=5)
    rho_bulk = project(Constant(1.0), FunctionSpace(mesh, 'CG', 1))
    rho_0 = assemble(rho_bulk*dx)
    rho_f = assemble(rho*dx)
    rel = rho_f/rho_0
    val = rel - target
    return val

HSLoptimize(N*3, x0, simulator, constraints)