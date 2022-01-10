from dolfin import *
from dolfin_adjoint import *
import fenics_optimize as op
from ufl.operators import atan_2
from fenics_optimize.Mechanics import epsilon, stress_from_voigt, strain_to_voigt, rotated_lamina_stiffness_inplane
import numpy as np

def stress(Q, u):
    return stress_from_voigt(Q*strain_to_voigt(sym(grad(u))))

E1 = 6
E2 = 1
nu12 = 0.3
G12 = 1
f = Constant((0, -1e3))
p = 3
eps = 1.0e-3
target = 0.4
R = 0.001

rec_rho = op.Recorder('result/anisoElast', 'material')
rec_phi = op.Recorder('result/anisoElast', 'alinge')
logger = op.Logger('result/anisoElast', 'cost')

def k(rho):
    return eps + (1 - eps) * rho ** p

mesh = RectangleMesh(Point(0, 0), Point(20, 10), 100, 50)
N = mesh.num_vertices()

X = FunctionSpace(mesh, "CG", 1)
Xs = [X, X, X]
V = VectorFunctionSpace(mesh, "CG", 1)

class Bottom(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and x[1] < 1e-10

def clamped_boundary(x, on_boundary):
    return on_boundary and x[0] < 1e-10 or x[0] > 19.999

@op.with_derivative(Xs)
def forward(xs):
    rho = op.helmholtzFilter(xs[2], X, R)
    phi = op.helmholtzFilter(op.isoparametric2Dfilter(xs[0], xs[1]), V)
    theta = atan_2(phi[1], phi[0])
    facets = MeshFunction('size_t', mesh, 1)
    facets.set_all(0)
    bottom = Bottom()
    bottom.mark(facets, 1)
    ds = Measure('ds', subdomain_data=facets)
    Q = k(rho)*rotated_lamina_stiffness_inplane(E1, E2, G12, nu12, theta)
    u = TrialFunction(V)
    v = TestFunction(V)
    a = inner(stress(Q, u), epsilon(v))*dx
    L = inner(f, v)*ds(1)
    bc = DirichletBC(V, Constant((0, 0)), clamped_boundary)
    uh = Function(V)
    A, b = assemble_system(a, L, [bc])
    uh = op.AMGsolver(A, b).solve(uh, V, False, '2-D')
    rec_rho.rec(project(rho, X))
    rec_phi.rec(phi)
    cost = assemble(inner(stress(Q, uh), epsilon(uh))*dx)
    logger.rec(cost)
    return cost

@op.with_derivative(Xs, wrt=[2])
def constraint(xs):
    rho_bulk = project(Constant(1.0), X)
    rho_0 = assemble(rho_bulk*dx)
    rho_f = assemble(xs[2]*dx)
    rel = rho_f/rho_0
    return rel - target

z0 = np.ones(N)
e0 = np.zeros(N)
r0 = np.ones(N)*target
x0 = np.concatenate([z0, e0, r0])

z_min = - np.ones(N)
e_min = - np.ones(N)
r_min = np.zeros(N)

x_min = np.concatenate([z_min, e_min, r_min])
x_max = np.ones(N*3)

op.HSLoptimize(N*3, x0, forward, [constraint], maxeval=1000, bounds=[x_min, x_max], rel=1e-20, solver_type='ma97')