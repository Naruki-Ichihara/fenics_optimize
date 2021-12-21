from dolfin import *
from dolfin_adjoint import *
from morphogenesis.Chain import numpy2fenics
from morphogenesis.Solvers import SLUDsolver
import numpy as np

E = 1.0e9
nu = 0.3
p = 3
target = 0.4

mesh = RectangleMesh(MPI.comm_world, Point(0, 0), Point(20, 10), 300, 200)
N = mesh.num_vertices()

x0 = np.zeros(N)

X = FunctionSpace(mesh, "CG", 1)
V = VectorFunctionSpace(mesh, "Lagrange", 1)

class Bottom(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and x[1] < 1e-10

def clamped_boundary(x, on_boundary):
    return on_boundary and x[0] < 1e-10 or x[0] > 19.999

def evaluator(x, grad):
    x_ = numpy2fenics(x, X)
    rho = m.hevisideFilter(m.helmholtzFilter(x_, X))
    m.export_result(project(rho, FunctionSpace(mesh, 'DG', 0)), 'result/test.xdmf')
    facets = MeshFunction('size_t', mesh, 1)
    facets.set_all(0)
    bottom = Bottom()
    bottom.mark(facets, 1)
    ds = Measure('ds', subdomain_data=facets)
    f = Constant((0, -1e3))
    u = TrialFunction(V)
    v = TestFunction(V)
    a = inner(m.reducedSigma(rho, u, E, nu, p), m.epsilon(v))*dx
    L = inner(f, v)*ds(1)
    bc = DirichletBC(V, Constant((0, 0)), clamped_boundary)
    u_ = Function(V)
    solver = m.AMG2Dsolver(a, L, [bc])
    uh = solver.forwardSolve(u_, V, False)
    J = assemble(inner(m.reducedSigma(rho, uh, E, nu, p), m.epsilon(uh))*dx)
    dJdx = m.evalGradient(J, x_)
    grad[:] = dJdx
    print('Cost : {}'.format(J))
    return J

def volumeResponce(x, grad):
    x_ = m.numpy2fenics(x, X)
    rho_bulk = project(Constant(1.0), FunctionSpace(mesh, 'CG', 1))
    rho_0 = assemble(rho_bulk*dx)
    rho_f = assemble(m.hevisideFilter(m.helmholtzFilter(x_, X))*dx)
    rel = rho_f/rho_0
    val = rel - target
    dreldx = m.evalGradient(val, x_)
    grad[:] = dreldx
    print('Constraint : {}'.format(val))
    return val

m.MMAoptimize(N, x0, evaluator, volumeResponce)