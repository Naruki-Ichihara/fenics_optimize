from dolfin import *
import numpy as np
from mpi4py import MPI
comm = MPI.COMM_WORLD

gamma = 1.
g = 0.0
dt = 0.01
w0 = 0.5

def delta(x, delta=0.1):
    return (x - delta)/(1 - delta)

mesh = RectangleMesh(comm, Point((0, 0)), Point(75, 25), 75*3, 25*3)
refined_mesh = RectangleMesh(comm, Point((0, 0)), Point(75, 25), 75*4, 25*4)

Vh = FiniteElement('CG', refined_mesh.ufl_cell(), 2)
ME = FunctionSpace(refined_mesh, Vh*Vh)
X = VectorFunctionSpace(mesh, 'CG', 1)
X_r = VectorFunctionSpace(refined_mesh, 'CG', 1)
Y = FunctionSpace(mesh, 'CG', 1)
Y_r = FunctionSpace(refined_mesh, 'CG', 1)

theta = Function(X, 'results/simp/alinge.xml')
theta_r = Function(X_r)
LagrangeInterpolator.interpolate(theta_r, theta)
#theta_r = as_vector([-theta_r[1], theta_r[0]])
#vectorField = VectorField()
#theta.interpolate(vectorField)
rho = Function(Y, 'results/simp/density.xml')
rho_r = Function(Y_r)
LagrangeInterpolator.interpolate(rho_r, rho)
#scalerField = ScalerField()
#rho.interpolate(scalerField)

eps = 1.0
k = sqrt((np.pi*rho_r/w0)**2 - rho_r**2*gamma**2)
#k = sqrt((np.pi*rho_r/w0)**2)# - gamma**2)

class GaussianRandomField(UserExpression):
    def eval(self, val, x):
        val[0] = np.sqrt(1)*np.random.randn()
        val[1] = np.sqrt(1)*np.random.randn()
    def value_shape(self):
        return (2,)

class VectorField(UserExpression):
    def eval(self, val, x):
        val[0] = 0
        val[1] = 1
    def value_shape(self):
        return (2,)

class ScalerField(UserExpression):
    def eval(self, val, x):
        val[0] = 0.001

class Problem(NonlinearProblem):
    def __init__(self, a, L):
        NonlinearProblem.__init__(self)
        self.L = L
        self.a = a
    def F(self, b, x):
        assemble(self.L, tensor=b)
    def J(self, A, x):
        assemble(self.a, tensor=A)

class CustomSolver(NewtonSolver):
    def __init__(self):
        NewtonSolver.__init__(self, comm, PETScKrylovSolver(), PETScFactory.instance())
    def solver_setup(self, A, P, problem, iteration):
        self.linear_solver().set_operator(A)
        PETScOptions.set("ksp_type", "gmres")
        PETScOptions.set("ksp_monitor")
        PETScOptions.set("pc_type", "lu")
        self.linear_solver().set_from_options()

def G1(w, v):
    return -eps/2 - g/3*(w+v) + 1/4*(w**2+w*v+v**2)

def G2(w):
    return -eps/2*w - g/3*w**2 + 1/4*w**3

def A(w, v, k):
    return (dot(grad(w), grad(v)) - k**2*w*v)*dx

def B(w, v, theta, gamma):
    D = outer(theta, theta)
    return 2*gamma**2*dot(grad(w), dot(D, grad(v)))*dx

Uh = Function(ME)
Uh_0 = Function(ME)
U = TrialFunction(ME)
phi, psi = TestFunctions(ME)

initial = GaussianRandomField()
Uh.interpolate(initial)
Uh_0.interpolate(initial)

uh, qh = split(Uh)
uh_0, qh_0 = split(Uh_0)

qh_mid = 0.5*qh + 0.5*qh_0
dPhi = G1(uh, uh_0)*uh + G2(uh_0)

L0 = (uh-uh_0)*phi*dx + dt*A(qh_mid, phi, k) - dt*B(uh, phi, theta_r, gamma) + dt*dPhi*phi*dx
L1 = qh*psi*dx - A(uh, psi, k)

L = L0 + L1
a = derivative(L, Uh, U)

SH_problem = Problem(a, L)
solver = CustomSolver()

t = 0
T = 3

file = File('results/simp/field_trans.pvd')
while (t < T):
    print('time: {}'.format(t))
    t += dt
    Uh_0.vector()[:] = Uh.vector()
    solver.solve(SH_problem, Uh.vector())
    sol_c = project(Uh.split()[0], Y_r)
    #sol_r = Function(R)
    #LagrangeInterpolator.interpolate(sol_r, sol_c)
    sol_c.rename('field', 'label')
    file << (sol_c, t)
