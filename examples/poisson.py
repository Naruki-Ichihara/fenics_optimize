from dolfin import *
from dolfin_adjoint import *

parameters["std_out_all_processes"] = False

V = Constant(0.4)  # volume bound on the control
p = Constant(5)  # power used in the solid isotropic material
eps = Constant(1.0e-3)  # epsilon used in the solid isotropic material
alpha = Constant(1.0e-8)  # regularisation coefficient in functional

def k(a):
    return eps + (1 - eps) * a ** p

n = 250
mesh = UnitSquareMesh(n, n)
A = FunctionSpace(mesh, "CG", 1)  # function space for control
P = FunctionSpace(mesh, "CG", 1)  # function space for solution

class WestNorth(SubDomain):
    """The top and left boundary of the unitsquare, used to enforce the Dirichlet boundary condition."""

    def inside(self, x, on_boundary):
        return (x[0] == 0.0 or x[1] == 1.0) and on_boundary

bc = [DirichletBC(P, 0.0, WestNorth())]
f = interpolate(Constant(1.0e-2), P)  # the volume source term for the PDE

def forward(a):
    T = Function(P, name="Temperature")
    v = TestFunction(P)
    F = inner(grad(v), k(a) * grad(T)) * dx - f * v * dx
    solve(F == 0, T, bc, solver_parameters={"newton_solver": {"absolute_tolerance": 1.0e-7, "maximum_iterations": 20}})
    return T


a = interpolate(V, A)  # initial guess.
T = forward(a)  # solve the forward problem once.

controls = File("output/control_iterations.pvd")
a_viz = Function(A, name="ControlVisualisation")


def eval_cb(j, a):
    a_viz.assign(a)
    controls << a_viz

J = assemble(f * T * dx + alpha * inner(grad(a), grad(a)) * dx)
m = Control(a)
Jhat = ReducedFunctional(J, m, eval_cb_post=eval_cb)

lb = 0.0
ub = 1.0

class VolumeConstraint(InequalityConstraint):
    """A class that enforces the volume constraint g(a) = V - a*dx >= 0."""

    def __init__(self, V):
        self.V = float(V)
        self.smass = assemble(TestFunction(A) * Constant(1) * dx)
        self.tmpvec = Function(A)

    def function(self, m):
        from pyadjoint.reduced_functional_numpy import set_local
        set_local(self.tmpvec, m)

        integral = self.smass.inner(self.tmpvec.vector())
        if MPI.rank(MPI.comm_world) == 0:
            print("Current control integral: ", integral)
        return [self.V - integral]

    def jacobian(self, m):
        return [-self.smass]

    def output_workspace(self):
        return [0.0]

    def length(self):
        """Return the number of components in the constraint vector (here, one)."""
        return 1

problem = MinimizationProblem(Jhat, bounds=(lb, ub), constraints=VolumeConstraint(V))

parameters = {"acceptable_tol": 1.0e-3, "maximum_iterations": 100}
solver = IPOPTSolver(problem, parameters=parameters)
a_opt = solver.solve()

File("output/final_solution.pvd") << a_opt
xdmf_filename = XDMFFile(MPI.comm_world, "output/final_solution.xdmf")
xdmf_filename.write(a_opt)