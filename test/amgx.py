from dolfin import *
import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as splinalg
from scipy.sparse import csr_matrix
from morphogenesis.Elasticity import sigma, epsilon
from morphogenesis.FileIO import export_result

def build_nullspace_2D(V, x):
    #Function to build null space for 2D elasticity
    #
    # Create list of vectors for null space
    nullspace_basis = [x.copy() for i in range(2)]

    # Build translational null space basis - 2D problem
    #
    V.sub(0).dofmap().set(nullspace_basis[0], 1.0);
    V.sub(1).dofmap().set(nullspace_basis[1], 1.0);

    for x in nullspace_basis:
        x.apply("insert")

    # Create vector space basis and orthogonalize
    basis = VectorSpaceBasis(nullspace_basis)
    basis.orthonormalize()

    return basis

def tran2SparseMatrix(A):
    mat = as_backend_type(A).mat()
    return csr_matrix(mat.getValuesCSR()[::-1], shape=mat.size)

def build_nullspace_3D(V, x):
    """Function to build null space for 3D elasticity"""

    # Create list of vectors for null space
    nullspace_basis = [x.copy() for i in range(6)]

    # Build translational null space basis
    V.sub(0).dofmap().set(nullspace_basis[0], 1.0);
    V.sub(1).dofmap().set(nullspace_basis[1], 1.0);
    V.sub(2).dofmap().set(nullspace_basis[2], 1.0);

    # Build rotational null space basis
    V.sub(0).set_x(nullspace_basis[3], -1.0, 1);
    V.sub(1).set_x(nullspace_basis[3],  1.0, 0);
    V.sub(0).set_x(nullspace_basis[4],  1.0, 2);
    V.sub(2).set_x(nullspace_basis[4], -1.0, 0);
    V.sub(2).set_x(nullspace_basis[5],  1.0, 1);
    V.sub(1).set_x(nullspace_basis[5], -1.0, 2);

    for x in nullspace_basis:
        x.apply("insert")

    # Create vector space basis and orthogonalize
    basis = VectorSpaceBasis(nullspace_basis)
    basis.orthonormalize()

    return basis

import pyamgx



configPath = "/workspace/AMGX/build/configs/core/CG_DILU.json"

pyamgx.initialize()
cfg = pyamgx.Config().create_from_file(configPath)

rsc = pyamgx.Resources().create_simple(cfg)

# Create matrices and vectors:
A = pyamgx.Matrix().create(rsc)
b = pyamgx.Vector().create(rsc)
x = pyamgx.Vector().create(rsc)

# Create solver:
solver = pyamgx.Solver().create(rsc, cfg)

# Upload system:

mesh = RectangleMesh(Point(0, 0), Point(20, 10), 5000, 500)
""" 5 milion DOF"""
#mesh = BoxMesh(
#    MPI.comm_world, Point(0.0, 0.0, 0.0),
#                    Point(20.0, 10.0, 10.0), 50, 50, 50)

E = 1.0e9
nu = 0.3
mu = E/(2.0*(1.0 + nu))
lmbda = E*nu/((1.0 + nu)*(1.0 - 2.0*nu))

V = VectorFunctionSpace(mesh, "Lagrange", 1)

class Bottom(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and x[1] < 1e-10

def clamped_boundary(x, on_boundary):
    return on_boundary and x[0] > 20 - 1e-10

facets = MeshFunction('size_t', mesh, 1)
facets.set_all(0)
bottom = Bottom()
bottom.mark(facets, 1)
ds = Measure('ds', subdomain_data=facets)

f = Constant((0, -1e3))
u = TrialFunction(V)
v = TestFunction(V)
a = inner(sigma(u, E, nu), epsilon(v))*dx
L = inner(f, v)*ds(1)

bc = DirichletBC(V, Constant((0, 0)), clamped_boundary)
u = Function(V)
A_f, b_f = assemble_system(a, L, bc)
null_space = build_nullspace_2D(V, u.vector())
as_backend_type(A_f).set_near_nullspace(null_space)

A_ = tran2SparseMatrix(A_f)
b_ = b_f[:]
u_ = u.vector()[:]

A.upload_CSR(A_)
b.upload(b_)
x.upload(u_)

# Setup and solve system:
solver.setup(A)
solver.solve(b, x)

# Download solution
x.download(u_)
#print(type(u_))
u.vector()[:] = u_
export_result(u, 'amgx_test.xdmf')

# Clean up:
A.destroy()
x.destroy()
b.destroy()
solver.destroy()
rsc.destroy()
cfg.destroy()

pyamgx.finalize()