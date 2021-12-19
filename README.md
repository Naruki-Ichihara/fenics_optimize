# morphogenesis

<!-- # Short Description -->

morphogenesis is the high-resolution topology optimization toolkit using the FEniCS.

<!-- # Badges -->

[![Github issues](https://img.shields.io/github/issues/Naruki-Ichihara/morphogenesis?style=for-the-badge&logo=appveyor)](https://github.com/Naruki-Ichihara/morphogenesis/issues)
[![Github forks](https://img.shields.io/github/forks/Naruki-Ichihara/morphogenesis?style=for-the-badge&logo=appveyor)](https://github.com/Naruki-Ichihara/morphogenesis/network/members)
[![Github stars](https://img.shields.io/github/stars/Naruki-Ichihara/morphogenesis?style=for-the-badge&logo=appveyor)](https://github.com/Naruki-Ichihara/morphogenesis/stargazers)
[![Github top language](https://img.shields.io/github/languages/top/Naruki-Ichihara/morphogenesis?style=for-the-badge&logo=appveyor)](https://github.com/Naruki-Ichihara/morphogenesis/)
[![Github license](https://img.shields.io/github/license/Naruki-Ichihara/morphogenesis?style=for-the-badge&logo=appveyor)](https://github.com/Naruki-Ichihara/morphogenesis/)

# Advantages

Morphogenesis allows the high-level design of mechanical parts.

# Installation

This repository depends following library;

* FEniCS
* NLopt
* Numpy
* fecr

We highly recommend you use our docker image.
## Docker
Docker enables to build and ship the environment for **morphogenesis** for almost any platform, e.g., Linux, macOS, or windows.

First, please install Docker. Linux users should follow the [instraction](https://docs.docker.com/get-started/). Mac or Windows users should install the [Docker Desktop](https://www.docker.com/products/docker-desktop), which suits your platform.

Second, clone this repository on your system.
```
git clone https://github.com/Naruki-Ichihara/morphogenesis.git
```
and move to the docker-compose directory
```
cd morphogenesis/.docker_morphogenesis
```
Then, launch the docker-compose and pull the image from our docker hub.
```
docker-compose up
```
This container will survive until when you stop the container.

# Example
This repository includes following solvers for the _large scale_ elasticity problem.

* Smoothed Aggregation Algerbaric Multigrid method (AMG)
* SuperLU_dist
* Mumps

Here, we select the AMG solver to solve the 2-D elastic topology optimization problem.

```python
from dolfin import *
from dolfin_adjoint import *
from morphogenesis.Solvers.AMGsolver import AMG2Dsolver
from morphogenesis.Filters.Filters import helmholtzFilter, hevisideFilter
from morphogenesis.Helpers.Elasticity import sigma, epsilon
from morphogenesis.Helpers.Chain import evalGradient, numpy2fenics
from morphogenesis.Optimizer.MMAoptimizer import MMAoptimize
from morphogenesis.Utils.file_io import export_result
import numpy as np
import nlopt as nl

E = 1.0e9
nu = 0.3
target = 0.4

mesh = RectangleMesh(MPI.comm_world, Point(0, 0), Point(20, 10), 200, 100)
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
    rho = hevisideFilter(helmholtzFilter(x_, X))
    facets = MeshFunction('size_t', mesh, 1)
    facets.set_all(0)
    bottom = Bottom()
    bottom.mark(facets, 1)
    ds = Measure('ds', subdomain_data=facets)
    f = Constant((0, -1e3))
    u = TrialFunction(V)
    v = TestFunction(V)
    a = inner(rho**3*sigma(u, E, nu), epsilon(v))*dx
    L = inner(f, v)*ds(1)
    bc = DirichletBC(V, Constant((0, 0)), clamped_boundary)
    u_ = Function(V)
    solver = AMG2Dsolver(a, L, [bc])
    uh = solver.forwardSolve(u_, V, False)
    J = assemble(inner(rho**3*sigma(uh, E, nu), epsilon(uh))*dx)
    dJdx = evalGradient(J, x_)
    grad[:] = dJdx
    print('Cost : {}'.format(J))
    export_result(project(rho, FunctionSpace(mesh, 'DG', 0)), 'result/test.xdmf')
    return J

def volumeResponce(x, grad):
    x_ = numpy2fenics(x, X)
    rho_bulk = project(Constant(1.0), FunctionSpace(mesh, 'CG', 1))
    rho_0 = assemble(rho_bulk*dx)
    rho_f = assemble(hevisideFilter(helmholtzFilter(x_, X))*dx)
    rel = rho_f/rho_0
    val = rel - target
    dreldx = evalGradient(val, x_)
    grad[:] = dreldx
    print('Constraint : {}'.format(val))
    return val

MMAoptimize(N, x0, evaluator, volumeResponce)
```

# Contributors

- [Naruki-Ichihara](https://github.com/Naruki-Ichihara)

# References
# Cite