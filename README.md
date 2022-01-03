# fenics-optimize
<!-- # Short Description -->

**fenics-optimize** is a module of the [FEniCS computing platform](https://fenicsproject.org/) for the multiphysical optimization problems.

<!-- # Badges -->

[![Github issues](https://img.shields.io/github/issues/Naruki-Ichihara/morphogenesis?style=for-the-badge&logo=appveyor)](https://github.com/Naruki-Ichihara/morphogenesis/issues)
[![Github forks](https://img.shields.io/github/forks/Naruki-Ichihara/morphogenesis?style=for-the-badge&logo=appveyor)](https://github.com/Naruki-Ichihara/morphogenesis/network/members)
[![Github stars](https://img.shields.io/github/stars/Naruki-Ichihara/morphogenesis?style=for-the-badge&logo=appveyor)](https://github.com/Naruki-Ichihara/morphogenesis/stargazers)
[![Github top language](https://img.shields.io/github/languages/top/Naruki-Ichihara/morphogenesis?style=for-the-badge&logo=appveyor)](https://github.com/Naruki-Ichihara/morphogenesis/)
[![Github license](https://img.shields.io/github/license/Naruki-Ichihara/morphogenesis?style=for-the-badge&logo=appveyor)](https://github.com/Naruki-Ichihara/morphogenesis/)

## Motivation and significance

### Automatic derivative
Sensitivities that need to optimization will be derived automatically from the fenics chain. 

```python
Xs = [X1, X2, ..., Xn]  # Function Spaces
@op.with_derivative(Xs)
def forward(xs):
    process with xs ..
    return J(xs)
```

### Filters
Built-in filters for topology optimizations.

* Heviside Filter
* Helmholtz Filter
* Isoparametric Filter

### Built-in solvers
**fenics-optimize** contains the Smoothed aggregation algebaric multigrid for large-scale partial differential equation (PDE) using the [PETSc](https://petsc.org/release/) backend.

### Optimizer
**fenics-optimize** supports some optimizers based on NLopt or IPOPT. Currently supported

* Method of Moving Asymptotes (MMA)
* Ipopt-HSL (ma27)

## Installation
### Docker

[![dockeri.co](https://dockeri.co/image/ichiharanaruki/fenics-optimize)](https://hub.docker.com/r/ichiharanaruki/fenics-optimize)

Docker enables to build and ship the environment for **fenics-optimize** for almost any platform, e.g., Linux, macOS, or windows.

First, please install Docker. Linux users should follow the [instraction](https://docs.docker.com/get-started/). Mac or Windows users should install the [Docker Desktop](https://www.docker.com/products/docker-desktop), which suits your platform.

Second, clone this repository on your system.
```
git clone https://github.com/Naruki-Ichihara/fenics-optimize.git
```
and move to the docker-compose directory
```
cd morphogenesis/.docker_optimize
```
Then, launch the docker-compose and pull the image from our docker hub.
```
docker-compose up
```
This container will survive until when you stop the container.

### Manual installation
First, make sure to install the following dependencies:

* [FEniCS](https://fenicsproject.org/)
* [pyadjoint](https://github.com/dolfin-adjoint/pyadjoint)
* [cyipopt](https://github.com/mechmotum/cyipopt)
* [nlopt](https://github.com/stevengj/nlopt/) with python plugin
* [fecr](https://github.com/IvanYashchuk/fecr)

Second, clone this repository on your system and move in this directory.
```
git clone https://github.com/Naruki-Ichihara/fenics-optimize.git && cd fenics-optimize
```
Then, install this repository.
```
pip install .
```

## Example
2-D elastic topology optimization problem.

```python
from dolfin import *
from dolfin_adjoint import *
import optimize as op
from optimize.Elasticity import reducedSigma, epsilon
import numpy as np

E = 1.0e9
nu = 0.3
f = Constant((0, -1e3))
p = 3
target = 0.4
R = 0.1

mesh = RectangleMesh(Point(0, 0), Point(20, 10), 200, 100)
problemSize = mesh.num_vertices()
x0 = np.zeros(problemSize)

X = FunctionSpace(mesh, "CG", 1)
Xs = [X]
V = VectorFunctionSpace(mesh, "CG", 1)

class Bottom(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and x[1] < 1e-10

def clamped_boundary(x, on_boundary):
    return on_boundary and x[0] < 1e-10 or x[0] > 19.999

@op.with_derivative(Xs)
def forward(xs):
    rho = op.hevisideFilter(op.helmholtzFilter(xs[0], X, R))
    op.export_result(project(rho, X), 'result/test.xdmf')
    facets = MeshFunction('size_t', mesh, 1)
    facets.set_all(0)
    bottom = Bottom()
    bottom.mark(facets, 1)
    ds = Measure('ds', subdomain_data=facets)
    u = TrialFunction(V)
    v = TestFunction(V)
    a = inner(reducedSigma(rho, u, E, nu, p), epsilon(v))*dx
    L = inner(f, v)*ds(1)
    bc = DirichletBC(V, Constant((0, 0)), clamped_boundary)
    u_ = Function(V)
    A, b = assemble_system(a, L, [bc])
    uh = op.AMG2Dsolver(A, b).solve(u_, V, False)
    return assemble(inner(reducedSigma(rho, uh, E, nu, p), epsilon(uh))*dx)

@op.with_derivative(Xs)
def constraint(xs):
    rho_bulk = project(Constant(1.0), FunctionSpace(mesh, 'CG', 1))
    rho_0 = assemble(rho_bulk*dx)
    rho_f = assemble(op.hevisideFilter(op.helmholtzFilter(xs[0], X, R))*dx)
    rel = rho_f/rho_0
    return rel - target

op.HSLoptimize(problemSize, x0, forward, constraint)
```

## Contributors

- [Naruki-Ichihara](https://github.com/Naruki-Ichihara)

## References
## Cite
