# morphogenesis _Under constraction_

<!-- # Short Description -->

Morphogenesis is the high-resolution topology optimization toolkit using the FEniCS.

<!-- # Badges -->

[![Github issues](https://img.shields.io/github/issues/Naruki-Ichihara/morphogenesis?style=for-the-badge&logo=appveyor)](https://github.com/Naruki-Ichihara/morphogenesis/issues)
[![Github forks](https://img.shields.io/github/forks/Naruki-Ichihara/morphogenesis?style=for-the-badge&logo=appveyor)](https://github.com/Naruki-Ichihara/morphogenesis/network/members)
[![Github stars](https://img.shields.io/github/stars/Naruki-Ichihara/morphogenesis?style=for-the-badge&logo=appveyor)](https://github.com/Naruki-Ichihara/morphogenesis/stargazers)
[![Github top language](https://img.shields.io/github/languages/top/Naruki-Ichihara/morphogenesis?style=for-the-badge&logo=appveyor)](https://github.com/Naruki-Ichihara/morphogenesis/)
[![Github license](https://img.shields.io/github/license/Naruki-Ichihara/morphogenesis?style=for-the-badge&logo=appveyor)](https://github.com/Naruki-Ichihara/morphogenesis/)

# Tags

`Topology` `Optimization` `Topology-Optimization` `python` `FEniCS`

# Advantages

Morphogenesis allows the high-level design of mechanical parts.

# Installation

This repository depends following library;

* FEniCS
* NLopt
* Numpy

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

# Solve elasticity problem using _smoothed aggregation algerbaric multigrid method_
This repository include following solvers for the _large scale _elasticity problem.

* Smoothed Aggregation Algerbaric Multigrid method (SAAM)
* Super LU (TO DO)
* Mumps (TO DO)

Here is the minimum example for solving the linear 3-D elastisity problem using SAAM (without topology optimization).
```python
from dolfin import *
from dolfin_adjoint import *
from Morphogenesis.Solvers.AMGsolver import AMGsolver
from Morphogenesis.Utils.file_io import export_result

mesh = BoxMesh(
    MPI.comm_world, Point(0.0, 0.0, 0.0),
                     Point(100.0, 10.0, 10.0), 100, 10, 10)

# Elasticity parameters
E = 1.0e9
nu = 0.3
mu = E/(2.0*(1.0 + nu))
lmbda = E*nu/((1.0 + nu)*(1.0 - 2.0*nu))

# Stress computation
def sigma(v):
    return 2.0*mu*sym(grad(v)) + lmbda*tr(sym(grad(v)))*Identity(len(v))

# Create function space
V = VectorFunctionSpace(mesh, "Lagrange", 1)

# Define variational problem
f = Constant((0, 0, -1e3))
u = TrialFunction(V)
v = TestFunction(V)
a = inner(sigma(u), grad(v))*dx
L = inner(f, v)*dx

# Set up boundary condition on inner surface
def clamped_boundary(x, on_boundary):
    return on_boundary and x[0] < 1e-10

bc = DirichletBC(V, Constant((0, 0, 0)), clamped_boundary)
u = Function(V)

problem = AMGsolver(a, L, [bc])
u = problem.forwardSolve(u, V, False)

export_result(u, 'result/test.xdmf')
```
Then, run with the message passing interface multi processing.
```
mpirun -n 4 --allow-run-as-root python3 elasticity.py
```

# Contributors

- [Naruki-Ichihara](https://github.com/Naruki-Ichihara)

<!-- CREATED_BY_LEADYOU_README_GENERATOR -->

# References
# Cite