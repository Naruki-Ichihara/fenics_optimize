<div align="center"><img src="https://user-images.githubusercontent.com/70839257/146679821-86686362-c6a0-4b04-a52a-ad4d04dbbff4.png" width="400"/></div>

# morphogenesis
<!-- # Short Description -->
## *Language for Topology Optimization*

The **morphogenesis** is the domain-specific language for topology optimization researches. This repository depends on the [FEniCS computing platform](https://fenicsproject.org/).
<!-- # Badges -->

[![Github issues](https://img.shields.io/github/issues/Naruki-Ichihara/morphogenesis?style=for-the-badge&logo=appveyor)](https://github.com/Naruki-Ichihara/morphogenesis/issues)
[![Github forks](https://img.shields.io/github/forks/Naruki-Ichihara/morphogenesis?style=for-the-badge&logo=appveyor)](https://github.com/Naruki-Ichihara/morphogenesis/network/members)
[![Github stars](https://img.shields.io/github/stars/Naruki-Ichihara/morphogenesis?style=for-the-badge&logo=appveyor)](https://github.com/Naruki-Ichihara/morphogenesis/stargazers)
[![Github top language](https://img.shields.io/github/languages/top/Naruki-Ichihara/morphogenesis?style=for-the-badge&logo=appveyor)](https://github.com/Naruki-Ichihara/morphogenesis/)
[![Github license](https://img.shields.io/github/license/Naruki-Ichihara/morphogenesis?style=for-the-badge&logo=appveyor)](https://github.com/Naruki-Ichihara/morphogenesis/)

## Advantages

### Automatic sensitivity analysis
Sensitivities that need to optimization will be derived automatically from the UFL form. `evalGradient` method receive the assemble of the [UFL](https://github.com/FEniCS/ufl) form as the cost function and compute the Jacobian sequence with [pyadjoint](https://github.com/dolfin-adjoint/pyadjoint) backend.

### Filters
**morphogenesis** has built-in filters for topology optimizations

* Heviside Filter
* Helmholtz Filter
* Isoparametric Filter (TODO)

### Built-in solvers
**morphogenesis** contains the tune-up LU solvers and Krylov solvers for large-scale partial differential equation (PDE) using the [PETSc](https://petsc.org/release/) backend, including:

* Smoothed Aggregation Algebaric Multigrid method (AMG)
* SuperLU_dist
* Mumps
* AmgX (Limited)

### Optimizer
**morphogenesis** supports some optimizers based on NLopt or IPOPT. Currently supported

* Method of Moving Asymptotes (MMA)
* ipopt-HSL that supported MPI (TODO)

### Parallelization (Partially)
**morphogenesis** supports the message passing interface (MPI) partially. 

### CUDA-based solver (Very limited)
**morphogenesis** supports cuda-based AMG solver, [AmgX](https://github.com/NVIDIA/AMGX). 

## Installation

This repository depends following libraries;

* FEniCS project
* pyadjoint
* NLopt
* Numpy
* fecr
* AmgX

We highly recommend you use our docker image.
### Docker
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

### Manually installation
Run `pip` after installation all dependecies.
```
git clone https://github.com/Naruki-Ichihara/morphogenesis.git
pip install .
```

## Example
Here, we select the AMG solver to solve the 2-D elastic topology optimization problem.

```python
from dolfin import *
from dolfin_adjoint import *
from morphogenesis.Chain import numpy2fenics, evalGradient
from morphogenesis.Solvers import AMG2Dsolver
from morphogenesis.Filters import helmholtzFilter, hevisideFilter
from morphogenesis.FileIO import export_result
from morphogenesis.Elasticity import reducedSigma, epsilon
from morphogenesis.Optimizer import MMAoptimize
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
    rho = hevisideFilter(helmholtzFilter(x_, X))
    export_result(project(rho, FunctionSpace(mesh, 'DG', 0)), 'result/test.xdmf')
    facets = MeshFunction('size_t', mesh, 1)
    facets.set_all(0)
    bottom = Bottom()
    bottom.mark(facets, 1)
    ds = Measure('ds', subdomain_data=facets)
    f = Constant((0, -1e3))
    u = TrialFunction(V)
    v = TestFunction(V)
    a = inner(reducedSigma(rho, u, E, nu, p), epsilon(v))*dx
    L = inner(f, v)*ds(1)
    bc = DirichletBC(V, Constant((0, 0)), clamped_boundary)
    u_ = Function(V)
    solver = AMG2Dsolver(a, L, [bc])
    uh = solver.forwardSolve(u_, V, False)
    J = assemble(inner(reducedSigma(rho, uh, E, nu, p), epsilon(uh))*dx)
    dJdx = evalGradient(J, x_)
    grad[:] = dJdx
    print('Cost : {}'.format(J))
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

## Contributors

- [Naruki-Ichihara](https://github.com/Naruki-Ichihara)

## References
## Cite
