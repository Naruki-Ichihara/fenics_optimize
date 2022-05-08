<h3><a href="https://fenics-optimize.naruki-ichihara.com/"> Documents </a> | <a href="https://naruki-ichihara.github.io/fenics_optimize/"> API </a></h3></div>

# fenics-optimize
<!-- # Short Description -->

**fenics-optimize** is an add-on module of the [FEniCS computing platform](https://fenicsproject.org/) for multiphysics optimization problems. 

<!-- # Badges -->

[![Github issues](https://img.shields.io/github/issues/Naruki-Ichihara/fenics_optimize?style=for-the-badge&logo=appveyor)](https://github.com/Naruki-Ichihara/fenics-optimize/issues)
[![Github forks](https://img.shields.io/github/forks/Naruki-Ichihara/fenics_optimize?style=for-the-badge&logo=appveyor)](https://github.com/Naruki-Ichihara/fenics-optimize/network/members)
[![Github stars](https://img.shields.io/github/stars/Naruki-Ichihara/fenics_optimize?style=for-the-badge&logo=appveyor)](https://github.com/Naruki-Ichihara/fenics-optimize/stargazers)
[![Github top language](https://img.shields.io/github/languages/top/Naruki-Ichihara/fenics_optimize?style=for-the-badge&logo=appveyor)](https://github.com/Naruki-Ichihara/fenics-optimize/)
[![Github license](https://img.shields.io/github/license/Naruki-Ichihara/fenics_optimize?style=for-the-badge&logo=appveyor)](https://github.com/Naruki-Ichihara/fenics-optimize/)

## Motivation and significance

<p align="center">
  <img src="https://user-images.githubusercontent.com/70839257/148230717-e25da51a-3f96-461d-960f-8f22381387fc.png" width="600"/>
</p>

**fenics-optimize** enables reusable and straightforward UFL coding for physical optimization problems and provides decorators that bridge easily between a fenics calculation chain and optimizers.

## Example
2-D Poisson topology optimization problem.

First, import `fenics_optimize` with `dolfin`, `dolfin-adjoint` and `numpy`.
```python
from dolfin import *
from dolfin_adjoint import *
import fenics_optimize as op
import numpy as np
```

Then define the pysical model using the FEniCS with decorator `with_derivative`.

`with_derivative` will wrrap the fenics function to the numpy function and compute Jacobian automatically.

```python
@op.with_derivative([X])
def forward(xs):
    rho = op.helmholtzFilter(xs[0], X, R=r)
    Th = Function(X, name='Temperature')
    u = TrialFunction(X)
    v = TestFunction(X)
    a = inner(grad(u), k(rho)*grad(v))*dx
    L = f*v*dx
    A, b = assemble_system(a, L, bcs)
    Th = op.AMGsolver(A, b).solve(Th, X)
    cost = assemble(inner(grad(Th), k(rho)*grad(Th))*dx)
    return cost

@op.with_derivative([X])
def constraint(xs):
    rho_bulk = project(Constant(1.0), X)
    rho_0 = assemble(rho_bulk*dx)
    rho_f = assemble(xs[0]*dx)
    rel = rho_f/rho_0
    return rel - V
```

Finally, optimize the cost function with inequality constraints.

```python
op.MMAoptimize(problemSize, x0, forward, [constraint], [0], maxeval=100, bounds=[0, 1], rel=1e-20)
```

## Installation
### Singularity

We recommend using *Singularity* container system. First, install [SingularityCE](https://sylabs.io/singularity) on your host.
Then download our singularity image:
```
export VERSION=0.1.1-alpha
wget https://github.com/Naruki-Ichihara/fenics_optimize/releases/download/v${VERSION}/fenics-optimize.sif
```
then, move into singularity shell
```
singularity shell fenics-optimize.sif
```

### install on your local host
First, make sure to install the following dependencies:

* [FEniCS](https://fenicsproject.org/) + [pyadjoint](https://github.com/dolfin-adjoint/pyadjoint)
* [fecr](https://github.com/IvanYashchuk/fecr)
* [cyipopt](https://github.com/mechmotum/cyipopt)
* [nlopt](https://github.com/stevengj/nlopt/)

Second, install using pip.
```
pip install git+https://github.com/Naruki-Ichihara/fenics_optimize.git@main
```

[Installation page](https://github.com/Naruki-Ichihara/fenics_optimize/blob/main/INSTALL.md) details installation instruction on your local system.

## Contributors

- [Naruki-Ichihara](https://github.com/Naruki-Ichihara)

## References
## Cite
