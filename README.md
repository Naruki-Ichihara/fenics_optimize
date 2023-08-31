<div align="center">
<img src="https://user-images.githubusercontent.com/70839257/222454118-51044536-a213-41f3-9863-8bfbaf033ad6.jpg" alt="logo" width="50%"></img>
</div>

# opt *f*(x)
<!-- # Short Description -->
<!-- # Badges -->

[![Github issues](https://img.shields.io/github/issues/Naruki-Ichihara/fenics_optimize?style=for-the-badge&logo=appveyor)](https://github.com/Naruki-Ichihara/fenics-optimize/issues)
[![Github forks](https://img.shields.io/github/forks/Naruki-Ichihara/fenics_optimize?style=for-the-badge&logo=appveyor)](https://github.com/Naruki-Ichihara/fenics-optimize/network/members)
[![Github stars](https://img.shields.io/github/stars/Naruki-Ichihara/fenics_optimize?style=for-the-badge&logo=appveyor)](https://github.com/Naruki-Ichihara/fenics-optimize/stargazers)
[![Github top language](https://img.shields.io/github/languages/top/Naruki-Ichihara/fenics_optimize?style=for-the-badge&logo=appveyor)](https://github.com/Naruki-Ichihara/fenics-optimize/)
[![Github license](https://img.shields.io/github/license/Naruki-Ichihara/fenics_optimize?style=for-the-badge&logo=appveyor)](https://github.com/Naruki-Ichihara/fenics-optimize/)

## Motivation and significance
**optfx** is a library for the large scale field optimization platform based on [FEniCS ecosystem](https://fenicsproject.org/).
This module enables reusable and straightforward UFL coding for inverse problem of multi-physics.

>This software is the third-party module of fenics project.

>This software is based on Lagacy FEniCS (FEniCS2019.1.0). The new version, FEniCSx, is not supported.

>This README and documentation is now under construction.

## Getting started

```python
import numpy as np
import optfx as opt

opt.parameters["form_compiler"]["optimize"] = True
opt.parameters["form_compiler"]["cpp_optimize"] = True
opt.parameters['form_compiler']['quadrature_degree'] = 5

comm = opt.MPI.comm_world
m = 0.30   # Target rate of the material amount
p = 5      # Penalty parameter
eps = 1e-3 # Material lower bound
R = 0.01   # Helmholtz filter radius
n = 256    # Resolution

def k(a):
    return eps + (1 - eps) * a ** p

mesh = opt.UnitSquareMesh(comm, n, n)
X = opt.FunctionSpace(mesh, 'CG', 1)
f = opt.interpolate(opt.Constant(1e-2), X)

class Initial_density(opt.UserExpression):
    def eval(self, value, x):
        value[0] = m

class Left(opt.SubDomain):
    def inside(self, x, on_boundary):
        gamma = 1/250 + 1e-5
        return x[0] == 0.0 and 0.5 - gamma < x[1] < 0.5 + gamma and on_boundary

U = opt.FunctionSpace(mesh, 'CG', 1)
t = opt.TrialFunction(U)
dt = opt.TestFunction(U)
bc = opt.DirichletBC(U, opt.Constant(0.0), Left())

class PoissonProblem(opt.Module):
    def problem(self, controls):
        rho = controls[0]
        rho = opt.helmholtzFilter(rho, X, R)
        a = opt.inner(opt.grad(t), k(rho)*opt.grad(dt))*opt.dx
        L = opt.inner(f, dt)*opt.dx
        Th = opt.Function(U, name='Temperture')
        opt.solve(a==L, Th, bc)
        J = opt.assemble(opt.inner(opt.grad(Th), k(rho)*opt.grad(Th))*opt.dx)
        rho_bulk = opt.project(opt.Constant(1.0), X)
        rho_0 = opt.assemble(rho_bulk*opt.dx)
        rho_total = opt.assemble(controls[0]*opt.dx)
        rel = rho_total/rho_0
        self.volumeFraction = rel
        return J
    def constraint_volume(self):
        return self.volumeFraction - m

x0 = opt.Function(X)
x0.interpolate(Initial_density())
N = opt.Function(X).vector().size()
min_bounds = np.zeros(N)
max_bounds = np.ones(N)

setting = {'set_lower_bounds': min_bounds,
           'set_upper_bounds': max_bounds,
           'set_maxeval': 1000
          }
params = {'verbosity': 1}

problem = PoissonProblem()
solution = opt.optimize(problem, [x0], {'constraint_volume': [0]}, setting, params)
```

## Installation
### Docker
We recomennd to use our docker image. First you should install docker.io or relavant systems. Our docker image was stored in the [dockerhub](https://hub.docker.com/repository/docker/ichiharanaruki/optfx/general).

If you use the docker.io and docker-compose, the following command pull and run the above image.
```bash
docker-compose up
```

### install on your local host
First, make sure to install the following dependencies:

* [FEniCS2019.1.0](https://fenicsproject.org/) + [pyadjoint](https://github.com/dolfin-adjoint/pyadjoint)
* [nlopt](https://github.com/stevengj/nlopt/)

Second, install using pip.
```
pip install git+https://github.com/Naruki-Ichihara/fenics_optimize.git@main
```

## Contributors

- [Naruki-Ichihara](https://github.com/Naruki-Ichihara)

## API and documentation
API: [API](https://naruki-ichihara.github.io/fenics_optimize/)
Documentation: Docs

## References
## Cite
To cite this repository:

```
@software{fenics_optimize2023github,
  author = {Naruki Ichihara},
  title = {{optfx}: Large scale field optimization platform based on fenics in python},
  url = {https://github.com/Naruki-Ichihara/fenics_optimize},
  version = {0.3.0},
  year = {2023},
}
```
---
<div align="center">
<img src="https://user-images.githubusercontent.com/70839257/222695855-e3fbb18a-f857-47da-b900-00b1b60bb7ba.jpg" alt="logo" width="10%"></img>
</div>

