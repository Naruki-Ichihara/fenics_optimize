<h1 align=center>fenics-optimize</a></h1>

<!-- # Short Description -->
<!-- # Badges -->

[![Github issues](https://img.shields.io/github/issues/Naruki-Ichihara/fenics_optimize?style=for-the-badge&logo=appveyor)](https://github.com/Naruki-Ichihara/fenics-optimize/issues)
[![Github forks](https://img.shields.io/github/forks/Naruki-Ichihara/fenics_optimize?style=for-the-badge&logo=appveyor)](https://github.com/Naruki-Ichihara/fenics-optimize/network/members)
[![Github stars](https://img.shields.io/github/stars/Naruki-Ichihara/fenics_optimize?style=for-the-badge&logo=appveyor)](https://github.com/Naruki-Ichihara/fenics-optimize/stargazers)
[![Github top language](https://img.shields.io/github/languages/top/Naruki-Ichihara/fenics_optimize?style=for-the-badge&logo=appveyor)](https://github.com/Naruki-Ichihara/fenics-optimize/)
[![Github license](https://img.shields.io/github/license/Naruki-Ichihara/fenics_optimize?style=for-the-badge&logo=appveyor)](https://github.com/Naruki-Ichihara/fenics-optimize/)

## Motivation and significance
**fenics-optimize** is an add-on module of the [FEniCS computing platform](https://fenicsproject.org/) for multiphysics optimization problems. 
This module enables reusable and straightforward UFL coding for physical optimization problem.

>This repository is the third-party module of fenics project.

## Example

```python
def k(a):
    return eps + (1 - eps) * a ** p

mesh = UnitSquareMesh(n, n)
X = FunctionSpace(mesh, 'CG', 1)
f = interpolate(Constant(1e-2), X)

class Initial_density(UserExpression):
    def eval(self, value, x):
        value[0] = m

class Left(SubDomain):
    def inside(self, x, on_boundary):
        gamma = 1/n + eps
        return x[0] == 0.0 and on_boundary

U = FunctionSpace(mesh, 'CG', 1)
t = TrialFunction(U)
dt = TestFunction(U)
bc = DirichletBC(U, Constant(0.0), Left())

class PoissonProblem(op.Module):
    def problem(self, controls):
        rho = controls[0]
        rho = op.helmholtzFilter(controls[0], X, R=R)
        a = inner(grad(t), k(rho)*grad(dt))*dx
        L = f*dt*dx
        A, b = assemble_system(a, L, bc)
        T = Function(U, name='Temperture')
        T = op.amg(A, b, T, False)
        J = assemble(inner(grad(T), k(rho)*grad(T))*dx)
        rho_bulk = project(Constant(1.0), X)
        rho_0 = assemble(rho_bulk*dx)
        rho_total = assemble(controls[0]*dx)
        rel = rho_total/rho_0
        self.volumeFraction = rel
        return J
    def constraint_volume(self):
        return self.volumeFraction - m

x0 = Function(X)
x0.interpolate(Initial_density())
N = Function(X).vector().size()
min_bounds = np.zeros(N)
max_bounds = np.ones(N)

setting = {'set_lower_bounds': min_bounds,
           'set_upper_bounds': max_bounds,
           'set_maxeval': 100
          }
params = {'verbosity': 5}

problem = PoissonProblem()
solution = interface_nlopt(problem, [x0], [0], setting, params)
```

## Installation
### Singularity

We recommend using *Singularity* container system. First, install [SingularityCE](https://sylabs.io/singularity) on your host.
Then download our singularity image:
```
singularity pull fenics_optimize.sif docker://ichiharanaruki/fenics_optimize:latest
```
then, move into singularity shell
```
singularity shell fenics_optimize.sif
```

### install on your local host
First, make sure to install the following dependencies:

* [FEniCS](https://fenicsproject.org/) + [pyadjoint](https://github.com/dolfin-adjoint/pyadjoint)
* [fecr](https://github.com/IvanYashchuk/fecr)
* [nlopt](https://github.com/stevengj/nlopt/)

Second, install using pip.
```
pip install git+https://github.com/Naruki-Ichihara/fenics_optimize.git@main
```

[Installation page](https://github.com/Naruki-Ichihara/fenics_optimize/blob/main/INSTALL.md) details installation instruction on your local system.

## Contributors

- [Naruki-Ichihara](https://github.com/Naruki-Ichihara)

## API and documentation
API: [API](https://naruki-ichihara.github.io/fenics_optimize/)

Documentation: Docs

## References
## Cite
