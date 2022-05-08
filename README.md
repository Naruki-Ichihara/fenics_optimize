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

**fenics-optimize** enables reusable and straightforward UFL coding for physical optimization problem.

## Example

## Installation
### Singularity

We recommend using *Singularity* container system. First, install [SingularityCE](https://sylabs.io/singularity) on your host.
Then download our singularity image:
```
export VERSION=0.2.0-alpha
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
