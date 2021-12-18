# morphogenesis

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

# Minimal Example

To be..

# Contributors

- [Naruki-Ichihara](https://github.com/Naruki-Ichihara)

<!-- CREATED_BY_LEADYOU_README_GENERATOR -->