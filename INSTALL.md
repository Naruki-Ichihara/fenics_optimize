# Step-by-step installation on your local host

## FEniCS installation
Run the following commands on Ubuntu terminal:

```
apt-get install software-properties-common
add-apt-repository ppa:fenics-packages/fenics
apt-get update
apt-get install fenics
```

Then, use pip install command to install pyadjoint:

```
pip install git+https://github.com/dolfin-adjoint/pyadjoint.git@master
```

Now, complete the fenics installation.

## nlopt installation

NLopt is compiled and installed with the CMake build system.

```
git clone https://github.com/stevengj/nlopt.git
cd nlopt
mkdir build && cd build
```

run cmake with python executable option

```
cmake -DPYTHON_EXECUTABLE=/usr/bin/python3 ..
```

and run `make` and `make install`.

## fecr installation

Using `pip` to install

```
pip install git+https://github.com/IvanYashchuk/firedrake-numpy-adjoint@master
```




