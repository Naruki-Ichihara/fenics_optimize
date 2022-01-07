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

## ipopt installation

Clone [ThirdParty-HSL](https://github.com/coin-or-tools/ThirdParty-HSL)

```
git clone https://github.com/coin-or-tools/ThirdParty-HSL.git 
cd ThirdParty-HSL
```

obtain a tarball with HSL source code from [HSL](https://www.hsl.rl.ac.uk/ipopt/).

Then, unpack this tarball using:

```
gunzip coinhsl-x.y.z.tar.gz
tar xf coinhsl-x.y.z.tar
```

Rename the directory `coinhsl-x.y.z` to `coinhsl` and run

```
./configure
```
Run `make` to build and run `make install`. Next, clone [Ipopt](https://github.com/coin-or/Ipopt)

```
git clone https://github.com/coin-or/Ipopt.git
cd Ipopt
```

Then, run `./configure` and `make`.

```
./configure
make
make test
make install
```

## cyipopt installation

Clone [cyipopt](https://github.com/mechmotum/cyipopt)

```
git clone https://github.com/mechmotum/cyipopt.git
cd cyipopt
```

The `ipopt` executble should be discoverable by `pkg-config`.

```
pkg-config --libs --cflags ipopt
```

and run `setup.py` using python3

```
python3 setup.py install
```

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



