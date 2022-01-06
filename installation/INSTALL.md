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
Finally, run `make` to build and run `make install`.

## cyipopt installation

Clone [cyipopt](https://github.com/mechmotum/cyipopt)

```
git clone git@github.com:mechmotum/cyipopt.git
cd cyipopt
```

and run `setup.py` using python3

```
python3 setup.py install
```

## nlopt installation

To be.

