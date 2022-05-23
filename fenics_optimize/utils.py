from dolfin import *
from dolfin_adjoint import *
# TO DO
def to_numpy(function):
    return function.vector().get_local()

def from_numpy(array, function):
    return function.vector().set_local(array)
