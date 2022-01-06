# Copyright (C) 2015 Jack S. Hale
#
# This file is part of fenics-shells.
#
# fenics-shells is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# fenics-shells is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with fenics-shells. If not, see <http://www.gnu.org/licenses/>.

from dolfin import *


def membrane_energy(e, N):
    r"""Return internal membrane energy for a plate model.

    Args:
        e: Membrane strains, UFL or DOLFIN Function of rank (2, 2) (tensor).
        N: Membrane stress, UFL or DOLFIN Function of rank (2, 2) (tensor). 

    Returns:
        UFL form of internal elastic membrane energy for a plate model.
    """
    psi_m = (1.0/2.0)*inner(N, e)
    return psi_m


def membrane_bending_energy(e, k, A, D, B):
    r"""Return the coupled membrane-bending energy for a plate model.

    Args:
        e: Membrane strains, UFL or DOLFIN Function of rank (2, 2) (tensor).
        k: Curvature, UFL or DOLFIN Function of rank (2, 2) (tensor).
        A: Membrane stresses.
        D: Bending stresses.
        B: Coupled membrane-bending stresses.

    """
    psi_mb = (1.0/2.0)*inner(A, e) + \
             (1.0/2.0)*inner(D, k) + \
             (1.0/2.0)*inner(B, e)
    return psi_mb
