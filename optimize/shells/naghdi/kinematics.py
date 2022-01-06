# -*- coding: utf-8 -*-

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


def d(theta):
    r"""Director vector.

    .. math::
        d = \left\lbrace \sin(\theta_2)\cos(\theta_1),
                         -\sin(\theta_1),
                         \cos(\theta_2)\cos(\theta_1)
            \right\rbrace^T

    Args:
        Rotation vector

    Returns:
        UFL expression of director vector.
    """
    d = as_vector([sin(theta[1])*cos(theta[0]),
                   -sin(theta[0]),
                   cos(theta[1])*cos(theta[0])])
    return d


def G(F):
    r"""Returns the stretching tensor (1st non-linear Naghdi strain measure).

    .. math::
        G = \frac{1}{2}(F^{T}F - I)

    Args:
        F: Deformation gradient.

    Returns:
        UFL expression of stretching tensor. 
    """
    G = 0.5*(F.T*F - Identity(2))
    return G


def K(F, d):
    r"""Returns the curvature tensor (2nd non-linear Naghdi strain measure).

    .. math::
        K = \frac{1}{2}(F^{T}\nabla d + (\nabla d)^T F^{T})

    Args:
        F: Deformation gradient.
        d: Director vector.

    Returns:
        UFL expression of curvature tensor.
    """
    K = 0.5*(F.T*grad(d) + grad(d).T*F)
    return K


def g(F, d):
    r"""Returns the shear strain vector (3rd non-linear Naghdi strain measure).

    .. math::
        g = F^{T}d

    Args:
        F: Deformation gradient.
        d: Director vector.

    Returns:
        UFL expression of shear strain vector.
    """
    g = F.T*d
    return g
