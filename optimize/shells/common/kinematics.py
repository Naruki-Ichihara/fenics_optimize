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


def F(u):
    r"""Return deformation gradient tensor for non-linear plate model.

    Deformation gradient of 2-dimensional manifold embedded in 3-dimensional space.

    .. math::
        F = I + \nabla u

    Args:
        u: displacement field, typically UFL (3,1) coefficient.

    Returns:
        a UFL coeffient with shape (3,2)
    """
    I = as_tensor([[1.0, 0.0],
                   [0.0, 1.0],
                   [0.0, 0.0]])
    F = I + grad(u)
    return F


def e(u):
    r"""Return membrane strain tensor for linear plate model.

    .. math::
         e = \dfrac{1}{2}(\nabla u+\nabla u^T)

    Args:
        u: membrane displacement field, typically UFL (2,1) coefficient. 

    Returns:
        a UFL form with shape (2,2) 
    """
    return sym(grad(u))


def k(theta):
    r"""Return bending curvature tensor for linear plate model.

        .. math::
            k = \dfrac{1}{2}(\nabla \theta+\nabla \theta^T)

    Args:
        theta: rotation  field, typically UFL (2,1) form or a dolfin Function

    Returns:
        a UFL form with shape (2,2) 
    """
    return sym(grad(theta))
