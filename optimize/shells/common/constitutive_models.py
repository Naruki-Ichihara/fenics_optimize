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


def psi_M(k, **kwargs):
    r"""Returns bending moment energy density calculated from the curvature k
    using:

    Isotropic case:
    .. math::
    D = \frac{E*t^3}{24(1 - \nu^2)}
    W_m(k, \ldots) = D*((1 - nu)*tr(k**2) + nu*(tr(k))**2)

    Args:
        k: Curvature, typically UFL form with shape (2,2) (tensor).
        **kwargs: Isotropic case:
            E: Young's modulus, Constant or Expression.
            nu: Poisson's ratio, Constant or Expression.
            t: Thickness, Constant or Expression.

    Returns:
        UFL form of bending stress tensor with shape (2,2) (tensor).
    """
    # Isotropic case
    if 'E' in kwargs and 't' in kwargs and 'nu' in kwargs:
        E = kwargs['E']
        t = kwargs['t']
        nu = kwargs['nu']

        D = (E*t**3)/(24.0*(1.0 - nu**2))
        M = D*((1.0 - nu)*tr(k*k) + nu*(tr(k))**2)
    else:
        raise ArgumentError(
            "Invalid set of kwargs to specify bending energy density.")

    return M


def psi_N(e, **kwargs):
    r"""Returns membrane energy density calculated from e using:

    Isotropic case:
    .. math::
    B = \frac{E*t}{2(1 - \nu^2)}
    N(e, \ldots) = B(1 - \nu)e + \nu \mathrm{tr}(e)I

    Args:
        e: Membrane strain, typically UFL form with shape (2,2) (tensor).
        **kwargs: Isotropic case:
            E: Young's modulus, Constant or Expression.
            nu: Poisson's ratio, Constant or Expression.
            t: Thickness, Constant or Expression.

    Returns:
        UFL form of membrane stress tensor with shape (2,2) (tensor).
    """
    # Isotropic case
    if 'E' in kwargs and 't' in kwargs and 'nu' in kwargs:
        E = kwargs['E']
        t = kwargs['t']
        nu = kwargs['nu']

        B = (E*t)/(2.0*(1.0 - nu**2))
        N = B*((1.0 - nu)*tr(e*e) + nu*(tr(e))**2)
    else:
        raise ArgumentError(
            "Invalid set of kwargs to specify membrane stress tensor.")

    return N


def strain_to_voigt(e):
    r"""Returns the pseudo-vector in the Voigt notation associate to a 2x2
    symmetric strain tensor, according to the following rule (see e.g.
    https://en.wikipedia.org/wiki/Voigt_notation),

        .. math::
         e  = \begin{bmatrix} e_{00} & e_{01}\\ e_{01} & e_{11} \end{bmatrix}\quad\to\quad
         e_\mathrm{voigt}= \begin{bmatrix} e_{00} & e_{11}& 2e_{01} \end{bmatrix}

    Args:
        e: a symmetric 2x2 strain tensor, typically UFL form with shape (2,2)

    Returns:
        a UFL form with shape (3,1) corresponding to the input tensor in Voigt
        notation.
    """
    return as_vector((e[0, 0], e[1, 1], 2*e[0, 1]))


def stress_to_voigt(sigma):
    r"""Returns the pseudo-vector in the Voigt notation associate to a 2x2
    symmetric stress tensor, according to the following rule (see e.g.
    https://en.wikipedia.org/wiki/Voigt_notation),

        .. math::
         \sigma  = \begin{bmatrix} \sigma_{00} & \sigma_{01}\\ \sigma_{01} & \sigma_{11} \end{bmatrix}\quad\to\quad
         \sigma_\mathrm{voigt}= \begin{bmatrix} \sigma_{00} & \sigma_{11}& \sigma_{01} \end{bmatrix}

    Args:
        sigma: a symmetric 2x2 stress tensor, typically UFL form with shape
        (2,2).

    Returns:
        a UFL form with shape (3,1) corresponding to the input tensor in Voigt notation.
    """
    return as_vector((sigma[0, 0], sigma[1, 1], sigma[0, 1]))


def strain_from_voigt(e_voigt):
    r"""Inverse operation of strain_to_voigt.

    Args:
        sigma_voigt: UFL form with shape (3,1) corresponding to the strain
        pseudo-vector in Voigt format

    Returns:
        a symmetric stress tensor, typically UFL form with shape (2,2)
    """
    return as_matrix(((e_voigt[0], e_voigt[2]/2.), (e_voigt[2]/2., e_voigt[1])))


def stress_from_voigt(sigma_voigt):
    r"""Inverse operation of stress_to_voigt.

    Args:
        sigma_voigt: UFL form with shape (3,1) corresponding to the stress
        pseudo-vector in Voigt format.

    Returns:
        a symmetric stress tensor, typically UFL form with shape (2,2)
    """
    return as_matrix(((sigma_voigt[0], sigma_voigt[2]), (sigma_voigt[2], sigma_voigt[1])))
