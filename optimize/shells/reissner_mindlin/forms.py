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
import ufl


def gamma(theta, w):
    r"""Return shear strain vector calculated from primal variables:

    .. math::
        \gamma = \nabla w - \theta
    """
    return grad(w) - theta


def psi_T(gamma, **kwargs):
    r"""Returns transverse shear energy density calculated using:

    Isotropic case:
    .. math::
        \psi_T(\gamma, \ldots) = \frac{E \kappa t}{4(1 + \nu))}\gamma**2

    Args:
        gamma: Shear strain, typically UFL form with shape (2,).
        **kwargs: Isotropic case:
            E: Young's modulus, Constant or Expression.
            nu: Poisson's ratio, Constant or Expression.
            t: Thickness, Constant or Expression.
            kappa: Shear correction factor, Constant or Expression.

    Returns:
        UFL expression of transverse shear stress vector.
    """
    if 'E' in kwargs and 'kappa' in kwargs and 't' in kwargs and 'nu' in kwargs:
        E = kwargs['E']
        kappa = kwargs['kappa']
        t = kwargs['t']
        nu = kwargs['nu']

        T = ((E*kappa*t)/(4.0*(1.0 + nu)))*inner(gamma, gamma)
    else:
        raise ArgumentError(
            "Invalid set of kwargs to specify transverse shear stress vector.")

    return T


def inner_e(x, y, restrict_to_one_side=False, quadrature_degree=1):
    r"""The inner product of the tangential component of a vector field on all
    of the facets of the mesh (Measure objects dS and ds).

    By default, restrict_to_one_side is False. In this case, the function will
    return an integral that is restricted to both sides ('+') and ('-') of a
    shared facet between elements. You should use this in the case that you
    want to use the 'projected' version of DuranLibermanSpace.

    If restrict_to_one_side is True, then this will return an integral that is
    restricted ('+') to one side of a shared facet between elements. You should
    use this in the case that you want to use the `multipliers` version of
    DuranLibermanSpace.

    Args:
        x: DOLFIN or UFL Function of rank (2,) (vector).
        y: DOLFIN or UFL Function of rank (2,) (vector).
        restrict_to_one_side (Optional[bool]: Default is False.
        quadrature_degree (Optional[int]): Default is 1.

    Returns:
        UFL Form.
    """
    dSp = Measure('dS', metadata={'quadrature_degree': quadrature_degree})
    dsp = Measure('ds', metadata={'quadrature_degree': quadrature_degree})
    n = ufl.geometry.FacetNormal(x.ufl_domain())
    t = as_vector((-n[1], n[0]))
    a = (inner(x, t)*inner(y, t))('+')*dSp + \
        (inner(x, t)*inner(y, t))*dsp
    if not restrict_to_one_side:
        a += (inner(x, t)*inner(y, t))('-')*dSp
    return a
