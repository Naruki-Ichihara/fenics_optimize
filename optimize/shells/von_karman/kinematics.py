# -*- coding: utf-8 -*-

from dolfin import *


def e(u, theta):
    r"""Return the membrane strain tensor for the Von-Karman plate model.

    .. math::
        e(u, theta) = \mathrm{sym}\nabla u + \frac{\theta \otimes \theta}{2}

    Args:
        u: In-plane displacement.
        theta: Rotations.

    Returns:
        UFL form of Von-Karman membrane strain tensor.
    """
    return sym(grad(u)) + 0.5*outer(theta, theta)
