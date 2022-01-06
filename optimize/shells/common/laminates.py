# Copyright (C) 2015 Corrado Maurini
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

import numpy as np
import math as m

from dolfin import *
from ufl import transpose


def z_coordinates(hs):
    r"""Return a list with the thickness coordinate of the top surface of each layer
    taking the midplane as z = 0.

    Args:
        hs: a list giving the thinckesses of each layer
            ordered from bottom (layer - 0) to top (layer n-1).

    Returns:
        z: a list of coordinate of the top surface of each layer
           ordered from bottom (layer - 0) to top (layer n-1)
    """

    z0 = sum(hs)/2.
    #z = [(sum(hs)/2.- sum(hs for hs in hs[0:i])) for i in range(len(hs)+1)];
    z = [(-sum(hs)/2. + sum(hs for hs in hs[0:i])) for i in range(len(hs)+1)]
    return z


def rotated_lamina_stiffness_inplane(E1, E2, G12, nu12, theta):
    r"""Return the in-plane stiffness matrix of an orhtropic layer
    in a reference rotated by an angle theta wrt to the material one.
    It assumes Voigt notation and plane stress state.
    (see Reddy 1997, eqn 1.3.71)

    Args:
        E1: The Young modulus in the material direction 1.
        E2: The Young modulus in the material direction 2.
        G23: The in-plane shear modulus.
        nu12: The in-plane Poisson ratio.
        theta: The rotation angle from the material to the desired reference system.

    Returns:
        Q_theta: a 3x3 symmetric ufl matrix giving the stiffness matrix.
    """
    # Rotation matrix to rotate the in-plane stiffness matrix
    # in Voigt notation of an angle theta from the material directions
    # (See Reddy 1997 pg 91, eqn 2.3.7)

    c = cos(theta)
    s = sin(theta)
    T = as_matrix([[c**2, s**2, -2*s*c],
                   [s**2, c**2, 2*s*c],
                   [c*s, -c*s, c**2 - s**2]])
    # In-plane stiffness matrix of an orhtropic layer in material coordinates
    nu21 = E2/E1*nu12
    Q11 = E1/(1-nu12*nu21)
    Q12 = nu12*E2/(1-nu12*nu21)
    Q22 = E2/(1-nu12*nu21)
    Q66 = G12
    Q16 = 0.
    Q26 = 0.
    Q = as_matrix([[Q11, Q12, Q16],
                   [Q12, Q22, Q26],
                   [Q16, Q26, Q66]])
    # Rotated matrix in the main reference
    Q_theta = T*Q*transpose(T)

    return Q_theta


def rotated_lamina_stiffness_shear(G13, G23, theta, kappa=5./6.):
    r"""Return the shear stiffness matrix of an orhtropic layer
    in a reference rotated by an angle theta wrt to the material one.
    It assumes Voigt notation and plane stress state
    (see Reddy 1997, eqn 3.4.18).

    Args:
        G12: The transverse shear modulus between the material directions 1-2.
        G13: The transverse shear modulus between the material directions 1-3.
        kappa: The shear correction factor.

    Returns:
        Q_shear_theta: a 3x3 symmetric ufl matrix giving the stiffness matrix.
    """
    # The rotation matrix to rotate the shear stiffness matrix
    # in Voigt notation of an angle theta from the material directions
    # (See Reddy 1997 pg 91, eqn 2.3.7)
    c = m.cos(theta)
    s = m.sin(theta)
    T_shear = as_matrix([[c, s], [-s, c]])
    Q_shear = kappa*as_matrix([[G23, 0.], [0., G13]])
    Q_shear_theta = T_shear*Q_shear*transpose(T_shear)

    return Q_shear_theta


def rotated_lamina_stiffness_inplane(E1, E2, G12, nu12, theta):
    r"""Return the in-plane stiffness matrix of an orhtropic layer
    in a reference rotated by an angle theta wrt to the material one.
    It assumes Voigt notation and plane stress state.
    (See Reddy 1997, eqn 1.3.71)

    Args:
        E1: The Young modulus in the material direction 1.
        E2: The Young modulus in the material direction 2.
        G23: The in-plane shear modulus
        nu12: The in-plane Poisson ratio
        theta: The rotation angle from the material to the desired refence system

    Returns:
        Q_theta: a 3x3 symmetric ufl matrix giving the stiffness matrix

    """
    # Rotation matrix to rotate the in-plane stiffness matrix
    # in Voigt notation of an angle theta from the material directions
    # (See Reddy 1997 pg 91, eqn 2.3.7)

    c = cos(theta)
    s = sin(theta)
    T = as_matrix([[c**2, s**2, -2*s*c],
                   [s**2, c**2, 2*s*c],
                   [c*s, -c*s, c**2-s**2]])

    # In-plane stiffness matrix of an orhtropic layer in material coordinates
    nu21 = E2/E1*nu12
    Q11 = E1/(1-nu12*nu21)
    Q12 = nu12*E2/(1-nu12*nu21)
    Q22 = E2/(1-nu12*nu21)
    Q66 = G12
    Q16 = 0.
    Q26 = 0.
    Q = as_matrix([[Q11, Q12, Q16],
                   [Q12, Q22, Q26],
                   [Q16, Q26, Q66]])
    # Rotated matrix in the main reference
    Q_theta = T*Q*transpose(T)

    return Q_theta


def ABD(E1, E2, G12, nu12, hs, thetas):
    r"""Return the stiffness matrix of a kirchhoff-love model of a laminate
    obtained by stacking n orthotropic laminae with possibly different
    thinknesses and orientations (see Reddy 1997, eqn 1.3.71).

    It assumes a plane-stress state.

    Args:
        E1 : The Young modulus in the material direction 1.
        E2 : The Young modulus in the material direction 2.
        G12 : The in-plane shear modulus.
        nu12: The in-plane Poisson ratio.
        hs: a list with length n with the thicknesses of the layers (from top to bottom).
        theta: a list with the n orientations (in radians) of the layers (from top to bottom).

    Returns:
        A: a symmetric 3x3 ufl matrix giving the membrane stiffness in Voigt notation.
        B: a symmetric 3x3 ufl matrix giving the membrane/bending coupling stiffness in Voigt notation.
        D: a symmetric 3x3 ufl matrix giving the bending stiffness in Voigt notation.
    """
    assert (len(hs) == len(thetas)), "hs and thetas should have the same length !"

    z = z_coordinates(hs)
    A = 0.*Identity(3)
    B = 0.*Identity(3)
    D = 0.*Identity(3)

    for i in range(len(thetas)):
        Qbar = rotated_lamina_stiffness_inplane(E1, E2, G12, nu12, thetas[i])
        A += Qbar*(z[i+1]-z[i])
        B += .5*Qbar*(z[i+1]**2-z[i]**2)
        D += 1./3.*Qbar*(z[i+1]**3-z[i]**3)

    return (A, B, D)


def F(G13, G23, hs, thetas):
    r"""Return the shear stiffness matrix of a Reissner-Midlin model of a
    laminate obtained by stacking n orthotropic laminae with possibly different
    thinknesses and orientations.  (See Reddy 1997, eqn 3.4.18)

    It assumes a plane-stress state.

    Args:
        G13: The transverse shear modulus between the material directions 1-3.
        G23: The transverse shear modulus between the material directions 2-3.
        hs: a list with length n with the thicknesses of the layers (from top to bottom).
        theta: a list with the n orientations (in radians) of the layers (from top to bottom).

    Returns:
        F: a symmetric 2x2 ufl matrix giving the shear stiffness in Voigt notation.
    """
    assert (len(hs) == len(thetas)), "hs and thetas should have the same length !"

    z = z_coordinates(hs)
    F = 0.*Identity(2)

    for i in range(len(thetas)):
        Q_shear_theta = rotated_lamina_stiffness_shear(G13, G23, thetas[i])
        F += Q_shear_theta*(z[i+1]-z[i])

    return F


def rotated_lamina_expansion_inplane(alpha11, alpha22, theta):
    r"""Return the in-plane expansion matrix of an orhtropic layer
    in a reference rotated by an angle theta wrt to the material one.
    It assumes Voigt notation and plane stress state.
    (See Reddy 1997, eqn 1.3.71)

    Args:
        alpha11: Expansion coefficient in the material direction 1.
        alpha22: Expansion coefficient in the material direction 2.
        theta: The rotation angle from the material to the desired reference system.

    Returns:
        alpha_theta: a 3x1 ufl vector giving the expansion matrix in voigt notation.
    """
    # Rotated matrix, assuming alpha12 = 0
    c = cos(theta)
    s = sin(theta)
    alpha_xx = alpha11*c**2 + alpha22*s**2
    alpha_yy = alpha11*s**2 + alpha22*c**2
    alpha_xy = 2*(alpha11-alpha22)*s*c
    alpha_theta = as_vector([alpha_xx, alpha_yy, alpha_xy])

    return alpha_theta


def NM_T(E1, E2, G12, nu12, hs, thetas, DeltaT_0, DeltaT_1=0., alpha1=1., alpha2=1.):
    r"""Return the thermal stress and moment resultant of a Kirchhoff-Love model
    of a laminate obtained by stacking n orthotropic laminae with possibly
    different thinknesses and orientations.

    It assumes a plane-stress states and a temperature distribution in the from

    Delta(z) = DeltaT_0 + z * DeltaT_1

    Args:
        E1: The Young modulus in the material direction 1.
        E2: The Young modulus in the material direction 2.
        G12: The in-plane shear modulus.
        nu12: The in-plane Poisson ratio.
        hs: a list with length n with the thicknesses of the layers (from top to bottom).
        theta: a list with the n orientations (in radians) of the layers (from top to bottom).
        alpha1: Expansion coefficient in the material direction 1.
        alpha2: Expansion coefficient in the material direction 2.
        DeltaT_0: Average temperature field.
        DeltaT_1: Gradient of the temperature field.

    Returns:
        N_T: a 3x1 ufl vector giving the membrane inelastic stress.
        M_T: a 3x1 ufl vector giving the bending inelastic stress.
    """
    assert (len(hs) == len(thetas)), "hs and thetas should have the same length !"
    # Coordinates of the interfaces
    z = z_coordinates(hs)

    # Initialize to zero the voigt (ufl) vectors
    N_T = as_vector((0., 0., 0.))
    M_T = as_vector((0., 0., 0.))

    T0 = DeltaT_0
    T1 = DeltaT_1

    # loop over the layers to add the different contributions
    for i in range(len(thetas)):
        # Rotated stiffness
        Q_theta = rotated_lamina_stiffness_inplane(
            E1, E2, G12, nu12, thetas[i])
        alpha_theta = rotated_lamina_expansion_inplane(
            alpha1, alpha2, thetas[i])
        # numerical integration in the i-th layer
        z0i = (z[i+1] + z[i])/2  # Midplane of the ith layer
        # integral of DeltaT(z) in (z[i+1], z[i])
        integral_DeltaT = hs[i]*(T0 + T1 * z0i)
        # integral of DeltaT(z)*z in (z[i+1], z[i])
        integral_DeltaT_z = T1*(hs[i]*3/12 + z0i**2*hs[i]) + hs[i]*z0i*T0
        N_T += Q_theta*alpha_theta*integral_DeltaT
        M_T += Q_theta*alpha_theta*integral_DeltaT_z

    return (N_T, M_T)
