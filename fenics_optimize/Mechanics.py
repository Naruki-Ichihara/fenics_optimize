#! /usr/bin/python3
# -*- coding: utf-8 -*-
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
''' Helper module for machanical engineering problems.

This source based on `fenics-shells` project, from Corrado Maurini.
'''

from dolfin import *
from dolfin_adjoint import *
from ufl import transpose
import numpy as np

def sigma(v, E, nu):
    '''
    Compute stress tensor form.

    Args:
        v (dolfin_adjoint.Function): Displacement vector
        E (float): Yound's modulus
        nu (float): Poisson's ratio

    Returns:
        form (dolfin_adjoint.Form): Stress tensor
    '''
    mu = E/(2.0*(1.0 + nu))
    lmbda = E*nu/((1.0 + nu)*(1.0 - 2.0*nu))
    return 2.0*mu*sym(grad(v)) + lmbda*tr(sym(grad(v)))*Identity(len(v))

def epsilon(v):
    '''
    Compute strain tensor form.

    Args:
        v (dolfin_adjoint.Function): Displacement vector

    Returns:
        form (dolfin_adjoint.Form): Strain tensor
    '''
    return sym(grad(v))

def strain_to_voigt(e):
    '''
    Returns the pseudo-vector in the Voigt notation associate to a 2x2
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
    '''
    return as_vector((e[0, 0], e[1, 1], 2*e[0, 1]))


def stress_to_voigt(sigma):
    '''
    Returns the pseudo-vector in the Voigt notation associate to a 2x2
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
    '''
    return as_vector((sigma[0, 0], sigma[1, 1], sigma[0, 1]))


def strain_from_voigt(e_voigt):
    '''
    Inverse operation of strain_to_voigt.

    Args:
        sigma_voigt: UFL form with shape (3,1) corresponding to the strain
        pseudo-vector in Voigt format

    Returns:
        a symmetric stress tensor, typically UFL form with shape (2,2)
    '''
    return as_matrix(((e_voigt[0], e_voigt[2]/2.), (e_voigt[2]/2., e_voigt[1])))


def stress_from_voigt(sigma_voigt):
    '''
    Inverse operation of stress_to_voigt.

    Args:
        sigma_voigt: UFL form with shape (3,1) corresponding to the stress
        pseudo-vector in Voigt format.

    Returns:
        a symmetric stress tensor, typically UFL form with shape (2,2)
    '''
    return as_matrix(((sigma_voigt[0], sigma_voigt[2]), (sigma_voigt[2], sigma_voigt[1])))

def membrane_energy(e, N):
    '''
    Return internal membrane energy for a plate model.

    Args:
        e: Membrane strains, UFL or DOLFIN Function of rank (2, 2) (tensor).
        N: Membrane stress, UFL or DOLFIN Function of rank (2, 2) (tensor). 

    Returns:
        UFL form of internal elastic membrane energy for a plate model.
    '''
    psi_m = (1.0/2.0)*inner(N, e)
    return psi_m


def membrane_bending_energy(e, k, A, D, B):
    '''
    Return the coupled membrane-bending energy for a plate model.

    Args:
        e: Membrane strains, UFL or DOLFIN Function of rank (2, 2) (tensor).
        k: Curvature, UFL or DOLFIN Function of rank (2, 2) (tensor).
        A: Membrane stresses.
        D: Bending stresses.
        B: Coupled membrane-bending stresses.

    '''
    psi_mb = (1.0/2.0)*inner(A, e) + \
             (1.0/2.0)*inner(D, k) + \
             (1.0/2.0)*inner(B, e)
    return psi_mb

def z_coordinates(hs):
    '''
    Return a list with the thickness coordinate of the top surface of each layer
    taking the midplane as z = 0.

    Args:
        hs: a list giving the thinckesses of each layer
            ordered from bottom (layer - 0) to top (layer n-1).

    Returns:
        z: a list of coordinate of the top surface of each layer
           ordered from bottom (layer - 0) to top (layer n-1)
    '''

    z0 = sum(hs)/2.
    z = [(-sum(hs)/2. + sum(hs for hs in hs[0:i])) for i in range(len(hs)+1)]
    return z


def rotated_lamina_stiffness_inplane(E1, E2, G12, nu12, theta):
    '''
    Return the in-plane stiffness matrix of an orhtropic layer
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
    '''
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
    '''
    Return the shear stiffness matrix of an orhtropic layer
    in a reference rotated by an angle theta wrt to the material one.
    It assumes Voigt notation and plane stress state
    (see Reddy 1997, eqn 3.4.18).

    Args:
        G12: The transverse shear modulus between the material directions 1-2.
        G13: The transverse shear modulus between the material directions 1-3.
        kappa: The shear correction factor.

    Returns:
        Q_shear_theta: a 3x3 symmetric ufl matrix giving the stiffness matrix.
    '''
    # The rotation matrix to rotate the shear stiffness matrix
    # in Voigt notation of an angle theta from the material directions
    # (See Reddy 1997 pg 91, eqn 2.3.7)
    c = cos(theta)
    s = sin(theta)
    T_shear = as_matrix([[c, s], [-s, c]])
    Q_shear = kappa*as_matrix([[G23, 0.], [0., G13]])
    Q_shear_theta = T_shear*Q_shear*transpose(T_shear)

    return Q_shear_theta


def rotated_lamina_stiffness_inplane(E1, E2, G12, nu12, theta):
    '''Return the in-plane stiffness matrix of an orhtropic layer
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

    '''
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
    '''
    Return the stiffness matrix of a kirchhoff-love model of a laminate
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
    '''
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
    '''
    Return the shear stiffness matrix of a Reissner-Midlin model of a
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
    '''
    assert (len(hs) == len(thetas)), "hs and thetas should have the same length !"

    z = z_coordinates(hs)
    F = 0.*Identity(2)

    for i in range(len(thetas)):
        Q_shear_theta = rotated_lamina_stiffness_shear(G13, G23, thetas[i])
        F += Q_shear_theta*(z[i+1]-z[i])

    return F