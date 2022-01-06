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

"""fenics-shells is an open-source library that provides a wide range of thin
structural models (beams, plates and shells) expressed in the Unified Form
Language (UFL) of the FEniCS Project."""

__version__ = "0.1.0-alpha"

import dolfin as df
# We use DG everywhere, turn this on!
df.parameters["ghost_mode"] = "shared_facet"

from .common.constitutive_models import psi_M, psi_N
from .common.constitutive_models import stress_to_voigt, strain_to_voigt
from .common.constitutive_models import stress_from_voigt, strain_from_voigt
from .common.kinematics import e, k, F
from .common.energy import membrane_bending_energy
from .common.laminates import rotated_lamina_stiffness_inplane, rotated_lamina_stiffness_shear, ABD, rotated_lamina_expansion_inplane, NM_T

from .reissner_mindlin.forms import gamma
from .reissner_mindlin.forms import psi_T
from .reissner_mindlin.forms import inner_e

from .kirchhoff_love.forms import theta as kirchhoff_love_theta

from .von_karman.kinematics import e as von_karman_e

from .naghdi.kinematics import d, G, K, g