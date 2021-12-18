#! /usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (C) 2021 Naruki Ichihara
#
# This file is part of Morphogenesis
#
# SPDX-License-Identifier:    MIT
"""Utilities"""
from dolfin import *
from dolfin_adjoint import *

def export_result(u, filepath):
    """# Utils
    ## io
    ### export_result
    Export static result as the xdmf format.

    ### Args:
        u : function
        filepath : file name

    ### Examples:

    ### Note:

    """
    file = XDMFFile(filepath)
    file.write(u)
    pass

def read_mesh(filepath, comm):
    """# Utils
    ## io
    ### read_mesh
    Read mesh with the MPI comminication world.

    ### Args:
        filepath : file name

    ### Examples:

    ### Note:

    """
    mesh = Mesh()
    XDMFFile(comm, filepath).read(mesh)
    return mesh