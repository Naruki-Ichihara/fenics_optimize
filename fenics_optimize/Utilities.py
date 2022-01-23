#! /usr/bin/python3
# -*- coding: utf-8 -*-
''' Utilities.
[detail]
'''

from dolfin import *
from dolfin_adjoint import *
from fecr import from_numpy, to_numpy
import datetime

class Recorder():
    '''
    Recording a field to .pvd file at each iterations.

    Attributes:
        dir (string): Directory name
        name (string): File name
    '''
    def __init__(self, dir, name):  
        self.dir = dir
        self.name = name
        self.f = File(self.dir + '/' + self.name + '/' + self.name + '.pvd')

    def rec(self, u):
        '''
        Recording.

        Args:
            u (dolfin_adjoin.Function): Field
        '''
        u.rename(self.name, 'label')    
        self.f << u

class Logger():
    '''
    Logging a cost to .csv file at each iterations.

    Attributes:
        dir (string): Directory name
        name (string): File name
    '''
    def __init__(self, dir, name):  
        self.dir = dir
        self.name = name
        with open(self.dir + '/' + self.name + '.csv', 'w') as f:
            f.write('fenics-optimize\n')
            f.write('Date, {}\n'.format(datetime.datetime.now()))
            f.write('Iter.,' + self.name + '\n')
        self.count = 0

    def rec(self, value):
        '''
        Logging.

        Args:
            value (float): Logging Value.
        '''
        self.count += 1
        with open(self.dir + '/' + self.name + '.csv', 'a') as f:
            f.write('{}, {}\n'.format(self.count, value))

def catch_problemSize(Xs):
    size = 0
    for X in Xs:
        size += Function(X).vector().size()
    return int(size)