import unittest
from dolfin import *
from dolfin_adjoint import *
import numpy as np
import fenics_optimize as op

class FenicsOptimizeTestCase(unittest.TestCase):
    def test_AMGsolver_solve(self):
        mesh = UnitSquareMesh(10, 10)
        X = FunctionSpace(mesh, 'CG', 1)
        V = VectorFunctionSpace(mesh, 'CG', 1)
        uh_test = Function(X)
        uh_reference = Function(X)
        u = TrialFunction(X)
        du = TestFunction(X)
        f = Constant((1, 1))
        a = inner(grad(u), grad(du))*dx
        L = du*dx

        class Boundary(SubDomain):
            def inside(self, x, on_boundary):
                return on_boundary

        bcs = [DirichletBC(X, Constant(0.0), Boundary())]
        A, b = assemble_system(a, L, bcs)
        solver = op.AMGsolver(A, b)
        uh_test = solver.solve(uh_test, X)
        cost_test = assemble(inner(grad(uh_test), grad(uh_test))*dx)

        # fenics built-in solver
        solve(a==L, uh_reference, bcs)
        cost_reference = assemble(inner(grad(uh_reference), grad(uh_reference))*dx)
        
        self.assertAlmostEqual(cost_test, cost_reference, 10)

    def test_AMGsolver_nullspace(self):
        mesh = UnitSquareMesh(10, 10)
        X = FunctionSpace(mesh, 'CG', 1)
        V = VectorFunctionSpace(mesh, 'CG', 1)
        uh_test = Function(X)
        uh_reference = Function(X)
        u = TrialFunction(X)
        du = TestFunction(X)
        f = Constant((1, 1))
        a = inner(grad(u), grad(du))*dx
        L = du*dx

        class Boundary(SubDomain):
            def inside(self, x, on_boundary):
                return on_boundary

        bcs = [DirichletBC(X, Constant(0.0), Boundary())]
        A, b = assemble_system(a, L, bcs)
        solver = op.AMGsolver(A, b)
        with self.assertRaises(ValueError, msg='Invalid demension is detected.'):
            solver.solve(uh_test, X, build_null_space='2-D')
            solver.solve(uh_test, X, build_null_space='3-D')

        with self.assertRaises(ValueError, msg='Invalid key is detected.'):
            solver.solve(uh_test, X, build_null_space=True)
            solver.solve(uh_test, X, build_null_space=False)
            solver.solve(uh_test, X, build_null_space='xxx')
            solver.solve(uh_test, X, build_null_space=2)

if __name__ == '__main__':
    unittest.main()