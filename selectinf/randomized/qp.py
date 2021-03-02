#!/usr/bin/python

# Copyright 2018, Gurobi Optimization, LLC

# This example formulates and solves the following simple QP model:
#  minimize
#      x^2 + x*y + y^2 + y*z + z^2 + 2 x
#  subject to
#      x + 2 y + 3 z >= 4
#      x +   y       >= 1
#
# It solves it once as a continuous model, and once as an integer model.

import numpy as np
import warnings
# from ..smooth import sum as smooth_sum
# from ..smooth.quadratic import quadratic_loss
# from ..affine import aslinear, astransform
# from ..identity_quadratic import identity_quadratic
have_gurobipy = True
try:
        from gurobipy import *
except ImportError:
        warnings.warn('gurobipy not available -- genlasso not going to work!')
        have_gurobipy = False

if have_gurobipy:
        ##### NOTE: rn only for squared error loss
        ##### TODO: make general later
        class qp_problem(object):


                def __init__(self,
                             X,
                             y,
                             D,
                             ridge_term=0.01,
                             lam=1.5,
                             randomization=None):

                        self.n, self.p = n, p = X.shape
                        q = D.shape[0]

                        # Create a new model
                        self.m = Model("qp")
                        self.m.setParam( 'OutputFlag', False )

                        # Create variables
                        beta = self.m.addVars(p, lb=-GRB.INFINITY, name="beta")
                        t = self.m.addVars(q, lb=-GRB.INFINITY, name="t")

                        # Set objective: beta^T(X^TX + ridge_term/2)beta - 2(y^TX+w)beta + lam*1^Tt
                        quad = 0.5 * (X.T.dot(X))# + ridge_term * np.eye(p))
                        quad_dict = {i: beta.prod({j: quad[j,i] for j in range(p)}) for i in range(p)}

                        if randomization is None:
                                randomization = np.random.standard_normal(p)

                        linear = y.T.dot(X) + randomization
                        linear_dict = {i: lin for i,lin in enumerate(linear)}

                        obj = beta.prod(quad_dict) - beta.prod(linear_dict) + lam*t.sum()
                        self.m.setObjective(obj)

                        # Add constraints: -t <= Dbeta <= t
                        for i in range(q):
                                D_dict = beta.prod({j: D[i,j] for j in range(p)})
                                self.m.addConstr(D_dict <= t[i], "pc{}".format(i))
                                self.m.addConstr(D_dict >= -t[i], "nc{}".format(i))

                def solve(self):
                        self.m.optimize()

                        soln = np.zeros(self.p)
                        i = 0

                        for v in self.m.getVars():
                                if 'beta' in v.VarName:
                                        soln[i] = v.x
                                        i += 1

                        self.soln = soln

                        return soln









