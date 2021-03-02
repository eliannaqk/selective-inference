import numpy as np
import nose.tools as nt

import regreg.api as rr 
import regreg.affine as ra
from ..qp import qp_problem

def test_qp(n=500, p=100):
    # np.random.seed(0)
    X = np.random.standard_normal((n, p))
    y = np.random.standard_normal(n)
    # def is_pos_def(x):
    #     return np.all(np.linalg.eigvals(x) > 0)
    # print(is_pos_def(X.T.dot(X)))
    # print(X)
    # loss = rr.squared_error(X, Y)
    D = np.identity(p)
    pen = rr.l1norm(p, lagrange=1.5)

    QP = qp_problem(X, y, np.eye(p), ridge_term=0)
    soln = QP.solve()

    closed_form_soln = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)

    # tol = 1e-5
    # yield np.testing.assert_allclose, closed_form_soln, soln, tol, tol, False, 'checking initial true and qp solutions'


    # print(QP.m.getVars())
    # ADMM.solve(niter=1000)

    # coef1 = ADMM.atom_coefs
    # problem2 = rr.simple_problem(loss, pen)
    # coef2 = problem2.solve(tol=1.e-12, min_its=500)

    # np.testing.assert_allclose(coef1, coef2, rtol=1.e-3, atol=1.e-4)
    
