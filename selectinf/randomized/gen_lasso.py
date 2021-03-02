from __future__ import print_function
import functools
from copy import copy

import numpy as np
from scipy.stats import norm as ndist, t as tdist

# import functools
# from copy import copy

# import numpy as np
# from scipy.stats import norm as ndist

import regreg.api as rr
import regreg.affine as ra

from .qp import qp_problem

from ..constraints.affine import constraints
from ..algorithms.sqrt_lasso import solve_sqrt_lasso, choose_lambda

from .query import (gaussian_query,
                    affine_gaussian_sampler)

from .reconstruction import reconstruct_opt
from .randomization import randomization
from ..base import restricted_estimator
from ..glm import (pairs_bootstrap_glm,
                  glm_nonparametric_bootstrap,
                  glm_parametric_covariance)
from ..algorithms.debiased_lasso import debiasing_matrix

from .gen_lasso_utils import find_trendfiltering_nspaceb, find_fusedlasso_nspaceb, find_nspaceb_identity, find_nullspace_basis, create_penalty_matrix

#### High dimensional version
#### - parametric covariance
#### - Gaussian randomization

class gen_lasso(gaussian_query):
    r"""
    A class for the randomized generalized LASSO for post-selection inference.
    The problem solved is

    .. math::

        \text{minimize}_{\beta} \ell(\beta) + 
            \sum_{i=1}^p \lambda_i |D_i\beta| - \omega^T\beta + \frac{\epsilon}{2} \|\beta\|^2_2

    where $\lambda$ is `lam`, $D$ is the penalty matrix, $\omega$ is a 
    randomization generated below and the last term is a small ridge 
    penalty. Each static method forms $\ell$ as well as the $\ell_1$ 
    penalty. The generic class forms the remaining two terms in the 
    objective.

    """

    def __init__(self,
                 loglike,
                 feature_weights,
                 ridge_term,
                 randomizer,
                 perturb=None,
                 penalty_param=None,
                 penalty_type=None,
                 fused_dims=None):
        r"""
        Create a new post-selection object for the LASSO problem

        Parameters
        ----------

        loglike : `regreg.smooth.glm.glm`
            A (negative) log-likelihood as implemented in `regreg`.

        feature_weights : float
            $\lambda$ value for L-1 penalty.

        penalty_param : [np.ndarray, int]
            Either the penalty matrix for L-1 penalty (the D in |D\beta|_1),
            or if penalty_type is provided, this parametrizes the construction 
            of the penalty matrix D (k-D fused or k-th order polynomial 
            trend-filtering), or is a custom penalty matrix of the specified 
            type in penalty_matrix. If None, defaults to 1D fused lasso, if 
            penalty_type='fused' or linear trend-filtering if 
            penalty_type='trendfiltering'. At least one of penalty_param and 
            penalty_type must not be None.

        ridge_term : float
            How big a ridge term to add?

        randomizer : object
            Randomizer -- contains representation of randomization density.

        perturb : np.ndarray
            Random perturbation subtracted as a linear
            term in the objective function.

        penalty_type : [str, np.ndarray]
            If None (default), a general solver is used, or a string in 
            ["lasso", "fused","trendfiltering"], in which case an optimized
            solver is used. At least one of penalty_param and 
            penalty_type must not be None.

        fused_dims : integer sequence
            Can't be None if penalty_matrix is "fused" and k is an int > 1,
            must be a sequence of length k specifying the dimensions of the grid.
        """

        self.loglike = loglike
        self.nfeature = p = self.loglike.shape[0]

        # if np.asarray(feature_weights).shape == ():
        #     feature_weights = np.ones(loglike.shape) * feature_weights
        # else:
        if np.asarray(feature_weights).shape != ():
            raise ValueError("'feature_weights' must be a scalar")
        self.feature_weights = feature_weights # np.asarray(feature_weights)

        self.ridge_term = ridge_term

        if penalty_param is None and penalty_type is None:
            raise ValueError("'penalty_param' and 'penalty_type' cannot both be None")
        if type(penalty_param) is np.ndarray:
            self.D = penalty_param
        elif type(penalty_type) is str:
            if penalty_param is None or type(penalty_param) is int:
                self.D = create_penalty_matrix(penalty_type,p,k=penalty_param,fused_dims=fused_dims)
            else:
                raise ValueError("'penalty_param' must be an int or None if 'penalty_type' is not None")
        else:
            raise ValueError("'penalty_param' must be a numpy array or 'penalty_type' must be a string")

        self.structure = penalty_type


        self.penalty = rr.l1norm(self.D.shape[0],lagrange=feature_weights)
        self._initial_omega = perturb  # random perturbation

        self.randomizer = randomizer

    def fit(self,
            use_admm=True,
            use_closed_form=True,
            solve_args={'tol': 1.e-12, 'min_its': 50},
            perturb=None):
        """
        Fit the randomized lasso using `regreg`.
        Parameters
        ----------
        solve_args : keyword args
             Passed to `regreg.problems.simple_problem.solve`.
        Returns
        -------
        signs : np.float
             Support and non-zero signs of randomized lasso solution.

        """

        p = self.nfeature

        # take a new perturbation if supplied
        if perturb is not None:
            self._initial_omega = perturb
        if self._initial_omega is None:
            self._initial_omega = self.randomizer.sample()

        quad = rr.identity_quadratic(self.ridge_term, 0, -self._initial_omega, 0)

        X, y = self.loglike.data
        if use_admm:
            admm_quad = (X.T.dot(X) + self.ridge_term * np.eye(p)) if use_closed_form else None
            problem = rr.admm_problem(self.loglike, 
                                      self.penalty, 
                                      self.D,
                                      0.5, 
                                      X.T.dot(y),
                                      self._initial_omega,
                                      admm_quad)
            # problem = rr.admm_problem(self.loglike, 
            #                           self.penalty, 
            #                           self.D, 
            #                           quadratic=self.loglike.quadratic,
            #                           rho_quadratic=rho_quadratic)
            problem.solve(niter=25)
            # problem.solve(quadratic=quad, niter=250)
            self.initial_soln = problem.loss_coefs # \beta
            self.initial_penalty_soln = problem.atom_coefs # D\beta
        else:
            problem = qp_problem(X, y, self.D, 
                                 ridge_term=self.ridge_term, 
                                 lam=self.feature_weights, 
                                 randomization=self._initial_omega)
            # problem.solve()
            self.exact_initial_soln = self.initial_soln = problem.solve() # \beta
            self.exact_initial_penalty_soln = self.initial_penalty_soln = self.D.dot(self.initial_soln) # D\beta
            self.initial_penalty_soln = self.initial_penalty_soln * (np.fabs(self.initial_penalty_soln) > 1e-6*np.max(X))

        active_signs = self.active_signs = np.sign(self.initial_penalty_soln)
        active = self._active = active_signs != 0 # used for finding nullspace basis

        coef_active_signs = np.sign(self.initial_soln)
        coef_active_variables = coef_active_signs != 0
        self.selection_variable = {'sign': active_signs,
                                   'variables': active}

        # initial state for opt variables
        initial_subgrad = -(self.loglike.smooth_objective(self.initial_soln, 'grad') +
                            quad.objective(self.initial_soln, 'grad'))
        self.initial_subgrad = initial_subgrad

        if self.structure == "fused":
            nsb = find_fusedlasso_nspaceb(self.D, self._active)
        elif self.structure == "trendfiltering":
            nsb = find_trendfiltering_nspaceb(self.D, self._active)
        elif self.structure == "lasso":
            nsb = find_nspaceb_identity(self._active)
        else:
            nsb = find_nullspace_basis(self.D, self._active)

        self.nsb = nsb

        if active.sum() > 0:
            nsb_pinv = np.linalg.inv(nsb.T.dot(nsb)).dot(nsb.T)
            self.initial_alpha = nsb_pinv.dot(self.initial_soln) # \beta = nsb\alpha => (nsb^Tnsb)^{-1}nsb^T\beta = \alpha
            self.observed_opt_state = self.initial_alpha
        else:
            self.observed_opt_state = self.initial_alpha = self.initial_soln

        self.num_opt_var = self.observed_opt_state.shape[0] ###### opt_var is now alpha

        W = self._W = self.loglike.saturated_loss.hessian(X.dot(self.initial_soln))
        _hessian = np.dot(X.T,X*W[:,None])

        _score_linear_term = -_hessian

        # set the observed score (data dependent) state
        self.observed_score_state = self.loglike.smooth_objective(self.initial_soln, 'grad') + _score_linear_term.dot(self.initial_soln)

        _opt_linear_term = (_hessian + self.ridge_term * np.eye(p)).dot(nsb)

        self.opt_transform = (_opt_linear_term, self.initial_subgrad)
        self.score_transform = (_score_linear_term, np.zeros(_score_linear_term.shape[0]))

        # now store everything needed for the projections
        # the projection acts only on the optimization
        # variables

        self._setup = True
        self.ndim = self.loglike.shape[0]

        if False:#active.sum() == 0:
            self.sampler = None
        else:
            # compute implied mean and covariance

            _, prec = self.randomizer.cov_prec
            opt_linear, opt_offset = self.opt_transform
            A = _hessian + self.ridge_term * np.eye(p)
            if active.sum() > 0:
                A = A.dot(nsb)

            if np.asarray(prec).shape in [(), (0,)]:
                cond_precision = A.T.dot(A) * prec
                cond_cov = np.linalg.inv(cond_precision)
                logdens_linear = cond_cov.dot(A.T) * prec
            else:
                cond_precision = A.T.dot(prec.dot(A))
                cond_cov = np.linalg.inv(cond_precision)
                logdens_linear = cond_cov.dot(A.T).dot(prec)

            cond_mean = - logdens_linear.dot(self._initial_omega - (_hessian + self.ridge_term * np.eye(p)).dot(self.initial_soln))

            # density as a function of score and optimization variables

            def log_density(logdens_linear, offset, cond_prec, score, opt):
                if score.ndim == 1:
                    mean_term = logdens_linear.dot(score.T + offset).T
                else:
                    mean_term = logdens_linear.dot(score.T + offset[:, None]).T
                arg = opt + mean_term
                return - 0.5 * np.sum(arg * cond_prec.dot(arg.T).T, 1)

            log_density = functools.partial(log_density, logdens_linear, opt_offset, cond_precision)


            # constrain |D\beta| = 0
            if active.sum() == 0:
                A_scaling = np.vstack((self.D, -self.D))
                b_scaling = np.ones(2*self.D.shape[0]) * 1e-8 * np.max(X) ## tolerance since QP soln is not exact
            # constrain -sign(D\hat\beta)D\Gamma\aplha <= 0
            else:
                A_scaling = -np.diag(active_signs[active]).dot(self.D[active,:].dot(nsb))
                b_scaling = np.zeros(active.sum())

            affine_con = constraints(A_scaling,
                                     b_scaling,
                                     mean=cond_mean,
                                     covariance=cond_cov)


            logdens_transform = (logdens_linear, opt_offset)

            self.sampler = affine_gaussian_sampler(affine_con,
                                                   self.observed_opt_state,
                                                   self.observed_score_state,
                                                   log_density,
                                                   logdens_transform,
                                                   selection_info=self.selection_variable)  # should be signs and the subgradients we've conditioned on

        return coef_active_signs, active_signs # if active.sum() > 0 else 2*np.ones_like(active_signs)

    def summary(self,
                observed_target, 
                cov_target, 
                cov_target_score, 
                alternatives,
                opt_sample=None,
                parameter=None,
                level=0.9,
                ndraw=10000,
                burnin=2000,
                compute_intervals=False):
        """
        Produce p-values and confidence intervals for targets
        of model including selected features
        Parameters
        ----------
        target : one of ['selected', 'full']
        features : np.bool
            Binary encoding of which features to use in final
            model and targets.
        parameter : np.array
            Hypothesized value for parameter -- defaults to 0.
        level : float
            Confidence level.
        ndraw : int (optional)
            Defaults to 1000.
        burnin : int (optional)
            Defaults to 1000.
        compute_intervals : bool
            Compute confidence intervals?
        dispersion : float (optional)
            Use a known value for dispersion, or Pearson's X^2?
        """


        if parameter is None:
            parameter = np.zeros_like(observed_target)

        ## handle case where active set is empty
        if self.structure != "fused" and self._active.sum() == 0:
            (ind_unbiased_estimator, 
            cov_unbiased_estimator,
            unbiased_Z_scores, 
            unbiased_pvalues,
            unbiased_intervals) = self.selective_MLE(observed_target,
                                                     cov_target,
                                                     cov_target_score,
                                                     level=level)[5:]

            sd_target = np.sqrt(np.diag(cov_unbiased_estimator))
            pivots = ndist.cdf((ind_unbiased_estimator - parameter)/sd_target)
    
            if not np.all(parameter == 0):
                pvalues = ndist.cdf((ind_unbiased_estimator - np.zeros_like(parameter))/sd_target)
            else:
                pvalues = pivots

            if alternatives == 'twosided':
                pvalues = 2 * np.minimum(pvalues, 1-pvalues)
            if alternatives == 'greater':
                pvalues = 1 - pvalues


            intervals = None
            if compute_intervals:
                _, y = self.loglike.data
                n = y.shape[0]
                alpha = 1 - level
                z = ndist.ppf(1 - alpha/2)
                # t = tdist.ppf(1 - alpha/2, n - ind_unbiased_estimator.shape[0])
                lower_limit = ind_unbiased_estimator - z*sd_target
                upper_limit = ind_unbiased_estimator + z*sd_target

                lower_limit = np.expand_dims(lower_limit, 1)
                upper_limit = np.expand_dims(upper_limit, 1)
                intervals = np.hstack((lower_limit, upper_limit))
        else:
            if opt_sample is None:
                opt_sample = self.sampler.sample(ndraw, burnin)
            else:
                ndraw = opt_sample.shape[0]

            pivots = self.sampler.coefficient_pvalues(observed_target,
                                                      cov_target,
                                                      cov_target_score,
                                                      parameter=parameter,
                                                      sample=opt_sample,
                                                      alternatives=alternatives)

            if not np.all(parameter == 0):
                pvalues = self.sampler.coefficient_pvalues(observed_target,
                                                           cov_target,
                                                           cov_target_score,
                                                           parameter=np.zeros_like(parameter),
                                                           sample=opt_sample,
                                                           alternatives=alternatives)
            else:
                pvalues = pivots

            intervals = None
            if compute_intervals:

                # MLE_intervals = self.selective_MLE(observed_target,
                #                                    cov_target,
                #                                    cov_target_score,
                #                                    level=level)[4]

                intervals = self.sampler.confidence_intervals(observed_target,
                                                              cov_target,
                                                              cov_target_score,
                                                              sample=opt_sample,
                                                              # initial_guess=MLE_intervals,
                                                              level=level)

        return pivots, pvalues, intervals

    @staticmethod
    def gaussian(X,
                 Y,
                 feature_weights,
                 penalty_param=None,
                 penalty_type=None,
                 sigma=1.,
                 quadratic=None,
                 ridge_term=None,
                 randomizer_scale=None):
        r"""
        Squared-error LASSO with feature weights.
        Objective function is (before randomization)

        $$
        \beta \mapsto \frac{1}{2} \|Y-X\beta\|^2_2 + \sum_{i=1}^p \lambda_i |\beta_i|
        $$

        where $\lambda$ is `feature_weights`. The ridge term
        is determined by the Hessian and `np.std(Y)` by default,
        as is the randomizer scale.
        Parameters
        ----------
        X : ndarray
            Shape (n,p) -- the design matrix.
        Y : ndarray
            Shape (n,) -- the response.
        feature_weights: [float, sequence]
            Penalty weights. An intercept, or other unpenalized
            features are handled by setting those entries of
            `feature_weights` to 0. If `feature_weights` is
            a float, then all parameters are penalized equally.
        sigma : float (optional)
            Noise variance. Set to 1 if `covariance_estimator` is not None.
            This scales the loglikelihood by `sigma**(-2)`.
        quadratic : `regreg.identity_quadratic.identity_quadratic` (optional)
            An optional quadratic term to be added to the objective.
            Can also be a linear term by setting quadratic
            coefficient to 0.
        ridge_term : float
            How big a ridge term to add?
        randomizer_scale : float
            Scale for IID components of randomizer.
        randomizer : str
            One of ['laplace', 'logistic', 'gaussian']
        Returns
        -------
        L : `selection.randomized.convenience.lasso`

        """

        loglike = rr.glm.gaussian(X, Y, coef=1. / sigma ** 2, quadratic=quadratic)
        n, p = X.shape

        mean_diag = np.mean((X ** 2).sum(0))
        if ridge_term is None:
            ridge_term = np.std(Y) * np.sqrt(mean_diag) / np.sqrt(n - 1)

        if randomizer_scale is None:
            randomizer_scale = np.sqrt(mean_diag) * 0.5 * np.std(Y) * np.sqrt(n / (n - 1.))

        randomizer = randomization.isotropic_gaussian((p,), randomizer_scale)

        return gen_lasso(loglike, 
                     np.asarray(feature_weights) / sigma ** 2,
                     ridge_term, randomizer,
                     penalty_param=penalty_param,
                     penalty_type=penalty_type)

    @staticmethod
    def logistic(X,
                 successes,
                 feature_weights,
                 trials=None,
                 quadratic=None,
                 ridge_term=None,
                 randomizer_scale=None):
        r"""
        Logistic LASSO with feature weights (before randomization)
        $$
        \beta \mapsto \ell(X\beta) + \sum_{i=1}^p \lambda_i |\beta_i|
        $$
        where $\ell$ is the negative of the logistic
        log-likelihood (half the logistic deviance)
        and $\lambda$ is `feature_weights`.
        Parameters
        ----------
        X : ndarray
            Shape (n,p) -- the design matrix.
        successes : ndarray
            Shape (n,) -- response vector. An integer number of successes.
            For data that is proportions, multiply the proportions
            by the number of trials first.
        feature_weights: [float, sequence]
            Penalty weights. An intercept, or other unpenalized
            features are handled by setting those entries of
            `feature_weights` to 0. If `feature_weights` is
            a float, then all parameters are penalized equally.
        trials : ndarray (optional)
            Number of trials per response, defaults to
            ones the same shape as Y.
        quadratic : `regreg.identity_quadratic.identity_quadratic` (optional)
            An optional quadratic term to be added to the objective.
            Can also be a linear term by setting quadratic
            coefficient to 0.
        ridge_term : float
            How big a ridge term to add?
        randomizer_scale : float
            Scale for IID components of randomizer.
        randomizer : str
            One of ['laplace', 'logistic', 'gaussian']
        Returns
        -------
        L : `selection.randomized.convenience.lasso`

        """
        n, p = X.shape

        loglike = rr.glm.logistic(X, successes, trials=trials, quadratic=quadratic)

        mean_diag = np.mean((X ** 2).sum(0))

        if ridge_term is None:
            ridge_term = np.std(Y) * np.sqrt(mean_diag) / np.sqrt(n - 1)

        if randomizer_scale is None:
            randomizer_scale = np.sqrt(mean_diag) * 0.5

        randomizer = randomization.isotropic_gaussian((p,), randomizer_scale)

        return lasso(loglike, 
                     np.asarray(feature_weights),
                     ridge_term, randomizer)

    @staticmethod
    def coxph(X,
              times,
              status,
              feature_weights,
              quadratic=None,
              ridge_term=None,
              randomizer_scale=None):
        r"""
        Cox proportional hazards LASSO with feature weights.
        Objective function is (before randomization)

        $$
        \beta \mapsto \ell^{\text{Cox}}(\beta) + \sum_{i=1}^p \lambda_i |\beta_i|
        $$
        where $\ell^{\text{Cox}}$ is the
        negative of the log of the Cox partial
        likelihood and $\lambda$ is `feature_weights`.
        Uses Efron's tie breaking method.
        Parameters
        ----------
        X : ndarray
            Shape (n,p) -- the design matrix.
        times : ndarray
            Shape (n,) -- the survival times.
        status : ndarray
            Shape (n,) -- the censoring status.
        feature_weights: [float, sequence]
            Penalty weights. An intercept, or other unpenalized
            features are handled by setting those entries of
            `feature_weights` to 0. If `feature_weights` is
            a float, then all parameters are penalized equally.
        covariance_estimator : optional
            If None, use the parameteric
            covariance estimate of the selected model.
        quadratic : `regreg.identity_quadratic.identity_quadratic` (optional)
            An optional quadratic term to be added to the objective.
            Can also be a linear term by setting quadratic
            coefficient to 0.
        ridge_term : float
            How big a ridge term to add?
        randomizer_scale : float
            Scale for IID components of randomizer.
        randomizer : str
            One of ['laplace', 'logistic', 'gaussian']
        Returns
        -------
        L : `selection.randomized.convenience.lasso`

        """
        loglike = coxph_obj(X, times, status, quadratic=quadratic)

        # scale for randomization seems kind of meaningless here...

        mean_diag = np.mean((X ** 2).sum(0))

        if ridge_term is None:
            ridge_term = np.std(times) * np.sqrt(mean_diag) / np.sqrt(n - 1)

        if randomizer_scale is None:
            randomizer_scale = np.sqrt(mean_diag) * 0.5 * np.std(Y) * np.sqrt(n / (n - 1.))

        randomizer = randomization.isotropic_gaussian((p,), randomizer_scale)

        return lasso(loglike,
                     feature_weights,
                     ridge_term,
                     randomizer)

    @staticmethod
    def poisson(X,
                counts,
                feature_weights,
                quadratic=None,
                ridge_term=None,
                randomizer_scale=None):
        r"""
        Poisson log-linear LASSO with feature weights.
        Objective function is (before randomization)

        $$
        \beta \mapsto \ell^{\text{Poisson}}(\beta) + \sum_{i=1}^p \lambda_i |\beta_i|
        $$
        where $\ell^{\text{Poisson}}$ is the negative
        of the log of the Poisson likelihood (half the deviance)
        and $\lambda$ is `feature_weights`.
        Parameters
        ----------
        X : ndarray
            Shape (n,p) -- the design matrix.
        counts : ndarray
            Shape (n,) -- the response.
        feature_weights: [float, sequence]
            Penalty weights. An intercept, or other unpenalized
            features are handled by setting those entries of
            `feature_weights` to 0. If `feature_weights` is
            a float, then all parameters are penalized equally.
        quadratic : `regreg.identity_quadratic.identity_quadratic` (optional)
            An optional quadratic term to be added to the objective.
            Can also be a linear term by setting quadratic
            coefficient to 0.
        ridge_term : float
            How big a ridge term to add?
        randomizer_scale : float
            Scale for IID components of randomizer.
        randomizer : str
            One of ['laplace', 'logistic', 'gaussian']
        Returns
        -------
        L : `selection.randomized.convenience.lasso`

        """
        n, p = X.shape
        loglike = rr.glm.poisson(X, counts, quadratic=quadratic)

        # scale for randomizer seems kind of meaningless here...

        mean_diag = np.mean((X ** 2).sum(0))

        if ridge_term is None:
            ridge_term = np.std(counts) * np.sqrt(mean_diag) / np.sqrt(n - 1)

        if randomizer_scale is None:
            randomizer_scale = np.sqrt(mean_diag) * 0.5 * np.std(counts) * np.sqrt(n / (n - 1.))

        randomizer = randomization.isotropic_gaussian((p,), randomizer_scale)

        return lasso(loglike,
                     feature_weights,
                     ridge_term,
                     randomizer)

    @staticmethod
    def sqrt_lasso(X,
                   Y,
                   feature_weights,
                   quadratic=None,
                   ridge_term=None,
                   randomizer_scale=None,
                   solve_args={'min_its': 200},
                   perturb=None):
        r"""
        Use sqrt-LASSO to choose variables.
        Objective function is (before randomization)

        $$
        \beta \mapsto \|Y-X\beta\|_2 + \sum_{i=1}^p \lambda_i |\beta_i|
        $$
        where $\lambda$ is `feature_weights`. After solving the problem
        treat as if `gaussian` with implied variance and choice of
        multiplier. See arxiv.org/abs/1504.08031 for details.
        Parameters
        ----------
        X : ndarray
            Shape (n,p) -- the design matrix.
        Y : ndarray
            Shape (n,) -- the response.
        feature_weights: [float, sequence]
            Penalty weights. An intercept, or other unpenalized
            features are handled by setting those entries of
            `feature_weights` to 0. If `feature_weights` is
            a float, then all parameters are penalized equally.
        quadratic : `regreg.identity_quadratic.identity_quadratic` (optional)
            An optional quadratic term to be added to the objective.
            Can also be a linear term by setting quadratic
            coefficient to 0.
        covariance : str
            One of 'parametric' or 'sandwich'. Method
            used to estimate covariance for inference
            in second stage.
        solve_args : dict
            Arguments passed to solver.
        ridge_term : float
            How big a ridge term to add?
        randomizer_scale : float
            Scale for IID components of randomizer.
        randomizer : str
            One of ['laplace', 'logistic', 'gaussian']
        Returns
        -------
        L : `selection.randomized.convenience.lasso`

        Notes
        -----
        Unlike other variants of LASSO, this
        solves the problem on construction as the active
        set is needed to find equivalent gaussian LASSO.
        Assumes parametric model is correct for inference,
        i.e. does not accept a covariance estimator.
        """

        n, p = X.shape

        if np.asarray(feature_weights).shape == ():
            feature_weights = np.ones(p) * feature_weights

        mean_diag = np.mean((X ** 2).sum(0))
        if ridge_term is None:
            ridge_term = np.sqrt(mean_diag) / (n - 1)

        if randomizer_scale is None:
            randomizer_scale = 0.5 * np.sqrt(mean_diag) / np.sqrt(n - 1)

        if perturb is None:
            perturb = np.random.standard_normal(p) * randomizer_scale

        randomQ = rr.identity_quadratic(ridge_term, 0, -perturb, 0)  # a ridge + linear term

        if quadratic is not None:
            totalQ = randomQ + quadratic
        else:
            totalQ = randomQ

        soln, sqrt_loss = solve_sqrt_lasso(X,
                                           Y,
                                           weights=feature_weights,
                                           quadratic=totalQ,
                                           solve_args=solve_args,
                                           force_fat=True)

        denom = np.linalg.norm(Y - X.dot(soln))
        loglike = rr.glm.gaussian(X, Y)

        randomizer = randomization.isotropic_gaussian((p,), randomizer_scale * denom)

        obj = lasso(loglike, 
                    np.asarray(feature_weights) * denom,
                    ridge_term * denom,
                    randomizer,
                    perturb=perturb * denom)
        obj._sqrt_soln = soln

        return obj

# Targets of inference
# and covariance with score representation

def fused_targets(loglike, 
                  W, 
                  ccs, 
                  sign_info={}, 
                  dispersion=None,
                  solve_args={'tol': 1.e-12, 'min_its': 50}):

    X, y = loglike.data
    n, p = X.shape

    if (len(ccs) == 0):
        raise ValueError("No connected components provided, no targets to estimate.")
    if (len(ccs) > 1):
        Xcc = np.vstack([X[:,cc].sum(axis=1) for cc in ccs]).T
    else:
        Xcc = np.array(X[:,ccs[0]].sum(axis=1))[:,None]
    ncc = Xcc.shape[1]
    loglikecc = rr.glm.gaussian(Xcc, y, coef=loglike.coef, quadratic=None)#loglike.quadratic)
    observed_target = loglikecc.solve(**solve_args)
    if W.ndim == 1:
        Qcc = Xcc.T.dot(W[:,None] * Xcc)
        _score_linear = -Xcc.T.dot(W[:,None] * X).T
    else:
        Qcc = Xcc.T.dot(W.dot(Xcc))
        _score_linear = -Xcc.T.dot(W.dot(X)).T
    cov_target = Qcc/(np.diag(Xcc.T.dot(Xcc))**2) # np.linalg.inv(Qcc)
    crosscov_target_score = _score_linear.dot(np.linalg.inv(Xcc.T.dot(Xcc)))
    # assert(0==1)
    alternatives = ['twosided'] * ncc
    ccs_idx = np.arange(ncc)

    for i in range(len(alternatives)):
        if ccs_idx[i] in sign_info.keys():
            alternatives[i] = sign_info[ccs_idx[i]]

    if dispersion is None:
        if W.ndim == 1:
            dispersion = ((y-loglikecc.saturated_loss.mean_function(
                           Xcc.dot(observed_target))) ** 2 / W).sum() / (n-ncc)#-1)
        else:
            dispersion = ((y-loglikecc.saturated_loss.mean_function(
                           Xcc.dot(observed_target))) ** 2 / np.diag(W)).sum() / (n-ncc)#-1)

    return observed_target, cov_target * dispersion, crosscov_target_score.T * dispersion, alternatives

def selected_targets(loglike, 
                     W, 
                     features, 
                     sign_info={}, 
                     dispersion=None,
                     solve_args={'tol': 1.e-12, 'min_its': 50}):

    # if features.sum() == 0:
    #     raise ValueError('Empty feature set, no targets to estimate.')
    
    X, y = loglike.data
    n, p = X.shape

    Xfeat = X[:, features]
    Qfeat = Xfeat.T.dot(W[:, None] * Xfeat)
    observed_target = restricted_estimator(loglike, features, solve_args=solve_args)
    cov_target = np.linalg.inv(Qfeat)
    _score_linear = -Xfeat.T.dot(W[:, None] * X).T
    crosscov_target_score = _score_linear.dot(cov_target)
    alternatives = ['twosided'] * features.sum()
    features_idx = np.arange(p)[features]

    for i in range(len(alternatives)):
        if features_idx[i] in sign_info.keys():
            alternatives[i] = sign_info[features_idx[i]]

    if dispersion is None:  # use Pearson's X^2
        dispersion = ((y - loglike.saturated_loss.mean_function(
            Xfeat.dot(observed_target))) ** 2 / W).sum() / (n - Xfeat.shape[1])

    return observed_target, cov_target * dispersion, crosscov_target_score.T * dispersion, alternatives

def full_targets(loglike, 
                 W, 
                 features, 
                 dispersion=None,
                 solve_args={'tol': 1.e-12, 'min_its': 50}):

    if features.sum() == 0:
        raise ValueError('Empty feature set, no targets to estimate.')

    X, y = loglike.data
    n, p = X.shape
    features_bool = np.zeros(p, np.bool)
    features_bool[features] = True
    features = features_bool

    # target is one-step estimator

    Qfull = X.T.dot(W[:, None] * X)
    Qfull_inv = np.linalg.inv(Qfull)
    full_estimator = loglike.solve(**solve_args)
    cov_target = Qfull_inv[features][:, features]
    observed_target = full_estimator[features]
    crosscov_target_score = np.zeros((p, cov_target.shape[0]))
    crosscov_target_score[features] = -np.identity(cov_target.shape[0])

    if dispersion is None:  # use Pearson's X^2
        dispersion = (((y - loglike.saturated_loss.mean_function(X.dot(full_estimator))) ** 2 / W).sum() / 
                      (n - p))

    alternatives = ['twosided'] * features.sum()
    return observed_target, cov_target * dispersion, crosscov_target_score.T * dispersion, alternatives

def debiased_targets(loglike, 
                     W, 
                     features, 
                     sign_info={}, 
                     penalty=None, #required kwarg
                     dispersion=None,
                     debiasing_args={}):

    if features.sum() == 0:
        raise ValueError('Empty feature set, no targets to estimate.')

    if penalty is None:
        raise ValueError('require penalty for consistent estimator')

    X, y = loglike.data
    n, p = X.shape
    features_bool = np.zeros(p, np.bool)
    features_bool[features] = True
    features = features_bool

    # relevant rows of approximate inverse

    Qinv_hat = np.atleast_2d(debiasing_matrix(X * np.sqrt(W)[:, None], 
                                              np.nonzero(features)[0],
                                              **debiasing_args)) / n

    problem = rr.simple_problem(loglike, penalty)
    nonrand_soln = problem.solve()
    G_nonrand = loglike.smooth_objective(nonrand_soln, 'grad')

    observed_target = nonrand_soln[features] - Qinv_hat.dot(G_nonrand)

    if p > n:
        M1 = Qinv_hat.dot(X.T)
        cov_target = (M1 * W[None, :]).dot(M1.T)
        crosscov_target_score = -(M1 * W[None, :]).dot(X).T
    else:
        Qfull = X.T.dot(W[:, None] * X)
        cov_target = Qinv_hat.dot(Qfull.dot(Qinv_hat.T))
        crosscov_target_score = -Qinv_hat.dot(Qfull).T

    if dispersion is None:  # use Pearson's X^2
        Xfeat = X[:, features]
        Qrelax = Xfeat.T.dot(W[:, None] * Xfeat)
        relaxed_soln = nonrand_soln[features] - np.linalg.inv(Qrelax).dot(G_nonrand[features])
        dispersion = (((y - loglike.saturated_loss.mean_function(Xfeat.dot(relaxed_soln)))**2 / W).sum() / 
                      (n - features.sum()))

    alternatives = ['twosided'] * features.sum()
    return observed_target, cov_target * dispersion, crosscov_target_score.T * dispersion, alternatives

def form_targets(target, 
                 loglike, 
                 W, 
                 features, 
                 **kwargs):
    _target = {'full':full_targets,
               'selected':selected_targets,
               'debiased':debiased_targets,
               'fused':fused_targets}[target]
    return _target(loglike,
                   W,
                   features,
                   **kwargs)
