from __future__ import division, print_function

import numpy as np
import nose.tools as nt

from scipy.stats import norm as ndist

import regreg.api as rr

# import rpy2.robjects as rpy
# from rpy2.robjects import numpy2ri
# rpy.r('library(selectiveInference)')

from ..gen_lasso import gen_lasso, selected_targets, full_targets, debiased_targets, fused_targets
from ..lasso import lasso
from ...tests.instance import gaussian_instance
from ...algorithms.sqrt_lasso import choose_lambda, solve_sqrt_lasso
from ..randomization import randomization

def test_simple_lasso(n=50, p=10, signal_fac=1.5, s=2, sigma=1, penalty_type="lasso", target='selected', rho=0.4, randomizer_scale=1, ndraw=5000, burnin=1000):
  zero_int_results=np.array([],dtype=bool)
  nonzero_int_results=np.array([],dtype=bool)
  zero_results = np.array([],dtype=bool)
  nonzero_results = np.array([],dtype=bool)

  for i in range(100):
    if i % 10 == 0: print(i)
    inst, const = gaussian_instance, gen_lasso.gaussian
    signal = np.sqrt(signal_fac * np.log(p))
    X, Y, beta = inst(n=n,
                      p=p, 
                      signal=signal, 
                      s=s, 
                      equicorrelated=False, 
                      rho=rho, 
                      sigma=sigma, 
                      random_signs=True)[:3]

    n, p = X.shape

    sigma_ = np.std(Y)

    # print("sigma is ", sigma)
    # print("sigma_ is ", sigma_)
    if target is not 'debiased':
        # W = np.ones(X.shape[1]) * np.sqrt(1.5 * np.log(p)) * sigma_
        W = np.sqrt(1.5 * np.log(p)) * sigma_
    else:
        # W = np.ones(X.shape[1]) * np.sqrt(2 * np.log(p)) * sigma_
        W = np.sqrt(2 * np.log(p)) * sigma_


    conv = const(X, 
                 Y,
                 W, # make sure just a number
                 penalty_type=penalty_type,
                 randomizer_scale=randomizer_scale * sigma_)
    
    signs, penalty_signs = conv.fit()
    # nonzero = signs != 0
    nonzero = penalty_signs != 0

    if target == 'full':
        (observed_target, 
         cov_target, 
         cov_target_score, 
         alternatives) = full_targets(conv.loglike, 
                                      conv._W, 
                                      nonzero)
    elif target == 'selected':
        (observed_target, 
         cov_target, 
         cov_target_score, 
         alternatives) = selected_targets(conv.loglike, 
                                          conv._W, 
                                          nonzero)
    elif target == 'debiased':
        (observed_target, 
         cov_target, 
         cov_target_score, 
         alternatives) = debiased_targets(conv.loglike, 
                                          conv._W, 
                                          nonzero,
                                          penalty=conv.penalty)


    _, pval, intervals = conv.summary(observed_target, 
                                      cov_target, 
                                      cov_target_score, 
                                      alternatives,
                                      ndraw=ndraw,
                                      burnin=burnin, 
                                      compute_intervals=True)

    zero_idx = beta[nonzero] == 0
    nonzero_idx = beta[nonzero] != 0
    zero_intervals = intervals[zero_idx,:]
    nonzero_intervals = intervals[nonzero_idx,:]

    zero_beta = beta[nonzero][zero_idx]
    nonzero_beta = beta[nonzero][nonzero_idx]

    # print(zero_intervals)

    zero_int_results = np.concatenate((zero_int_results,np.logical_and(zero_beta > zero_intervals[:,0],zero_beta < zero_intervals[:,1])))
    nonzero_int_results = np.concatenate((nonzero_int_results,np.logical_and(nonzero_beta > nonzero_intervals[:,0],nonzero_beta < nonzero_intervals[:,1])))
    
    zero_results = np.concatenate((zero_results,pval[zero_idx]))
    nonzero_results = np.concatenate((nonzero_results,pval[nonzero_idx]))
    # print(intervals)

  # print(zero_int_results)
  # print(nonzero_int_results)
  # print(zero_results)
  # print(nonzero_results)

  zero_int_results = np.array(zero_int_results).flatten()
  nonzero_int_results = np.array(nonzero_int_results).flatten()
  print("zero intervals:", zero_int_results.mean())
  print("zero intervals:", zero_int_results.sum())
  print("nonzero intervals:", nonzero_int_results.mean())
  print("nonzero intervals:", nonzero_int_results.sum())
  if len(zero_results) > 0: 
    print("checking zero targets",(zero_results > .1).sum(),(zero_results > .1).mean())
  if len(nonzero_results) > 0: 
    print("checking nonzero targets",(nonzero_results < .1).sum(),(nonzero_results < .1).mean())
      
  return pval[beta[nonzero] == 0], pval[beta[nonzero] != 0]




# def test_simple_fused_lasso(p=100, 
#                             signal_fac=1.5, 
#                             ncc=2, 
#                             sigma=3, 
#                             penalty_param=None, 
#                             penalty_type="fused", 
#                             target='fused', 
#                             rho=0.4, 
#                             randomizer_scale=1, 
#                             ndraw=5000, 
#                             burnin=1000):
#   zero_int_results=np.array([],dtype=bool)
#   nonzero_int_results=np.array([],dtype=bool)
#   zero_results = np.array([],dtype=bool)
#   nonzero_results = np.array([],dtype=bool)
#   for i in range(100):
#     if i % 10 == 0: print(i)
#     const = gen_lasso.gaussian
#     signal = np.sqrt(signal_fac * np.log(p))

#     X = np.eye(p)
#     true_beta = beta = np.random.choice([-1,1],size=ncc,replace=False)#np.random.choice(np.linspace(-10,10,5),size=ncc,replace=False)
    
#     comp_size = int(p/ncc)
#     remain = p % ncc

#     beta_stack = []
#     for i in range(ncc):
#         beta_stack.append(beta[i]*np.ones(comp_size + (i < remain)))
#     tiled_beta = np.hstack(beta_stack)

#     Y = tiled_beta + ndist.rvs(size=p)*sigma

#     # sigma_ = np.std(Y)
#     sigma_ = np.std(Y-tiled_beta)

#     # print("sigma is ", sigma)
#     # print("sigma_ is ", sigma_)

#     if target is not 'debiased':
#         # W = np.ones(X.shape[1]) * np.sqrt(1.5 * np.log(p)) * sigma_
#         W = np.sqrt(1.5 * np.log(p)) * sigma_
#     else:
#         # W = np.ones(X.shape[1]) * np.sqrt(2 * np.log(p)) * sigma_
#         W = np.sqrt(2 * np.log(p)) * sigma_


#     conv = const(X, 
#                  Y,
#                  W, # make sure just a number
#                  penalty_param=penalty_param,
#                  penalty_type=penalty_type,
#                  randomizer_scale=randomizer_scale * sigma_)
    
#     signs, penalty_signs = conv.fit()
#     # nonzero = signs != 0
#     nonzero = penalty_signs != 0

#     if target == 'full':
#         (observed_target, 
#          cov_target, 
#          cov_target_score, 
#          alternatives) = full_targets(conv.loglike, 
#                                       conv._W, 
#                                       nonzero)
#     elif target == 'selected':
#         (observed_target, 
#          cov_target, 
#          cov_target_score, 
#          alternatives) = selected_targets(conv.loglike, 
#                                           conv._W, 
#                                           nonzero)
#     elif target == 'debiased':
#         (observed_target, 
#          cov_target, 
#          cov_target_score, 
#          alternatives) = debiased_targets(conv.loglike, 
#                                           conv._W, 
#                                           nonzero,
#                                           penalty=conv.penalty)
#     elif target == 'fused':
#         if nonzero.sum() == 0:
#             breaks = np.array([0,p])
#         else:
#           breaks = np.where(nonzero)[0]
#           breaks += 1
#           breaks=np.hstack((0,breaks,p))
#         I = np.eye(p)
#         # Wcc = np.array([conv._W[breaks[i]:breaks[i+1]].sum() for i in range(len(breaks)-1)]).squeeze()
#         # Wcc = np.diag(Wcc)
#         # print(Wcc)
#         # print("---------")
#         initial_ccs = [I[:,breaks[i]:breaks[i+1]].sum(axis=1) for i in range(len(breaks)-1)]
#         # print(initial_ccs)
#         # print("~~~~~~")
#         initial_ccs = np.array(initial_ccs,dtype=bool)
#         print("initial ccs", initial_ccs.T.astype(int))
#         print("nsb", conv.nsb.astype(int))
#         stop
#         # W = conv._W.dot(initial_ccs)
#         # print(initial_ccs)
#         # print(W)
#         (observed_target, 
#          cov_target, 
#          cov_target_score, 
#          alternatives) = fused_targets(conv.loglike, 
#                                           conv._W, 
#                                           initial_ccs)
#         nobserved = observed_target.shape[0]
#         nonzero = np.ones(nobserved).astype(bool)
#         beta = np.array([np.mean(tiled_beta[initial_ccs[i]]) for i in range(nobserved)])
#         # print("observed,",observed_target)
#         # print("beta,",beta)



#     _, pval, intervals = conv.summary(observed_target, 
#                                       cov_target, 
#                                       cov_target_score, 
#                                       alternatives,
#                                       ndraw=ndraw,
#                                       burnin=burnin, 
#                                       compute_intervals=True)

#     zero_idx = beta[nonzero] == 0
#     nonzero_idx = beta[nonzero] != 0
#     zero_intervals = intervals[zero_idx,:]
#     nonzero_intervals = intervals[nonzero_idx,:]

#     zero_beta = beta[zero_idx]
#     nonzero_beta = beta[nonzero_idx]

#     # print(zero_intervals)

#     zero_int_results = np.concatenate((zero_int_results,np.logical_and(zero_beta >= zero_intervals[:,0],zero_beta <= zero_intervals[:,1])))
#     nonzero_int_results = np.concatenate((nonzero_int_results,np.logical_and(nonzero_beta >= nonzero_intervals[:,0],nonzero_beta <= nonzero_intervals[:,1])))
    
#     zero_results = np.concatenate((zero_results,pval[zero_idx]))
#     nonzero_results = np.concatenate((nonzero_results,pval[nonzero_idx]))
#     # print(intervals)

#   # print(zero_int_results)
#   # print(nonzero_int_results)
#   # print(zero_results)
#   # print(nonzero_results)

#   zero_int_results = np.array(zero_int_results).flatten()
#   nonzero_int_results = np.array(nonzero_int_results).flatten()
#   print("zero intervals:", zero_int_results.mean())
#   print("zero intervals:", zero_int_results.sum())
#   print("nonzero intervals:", nonzero_int_results.mean())
#   print("nonzero intervals:", nonzero_int_results.sum())
#   if len(zero_results) > 0: 
#     print("checking zero targets",(zero_results > .1).sum(),(zero_results > .1).mean())
#   if len(nonzero_results) > 0: 
#     print("checking nonzero targets",(nonzero_results < .1).sum(),(nonzero_results < .1).mean())
#   # if len(zero_results) > 0: 
#   #   print("zero targets",zero_results)
#   # if len(nonzero_results) > 0: 
#   #   print("nonzero targets",nonzero_results)
#   # if len(zero_results) > 0: 
#   #   nt.assert_true((zero_results > .1).mean() >= .9)
#   # if len(nonzero_results) > 0: 
#   #   nt.assert_true((nonzero_results < .1).mean() >= .9)
#   # nt.assert_true(int_results.mean() >= .9)
  



