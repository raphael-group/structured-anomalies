import numpy as np
import scipy as sp

from common import single_em, compute_responsibilities

# find GMM subinterval: largest l elements
def find_gmm(arr, l):
    sorted_inds = np.flip(np.argsort(arr))
    return sorted_inds[:l]

# returns anomalous indices
def find_unconstrained(sample):
    mu_est, alpha_est = single_em(sample)
    sample_resp = compute_responsibilities(sample, mu_est, alpha_est)
    est_inds = find_gmm(sample_resp, alpha_n_est)
    return est_inds