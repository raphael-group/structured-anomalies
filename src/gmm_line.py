import numpy as np
import scipy as sp

from common import single_em, compute_responsibilities

# find GMM subinterval: max sum subarray of length l
def find_subint(arr, l):
    n = len(arr)
    max_sum = -np.inf
    max_left_end = -1
    max_right_end = -1
    for i in range(n-l):
        s = np.sum(arr[i:i+l])
        if s > max_sum:
            max_sum = s
            max_left_end = i
            max_right_end = i + l - 1
    return max_left_end, max_right_end

# returns anomalous indices
def find_line(sample):
    mu_est, alpha_est = single_em(sample)
    alpha_n_est = int(alpha_est*n)
    sample_resp = compute_responsibilities(sample, mu_est, alpha_est)
    le, re = find_subint(sample_resp, alpha_n_est)
    return range(le, re+1)