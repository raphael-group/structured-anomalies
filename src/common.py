import numpy as np
from scipy.stats import norm, entropy, hmean
import scipy as sp


################################################
# GMM helper functions
################################################

def pdf(x, mu=0.0, sigma=1.0):
    return sp.stats.norm.pdf(x, mu, sigma)

def log_likelihood(x, mu, pi):
    return np.nansum(np.log(pi*pdf(x, mu)+(1-pi)*pdf(x)))

def single_em(x, mu=0.0, pi=0.05, tol=1e-3, max_num_iter=10**3):
    x = np.asarray(x)
    a = np.zeros(np.shape(x))
    b = np.zeros(np.shape(x))
    gamma = np.zeros(np.shape(x))
    n = np.size(x)

    previous_log_likelihood = log_likelihood(x, mu, pi)

    for _ in range(max_num_iter):
        # Perform E step.
        a[:] = pi*pdf(x, mu)
        b[:] = (1-pi)*pdf(x)
        gamma[:] = a/(a+b)

        # Perform M step.
        sum_gamma = np.sum(gamma)
        mu = np.sum(gamma*x)/sum_gamma
        pi = sum_gamma/n

        # Check for convergence.
        current_log_likelihood = log_likelihood(x, mu, pi)
        if current_log_likelihood<(1+tol)*previous_log_likelihood:
            break
        else:
            previous_log_likelihood = current_log_likelihood

    return mu, pi

def compute_responsibilities(A, mu, alpha):
    top = alpha*pdf(A-mu)
    bottom = top + (1-alpha)*pdf(A)
    return top/bottom

################################################
# evaluating
################################################

def calc_fmeasure(pred_set,act_set):
    true_positives = np.intersect1d(pred_set, act_set)

    prec = len(true_positives) / len(pred_set)
    recall = len(true_positives) / len(act_set)

    # print('precision: {}'.format(prec))
    # print('recall: {}'.format(recall))

    if prec == 0 or recall == 0:
        return 0
    else:
        return hmean([prec, recall])

