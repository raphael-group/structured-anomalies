import numpy as np
from itertools import product

from scipy.stats import norm
import scipy as sp

import gurobipy as gp

from time import time

import argparse

################################################
# helpers for GMM
################################################

def pdf_entrywise_2d(A, mu=0.0):
    B = np.zeros_like(A)
    n,m = B.shape
    for i in range(n):
        for j in range(m):
            B[i,j] = pdf(A[i,j], mu=mu)
    return B

def pdf_entrywise_1d(A, mu=0.0):
    B = np.zeros_like(A)
    n = B.shape[0]
    for i in range(n):
        B[i] = pdf(A[i], mu=mu)
    return B

def pdf(x, mu=0.0, sigma=1.0):
    return np.exp(-0.5*((x-mu)/(sigma*1.0))**2)

def log_likelihood(x, mu, pi):
    return np.nansum(np.log(pi*pdf_entrywise_1d(x, mu)+(1-pi)*pdf_entrywise_1d(x)))

def single_em(x, mu=0.0, pi=0.05, tol=1e-3, max_num_iter=10**3):
    x = np.asarray(x)
    a = np.zeros(np.shape(x))
    b = np.zeros(np.shape(x))
    gamma = np.zeros(np.shape(x))
    n = np.size(x)

    previous_log_likelihood = log_likelihood(x, mu, pi)

    for _ in range(max_num_iter):
        # Perform E step.
        a[:] = pi*pdf_entrywise_1d(x, mu)
        b[:] = (1-pi)*pdf_entrywise_1d(x)
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
    top = alpha*pdf_entrywise_2d(A-mu)
    bottom = top + (1-alpha)*pdf_entrywise_2d(A)
    return top/bottom

############################################################################
# F-measure helpers
############################################################################

def calc_fmeasure(pred_set,act_set):
    true_positives = np.intersect1d(pred_set, act_set)
    prec = len(true_positives) / len(pred_set)
    recall = len(true_positives) / len(act_set)

    if prec == 0 or recall == 0:
        return 0
    else:
        return 2.0/((1.0/prec) + (1.0/recall))

def calc_fmeasure_mat(I_pred, J_pred, I_act, J_act):
    true_positives_I = np.intersect1d(list(I_pred), list(I_act))
    true_positives_J = np.intersect1d(list(J_pred), list(J_act))
    true_positives_num = len(true_positives_I) * len(true_positives_J)

    act_pos_num = len(I_pred) * len(J_pred)
    prec = true_positives_num / act_pos_num
    recall = true_positives_num / act_pos_num

    if prec == 0 or recall == 0:
        return 0
    else:
        return 2.0/((1.0/prec) + (1.0/recall))

############################################################################
############################################################################

# find submatrix of size \approx alpha_n with highest responsibility sum
def submatrix_max_resp(A, alpha_n):
    model = gp.Model()
    model.setParam('OutputFlag', False )

    n = A.shape[0]
    # Add variables to model
    # first n variables are x vector (indicator for rows, i.e. x[i]=1 if row i is selected), 
    # second n variables are y vector (indicator for columns)
    vars = []
    for j in range(2*n):
        vars.append(model.addVar(vtype=gp.GRB.BINARY))

    # Want to maximize x^T A y = sum of submatrix with x as rows and y as columns
    obj = gp.QuadExpr()
    for i in range(n):
        for j in range(n):
            obj += vars[i]*vars[n+j]*A[i,j]
    model.setObjective(obj, sense=gp.GRB.MAXIMIZE)
    model.optimize()

    res = np.zeros([2*n])
    for i in range(2*n):
        res[i] = vars[i].x

    row_inds = []
    col_inds = []

    for i in range(n):
        if vars[i].x > 0:
            row_inds.append(i)
    for j in range(n):
        if vars[n+j].x > 0:
            col_inds.append(j)
    return row_inds, col_inds

############################################################################
############################################################################

def find_submatrix(A):
    A_entries = A.flatten()
    mu_est, alpha_est = single_em(A_entries)
    A_resp_est = compute_responsibilities(A, mu_est, alpha_est)

     # get tau
     A_resp_entries = A_resp_est.flatten()
     sorted_A_entries = np.sort(A_resp_entries)[::-1] #highest to lowest
    if int(alpha_est*n**2) > 0:
        tau = sorted_A_entries[int(alpha_est*n**2)]
    else:
        tau = sorted_A_entries[0]

    A_resp_shifted = A_resp_est - tau
    I_pred, J_pred = submatrix_max_resp(A_resp_shifted, int(alpha_est*(n**2)))

    return I_pred, J_pred