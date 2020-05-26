import sys
import csv
import math
import time

import numpy as np
import networkx as nx

import gurobipy as gp
from gurobipy import GRB

def get_L(G, names, scores):

    n = len(scores)

    # should be ordered according to vertex_to_index
    W = scores
    
    A = np.zeros((n, n))
    D = np.zeros((n, n))
    W = np.zeros(n)
    
    for e in G.edges:
        v0 = e[0]
        v1 = e[1]

        index0 = names.index(v0)
        index1 = names.index(v1)
        
        A[index0][index1] = 1
        A[index1][index0] = 1
    
        D[index0][index0] += 1
        D[index1][index1] += 1

    L = D-A

    return L, W, vertex_dict

def solve_cut(L, W, k, rho):

    n = len(W)
        
    model = gp.Model()
    model.setParam('OutputFlag', False )

    # Add variables to model
    nodes = []
    for j in range(n):
        nodes.append(model.addVar(vtype=GRB.BINARY))

    # Want to maximize x * W
    obj = gp.LinExpr()
    for i in range(n):
        obj += nodes[i] * W[i]/math.sqrt(k)
    model.setObjective(obj, sense=GRB.MAXIMIZE)
    
    # Only k nodes in the subgraph allowed
    expr = gp.LinExpr()
    for i in range(n):
        expr += nodes[i]
    model.addConstr(expr, GRB.LESS_EQUAL, k)
    
    # Subgraph must have cut < rho
    # 1/4 * x * L * x
    expr2 = gp.LinExpr()
    for i in range(n):
        curr_x = nodes[i] - 0.5
        for j in range(n):
            expr2 += curr_x * L[i][j] * (nodes[j] - 0.5)
    model.addConstr(expr2, GRB.LESS_EQUAL, rho)
    
    model.optimize()
    
    
    count = 0
    scores = [None for _ in range(n)]
    node_vals = [0 for _ in range(n)]
    indices = []
    curr_index = 0
    for v in nodes:
        if v.x != 0:
            indices.append(curr_index)
        curr_index += 1

    return indices

def find_cutsize(G, names, scores, rho):

    mu_est, alpha_est = single_em(scores)
    alpha_n_est = int(alpha_est*n)

    L, W, vertex_dict = get_L(G, names, scores)

    indices = solve_cut(L, W, n, alpha_n_est, rho)

    return indices