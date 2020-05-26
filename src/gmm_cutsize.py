import sys
import csv
import math
import time

import numpy as np
import networkx as nx

import gurobipy as gp
from gurobipy import GRB

def get_L(G, scores_dict):

    n = len(scores_dict)
    
    # maps vertices to indices
    index_dict = {}
    
    # maps indices to vertices
    vertex_dict = {}
    
    A = np.zeros((n, n))
    D = np.zeros((n, n))
    W = np.zeros(n)
    
    avail_index = 0
    
    for e in G.edges:
        v0 = e[0]
        v1 = e[1]

        index0 = index_dict.get(v0)
        if index0 is None:
            index_dict[v0] = avail_index
            vertex_dict[avail_index] = v0
            index0 = avail_index
            avail_index += 1

        index1 = index_dict.get(v1)
        if index1 is None:
            index_dict[v1] = avail_index
            vertex_dict[avail_index] = v1
            index1 = avail_index
            avail_index += 1

    A[index0][index1] = 1
    A[index1][index0] = 1
                    
    D[index0][index0] += 1
    D[index1][index1] += 1

    L = D-A

    for i in range(n):
        W[i] = scores_dict[vertex_dict[i]]
    
    # just some checks
    if avail_index != n:
        print("avail_index != n; too many or too few vertices indexed")
        return None, None, None
    
    return L, W, vertex_dict

def solve_cut(L, W, k, rho):

    n = len(W)
        
    model = gp.Model()
    model.setParam('OutputFlag', False )

    # Add variables to model
    nodes = []
    for j in range(n):
        nodes.append(model.addVar(vtype=GRB.BINARY))

    print("added variables")
    # Want to maximize x * W
    obj = gp.LinExpr()
    for i in range(n):
        obj += nodes[i] * W[i]/math.sqrt(k)
    model.setObjective(obj, sense=GRB.MAXIMIZE)
    
    print("added obj")
    
    # Only k nodes in the subgraph allowed
    expr = gp.LinExpr()
    for i in range(n):
        expr += nodes[i]
    model.addConstr(expr, GRB.LESS_EQUAL, k)
    
    print("added constr")
    
    # Subgraph must have cut < rho
    # 1/4 * x * L * x
    expr2 = gp.LinExpr()
    for i in range(n):
        curr_x = nodes[i] - 0.5
        for j in range(n):
            expr2 += curr_x * L[i][j] * (nodes[j] - 0.5)
    model.addConstr(expr2, GRB.LESS_EQUAL, rho)
    
    print("added constr2")
    
    model.optimize()
    
    print("optimized")
    
    count = 0
    scores = [None for _ in range(n)]
    node_vals = [0 for _ in range(n)]
    indices = []
    curr_index = 0
    for v in nodes:
        if v.x != 0:
            indices.append(curr_index)
        curr_index += 1

    return indnices

def find_cutsize(G, scores_dict, rho):
    mu_est, alpha_est = single_em(scores_dict.values())
    alpha_n_est = int(alpha_est*n)
    L, W, vertex_dict = get_L(G, scores_dict)
    indices = solve_cut(L, W, n, alpha_n_est, rho)
    return indices