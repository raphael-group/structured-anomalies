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

################################################################################
#
# IO functions
#
################################################################################

def is_number(x):
    try:
        float(x)
        return True
    except ValueError:
        return False

def save_nodes(filename, nodes):
    '''
    Load nodes.
    '''
    with open(filename, 'w') as f:
        f.write('\n'.join(str(node) for node in nodes))

def load_node_score(filename):
    '''
    Load node scores.
    '''
    nodes = []
    scores = []
    node_to_score = dict()
    with open(filename, 'r') as f:
        for l in f:
            if not l.startswith('#'):
                arrs = l.strip().split()
                if len(arrs)==2:
                    node = arrs[0]
                    if is_number(arrs[1]):
                        score = float(arrs[1])
                        if np.isfinite(score):
                            nodes.append(node)
                            scores.append(score)
                        else:
                            raise Warning('{} is not a valid node score; input line omitted.'.format(l.strip()))    
                    else:
                        raise Warning('{} is not a valid node score; input line omitted.'.format(l.strip()))
                elif arrs:
                    raise Warning('{} is not a valid node score; input line omitted.'.format(l.strip()))

    if len(nodes) == 0:
        raise Exception('No node scores; check {}.'.format(filename))

    return nodes, scores

def load_edge_list(filename):
    '''
    Load edge list.
    '''
    edge_list = list()
    with open(filename, 'r') as f:
        for l in f:
            if not l.startswith('#'):
                arrs = l.strip().split()
                if len(arrs)>=2:
                    u, v = arrs[:2]
                    edge_list.append((u, v))
                elif arrs:
                    raise Warning('{} is not a valid edge; input line omitted.'.format(l.strip()))

    if not edge_list:
        raise Exception('Edge list has no edges; check {}.'.format(filename))

    return edge_list

def load_matrix(filename, matrix_name='A', dtype=np.float32):
    '''
    Load matrix.
    '''
    import h5py

    f = h5py.File(filename, 'r')
    if matrix_name in f:
        A = np.asarray(f[matrix_name].value, dtype=dtype)
    else:
        raise KeyError('Matrix {} is not in {}.'.format(matrix_name, filename))
    f.close()
    return A

def save_matrix(filename, A, matrix_name='A', dtype=np.float32):
    '''
    Save matrix.
    '''
    import h5py

    f = h5py.File(filename, 'a')
    if matrix_name in f:
        del f[matrix_name]
    f[matrix_name] = np.asarray(A, dtype=dtype)
    f.close()

def status(message=''):
    '''
    Write status message to screen; overwrite previous status message and do not
    advance line.
    '''
    import sys

    try:
        length = status.length
    except AttributeError:
        length = 0

    sys.stdout.write('\r'+' '*length + '\r'+str(message))
    sys.stdout.flush()
    status.length = max(len(str(message).expandtabs()), length)