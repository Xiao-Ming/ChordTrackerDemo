#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  6 13:49:41 2018

@author: wuyiming
"""

import numpy as np
from sklearn.decomposition import non_negative_factorization

def nmf_beta(V, k, W, H=None, beta=2, iteration=50, verbose=False):
    """
    NMF on beta-divergence, where W is fixed and updates H only
    """
    f = V.shape[0]
    t = V.shape[1]
    if H is None:
        H = np.random.uniform(size=(k, t))
    #e = np.ones(H.shape)
    for i in range(iteration):
        H = H * (np.dot(W.T, np.dot(W, H)**(beta-2)*V)/np.dot(W.T, np.dot(W, H)**(beta-1)))
        if verbose and (i%10 == 0):
            prod = np.dot(W, H)
            cost = np.sum((V**beta+(beta-1)*(prod**beta)-beta*((V*prod)**(beta-1))) / (beta*(beta-1)))
            print("iteration %d: cost=%.3f" % (i, cost))

    return W, H

def nmf_sklearn(V, k, W=None, H=None, beta_loss="frobenius", verbose=False):
    """
    NMF with sklearn.
    """
    f = V.shape[0]
    t = V.shape[1]
    if W is None:
        W = np.random.uniform(size=(f, k))
    if H is None:
        H = np.random.uniform(size=(k, t))

    W, H, _ = non_negative_factorization(V, W, H, k, init="custom", solver="mu", beta_loss=beta_loss, verbose=verbose)

    return W, H

def nmf_euclidean(V, k, W=None, H=None, iteration=50, verbose=False):
    """
    NMF on Euclidean distance.
    """
    f = V.shape[0]
    t = V.shape[1]
    if W is None:
        W = np.random.uniform(size=(f, k))
    if H is None:
        H = np.random.uniform(size=(k, t))

    for i in range(iteration):
        H = H * (np.dot(W.T, V)/np.dot(W.T, np.dot(W, H)))
        W = W * (np.dot(V, H.T)/np.dot(np.dot(W, H), H.T))
        if verbose and (i%10 == 0):
            cost = np.sum((V-np.dot(W, H)) ** 2)
            print("iteration %d: cost=%.3f" % (i, cost))

    return W, H

def nmf_KL(V, k, W=None, H=None, iteration=50, verbose=False):
    """
    NMF on KL divergence
    """
    f = V.shape[0]
    t = V.shape[1]
    if W is None:
        W = np.random.uniform(size=(f, k))
    if H is None:
        H = np.random.uniform(size=(k, t))

    for i in range(iteration):
        H = H * (np.dot(V/np.dot(W, H), H.T)/W.T)
        W = W * (np.dot(W.T, V/np.dot(W, H))/H.T)
        if verbose and (i%10 == 0):
            cost = (V*np.log(V/np.dot(W, H))) - V + np.dot(W, H)
            print("iteration %d: cost=%.3f" % (i, cost))

    return W, H

def nmf_IS(V, k, W=None, H=None, iteration=50, verbose=False):
    """
    NMF on IS divergence
    """
    f = V.shape[0]
    t = V.shape[1]
    if W is None:
        W = np.random.uniform(size=(f, k))
    if H is None:
        H = np.random.uniform(size=(k, t))

    for i in range(iteration):
        H = H * (np.dot(V/np.dot(W, H), H.T)/W.T)
        W = W * (np.dot(W.T, V/np.dot(W, H))/H.T)
        if verbose and (i%10 == 0):
            cost = (V*np.log(V/np.dot(W, H))) - V + np.dot(W, H)
            print("iteration %d: cost=%.3f" % (i, cost))

    return W, H
