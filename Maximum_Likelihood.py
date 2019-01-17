#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import division
import random

import numpy as np

import B_Spline
import MH
import generate_mu


def Maximum_Likelihood(hic, D, dis_inter, label, para1, para2, N_HIC, l_r1, l_r2):
    mu = generate_mu(hic, D, dis_inter, para2)
    N_PARA = np.size(para2, 1)

    """ 
    #参数sigma
    edge = generate_edge(n, N_HIC)
    grad0 = 0
    for i in range(len(edge)):
        grad0 = grad0 + (hic[edge[i][0]]-hic[edge[i][1]])**2*label[edge[i][0]]
        *label[edge[i][1]]/para0**3
    ##Gibbs sampling for p(y)
    """

    ############
    # parameter for 1
    p2 = np.zeros((2, N_PARA))
    grad_beta = np.zeros((2, N_PARA - 1))
    x = np.where(label == 1);
    x = x[0]
    if len(x) > 0:
        random.shuffle(x)
        sample_num = np.random.randint(10, 100)
        x = x[range(sample_num)]
        # MH alpha
        A = MH(hic[x], para2[1, 0], mu[1, x], 0, 0, 1)
        ids = len(A) - 1
        while A[ids] < 0:
            ids = ids - 1
        p2[1, 0] = A[ids]
        for i in x:
            # parameter mu1(beta)
            a = (hic[i] - mu[1, i]) / (1 + mu[1, i] * p2[1, 0])
            grad_beta[1, 0] += a
            b_s = B_Spline(D[i], dis_inter)
            b_s = b_s.reshape((len(b_s)))
            grad_beta[1, 1:(N_PARA - 1)] += a * b_s
    else:
        p2[1, 0] = para2[1, 0]
    ###########################


    ##parmeter for -1
    x = np.where(label == -1);
    x = x[0]
    if len(x) > 0:
        random.shuffle(x)
        sample_num = np.random.randint(10, 100)
        x = x[range(sample_num)]
        x1 = x[np.where(hic[x] == 0)[0]];
        x2 = x[np.where(hic[x] > 0)[0]]
        # parameter pi
        A = MH(hic[x2], para1, mu[0, x1], mu[0, x2], para2[0, 0], 2)
        ids = len(A) - 1
        while A[ids] < 0:
            ids = ids - 1
        p1 = A[ids]
        # parameter dispersion(alph)
        A = MH(hic[x2], para2[0, 0], mu[0, x1], mu[0, x2], p1, 3)
        ids = len(A) - 1
        while A[ids] < 0:
            ids = ids - 1
        p2[0, 0] = A[ids]
        for i in x1:
            a = (-mu[0, i]) / (p1 * (1 + mu[0, i] * p2[0, 0]) ** (1 / p2[0, 0] + 1) + (1 - p1) * (
                        1 + mu[0, i] * p2[0, 0]))
            grad_beta[0, 0] += a
            b_s = B_Spline(D[i], dis_inter)
            b_s = b_s.reshape((len(b_s)))
            grad_beta[0, 1:(N_PARA - 1)] += a * b_s
        for i in x2:
            # parameter mu1(beta)
            a = (hic[i] - mu[0, i]) / (1 + mu[0, i] * p2[0, 0])
            grad_beta[0, 0] += a
            b_s = B_Spline(D[i], dis_inter)
            b_s = b_s.reshape((len(b_s)))
            grad_beta[0, 1:(N_PARA - 1)] += a * b_s
    else:
        p1 = para1
        p2[0, 0] = para2[0, 0]
    Beta = para2[:, 1:N_PARA] + grad_beta * l_r2
    p2[:, 1:N_PARA] = Beta
    return p1, p2
