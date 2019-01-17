#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import division

import random
import sys

import numpy as np
from scipy.special import gamma

sys.path.append('tad_identification')
import pxy


def MH(hic, par, mu1, mu2, p1, flag):
    A = [par]
    sigma = 0.005  # array of stepsizes
    accepted = 0.0
    # Metropolis-Hastings with 10,000 iterations.
    for n in range(10000):
        old = A[len(A) - 1]  # old parameter value as array
        # Suggest new candidate from Gaussian proposal distribution.
        new = random.gauss(old, sigma)
        if flag == 1:
            old_loglik = np.sum(pxy.pxy(hic, mu1, old)) + np.log(
                1 ** 0.1 * old ** (0.1 - 1) * np.exp(-old * 1) / gamma(0.1))
            new_loglik = np.sum(pxy.pxy(hic, mu1, new)) + np.log(
                1 ** 0.1 * new ** (0.1 - 1) * np.exp(-new * 1) / gamma(0.1))
        else:
            if flag == 2:
                old_loglik = np.sum(
                    np.log(old + (1 - old) * (1 / (mu1 * p1 + 1)) ** (1 / p1))) + np.sum(
                    np.log(1 - old) + pxy.pxy(hic, mu2, p1))
                new_loglik = np.sum(
                    np.log(new + (1 - new) * (1 / (mu1 * p1 + 1)) ** (1 / p1))) + np.sum(
                    np.log(1 - new) + pxy.pxy(hic, mu2, p1))
            else:
                old_loglik = np.sum(
                    np.log(p1 + (1 - p1) * (1 / (mu1 * old + 1)) ** (1 / old))) + np.sum(
                    np.log(1 - p1) + pxy.pxy(hic, mu2, old)) + np.log(
                    1 ** 0.1 * old ** (0.1 - 1) * np.exp(-old * 1) / gamma(0.1))
                new_loglik = np.sum(
                    np.log(p1 + (1 - p1) * (1 / (mu1 * new + 1)) ** (1 / new))) + np.sum(
                    np.log(1 - p1) + pxy.pxy(hic, mu2, new)) + np.log(
                    1 ** 0.1 * new ** (0.1 - 1) * np.exp(-new * 1) / gamma(0.1))
        # Accept new candidate in Monte-Carlo fashing.
        if new_loglik > old_loglik:
            A.append(new)
            accepted = accepted + 1.0  # monitor acceptance
        else:
            u = random.uniform(0.0, 1.0)
            if u < np.exp(new_loglik - old_loglik):
                A.append(new)
                accepted = accepted + 1.0  # monitor acceptance
            else:
                A.append(old)
    print("alph1 Acceptance rate = " + str(accepted / 10000.0))
    A = np.array(A)

    return A
