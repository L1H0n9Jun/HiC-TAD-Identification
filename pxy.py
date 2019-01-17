#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import division
import numpy as np


def pxy(hic, mu_1, par):
    """
    ga1 = gamma(hic_w+np.int32(par**(-1)))
    ga2 = gamma(hic_w+1)
    ga3 = gamma(np.int32(par**(-1)))
    t1 = (mu_1*par)/(1+mu_1*par)
    pxy1=ga1*t1**hic_w*(1-t1)**(par**(-1))/(ga2*ga3)
    """
    ga = np.zeros((len(hic)))
    for i in range(len(hic)):
        if hic[i] == 0:
            ga[i] = 0
        else:
            ga[i] = np.sum(np.log(np.arange(np.int64(par ** (-1)),
                                            hic[i] + np.int64(par ** (-1))))) \
                    - np.sum(np.log(np.arange(1, hic[i] + 1)))
    t1 = hic * np.log((mu_1 * par) / (mu_1 * par + 1))
    t2 = -1 / par * np.log(1 + mu_1 * par)
    pxy1 = ga + t1 + t2
    return pxy1


"""
def generate_edge(n, N_HIC):
    EDGE = list()
    ids = np.arange(n)+1
    ids = ids[::-1]
    ids = np.cumsum(ids)
    ids1 = np.hstack((1,ids[0:(n-1)]+1))
    for i in range(n):
        x = np.where(ids == (i+1)); x = x[0]
        site = bisect(ids1, i+1)
        EDGE.append([i, i+1])
        if len(x)==0:
            EDGE.append([i, i+N_HIC-site])
    return(EDGE)
"""
