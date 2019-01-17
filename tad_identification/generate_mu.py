#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import division

import sys

import numpy as np

sys.path.append('tad_identification')
import B_Spline


def generate_mu(hic, D, dist_inter, para2):
    n = len(hic)
    mu = np.zeros((2, n))
    for i in range(n):
        dis_fit = B_Spline(D[i], dist_inter)
        dis_fit = dis_fit.reshape((len(dis_fit)))
        mu[:, i] = np.exp(np.sum(dis_fit * para2[:, 2:np.size(para2, 1)], 1)
                          + para2[:, 1])
    return mu
