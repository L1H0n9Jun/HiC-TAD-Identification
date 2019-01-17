#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import division

from bisect import bisect

import numpy as np


def B_Spline(dist, dist_inter):
    n = len(dist_inter)
    dist_inter = dist_inter.reshape((n, 1))
    site = bisect(dist_inter, dist)
    if site >= n:
        site = n - 1
    B_0 = np.zeros((n - 1, 1))
    B_0[site - 1] = 1
    x1 = (dist * np.ones((n - 2, 1)) - dist_inter[0:(n - 2)]) / (
            dist_inter[1:(n - 1)] - dist_inter[0:(n - 2)]) * B_0[0:(n - 2)]
    x2 = (dist_inter[2:n] - dist * np.ones((n - 2, 1))) / (
            dist_inter[2:n] - dist_inter[1:(n - 1)]) * B_0[1:(n - 1)]
    B_1 = x1 + x2
    x3 = (dist * np.ones((n - 3, 1)) - dist_inter[0:(n - 3)]) / (
            dist_inter[2:(n - 1)] - dist_inter[0:(n - 3)]) * B_1[0:(n - 3)]
    x4 = (dist_inter[3:n] - dist * np.ones((n - 3, 1))) / (
            dist_inter[3:n] - dist_inter[1:(n - 2)]) * B_1[1:(n - 2)]
    B_2 = x3 + x4
    return B_2
