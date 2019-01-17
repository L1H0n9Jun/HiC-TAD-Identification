#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import division
from bisect import bisect

import numpy as np


def py(label, para0, n):
    ids = np.arange(n) + 1
    ids = ids[::-1]
    ids = np.cumsum(ids)
    ids1 = np.hstack((1, ids[0:(n - 1)] + 1))
    energy_func = 0
    for i in range(len(label)):
        site = bisect(ids1, i + 1)
        x = np.where(ids1 == i + 1)
        x = x[0]
        if label[i] == label[i + 1]:
            energy_func = energy_func + para0
        else:
            energy_func = energy_func - para0
        if len(x) == 0:
            if label[i] == label[i + n - site]:
                energy_func = energy_func + para0
            else:
                energy_func = energy_func - para0
    return energy_func
