#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import division

import sys

import networkx as nx
import numpy as np

sys.path.append('tad_identification')

import generate_mu
import pxy


def Generate_label(hic, D, edge, dist_inter, para1, para2):
    # calculate p(x|y)
    n = len(hic)
    mu = generate_mu.generate_mu(hic, D, dist_inter, para2)
    # for y=1, t-link between s and pixel
    alph = para2[1, 0]
    PXY_s = pxy.pxy(hic, mu[1, :], alph)
    # for y=-1
    PXY_t = np.zeros(n)
    x1 = np.where(hic > 0)
    x1 = x1[0]
    x2 = np.where(hic == 0)
    x2 = x2[0]
    alph = para2[0, 0]
    # pxy_2 = (1-para1)/(1-(1+mu[0, x1]*par)**(-1/par))*pxy_2
    PXY_t[x1] = np.log(1 - para1) + pxy.pxy(hic[x1], mu[0, x1], alph)
    PXY_t[x2] = np.log(para1 + (1 - para1) * (1 / (1 + mu[0, x2] * alph)) ** (1 / alph))
    # alpha = para2[0, 0]
    # PXY_t = pxy.pxy(hic, mu[0, :], alpha)
    # t-link weights
    ws = -PXY_t
    wt = -PXY_s
    x = np.where(D <= 2)
    ws[x] = max(np.max(ws), np.max(wt))
    wt[x] = 0
    x = np.where(D > 300)
    wt[x] = max(np.max(ws), np.max(wt))
    ws[x] = 0
    # min-cut graph
    label = np.zeros((n))
    G = nx.Graph()
    G.add_nodes_from([np.str(i) for i in range(n)] + ['s', 't'])
    for i in range(n):
        G.add_edge(np.str(i), 's', capacity=ws[i])
        G.add_edge(np.str(i), 't', capacity=wt[i])
    for i in range(len(edge)):
        # G.add_edge(edge[i][0], edge[i][1], capacity=np.exp(
        # -(hic[edge[i][0]]-hic[edge[i][1]])**2/(2*para0**2)))
        G.add_edge(np.str(edge[i, 0]), np.str(edge[i, 1]), capacity=1)
    cut_value, partition = nx.minimum_cut(G, 's', 't', capacity='capacity')
    s = np.array(list(partition[0]))
    t = np.array(list(partition[1]))
    x = np.where(s == 's')[0][0]
    s = np.delete(s, x)
    x = np.where(t == 't')[0][0]
    t = np.delete(t, x)
    label[np.int64(s)] = 1
    label[np.int64(t)] = -1
    return label
