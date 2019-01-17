from __future__ import division

import numpy as np

from tad_ident import Generate_label
from tad_ident import Maximum_Likelihood


def main():
    mat = np.loadtxt("imr90.40Kb.raw.chr1.mat", dtype=np.float64)
    N = np.size(mat, 0)

    x = np.where(mat.any(axis=1) == 0)[0]
    x = np.sort(x)
    l = list()
    i = 0
    while i < len(x):
        j = i + 1
        if j >= len(x):
            break
        while x[j] - x[j - 1] == 1:
            j = j + 1
            if j >= len(x):
                break
        l.append([i, j - 1])
        i = j
    l = np.array(l)
    d = l[:, 1] - l[:, 0]
    y = np.where(d == max(d))[0]
    s = x[l[y, 0][0]]
    e = x[l[y, 1][0]]
    # matrix divide to two parts(centromere)
    mat1 = mat[0:s, 0:s]
    mat2 = mat[(e + 1):N, (e + 1):N]
    # reduce the dimension
    N_HIC1 = np.size(mat1, 0)
    w = 5
    w_l = np.int16(N_HIC1 / w)
    mat1_1 = np.zeros((w_l, w_l))
    for i in range(w_l):
        for j in np.arange(i, w_l):
            mat1_1[i, j] = np.sum(
                mat1[np.arange(i * w, i * w + w - 1), np.arange(j * w, j * w + w - 1)])
            mat1_1[j, i] = mat1_1[i, j]

    N_HIC1 = np.size(mat1_1, 0)
    dist_limit = 400
    D = np.matmul(np.ones((N_HIC1, 1)), np.arange(N_HIC1).reshape((1, N_HIC1))) - np.matmul(
        np.arange(N_HIC1).reshape((N_HIC1, 1)), np.ones((1, N_HIC1)))
    D = np.triu(D)
    x = np.where((D > 0) & (D <= dist_limit))
    x1 = x[0]
    x2 = x[1]
    hic1 = mat1_1[x1, x2]
    Dist1 = D[x1, x2]
    edge1 = list()
    for i in range(len(x1)):
        k1 = np.where((x1 == x1[i]) & (x2 == (x2[i] + 1)))[0]
        k2 = np.where((x1 == (x1[i] + 1)) & (x2 == x2[i]))[0]
        if len(k1) > 0:
            edge1.append([i, k1[0]])
        if len(k2) > 0:
            edge1.append([i, k2[0]])
        print(str(i))
    edge1 = np.array(edge1)
    np.savetxt("edge_1.txt", edge1, fmt="%d")
    N_HIC1 = len(hic1)
    k_nots = 6
    temp = 1
    dist_inter = np.linspace(0, dist_limit, k_nots)
    # EM algorithm

    # initialize parameter
    learning_rate1 = 0.5
    learning_rate2 = 0.01
    para1 = 0.5
    para2 = np.random.rand(2, 5)
    label = np.zeros((N_HIC1))
    label_new = np.ones(N_HIC1)
    iter_num = 0
    while len(np.where(label_new == label)[0]) / N_HIC1 < 0.95:
        label = label_new
        label_new = Generate_label(hic1, Dist1, edge1, dist_inter, para1, para2)
        [para1, para2] = Maximum_Likelihood(hic1, Dist1, dist_inter,
                                            label_new, para1, para2,
                                            N_HIC1, learning_rate1,
                                            learning_rate2)
        learning_rate1 = learning_rate1 * 0.1
        learning_rate2 = learning_rate2 * 0.1
        print("iter_num" + str(iter_num))
        iter_num += 1
    np.savetxt("lable.txt", label_new, fmt="%d")


if __name__ == "__main__":
    main()
