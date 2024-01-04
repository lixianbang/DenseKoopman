import numpy as np
from scipy import linalg

def H_DMD(X, delay):

    H = np.zeros((delay * X.shape[0], X.shape[1] - delay + 1), dtype=float)

    for k in range(delay):
        H[X.shape[0] * (k) : X.shape[0] * (k+1), :] = X[:, k : X.shape[1]-delay+k+1]

    X1 = H[:, : H.shape[1]-1]
    X2 = H[:, 1 : H.shape[1]]

    U, S, V = linalg.svd(X1, 0)
    S = np.diag(S)
    V = V.T
    #矩阵元素求倒数
    for i in range(S.shape[0]):
        for j in range(S.shape[1]):
            if S[i][j] > 0:
                S[i][j] = 1.0 / S[i][j]

    K = np.dot(U.T, X2)
    K = np.dot(K, V)
    K = np.dot(K, S)

    ##K = U.T * X2 * V * S
    Eigval, y = linalg.eig(K)

    #从列向量变为矩阵
    Eigval = np.diag(Eigval)
    Eigvec = np.dot(U, y)

    bo = np.dot(linalg.pinv(Eigvec), X1[:, 0])
    X = X1
    Y = X2

    return Eigval, Eigvec, bo, X, Y, H
