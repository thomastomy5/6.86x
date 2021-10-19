import numpy as np


def prob_1():
    S_0 = np.array([[0, 0]]).T
    W_ss = np.array([[-1, 0], [0, 1]])
    W_sx = np.array([[1, 0], [0, 1]])

    X_1 = np.array([[0, 1]]).T
    X_2 = np.array([[1, 0]]).T
    X_3 = np.array([[1, 0]]).T

    S_1 = np.matmul(W_ss, S_0) + np.matmul(W_sx, X_1)
    S_1 = np.where(S_1 > 0, S_1, 0)

    S_2 = np.matmul(W_ss, S_1) + np.matmul(W_sx, X_2)
    S_2 = np.where(S_2 > 0, S_2, 0)

    S_3 = np.matmul(W_ss, S_2) + np.matmul(W_sx, X_3)
    S_3 = np.where(S_3 > 0, S_3, 0)

    print(S_3)


# prob_1()


def prob_2():
    W_zy = np.array([[1, -1], [3, -1]])
    z = np.array([[1, 1]]).T
    n, _ = np.shape(z)

    y = np.matmul(W_zy, z)
    y = np.where(y > 0, 1, 0)
    print(y)

    # S_0 = np.zeros((n,1))
    S_0 = np.ones((n,1))
    W_ss = np.identity(n)
    W_sx = -1*np.identity(n)
    W_sy = 1*W_zy
    W_0 = np.sum(W_zy)*1
    # W_0=0

    X_1 = np.array([[0, 1]]).T
    X_2 = np.array([[1, 0]]).T
    X_3 = np.array([[1, 0]]).T

    S_1 = np.matmul(W_ss, S_0) + np.matmul(W_sx, X_1)
    S_1 = np.where(S_1 > 0, S_1, 0)

    S_2 = np.matmul(W_ss, S_1) + np.matmul(W_sx, X_2)
    S_2 = np.where(S_2 > 0, S_2, 0)

    S_3 = np.matmul(W_ss, S_2) + np.matmul(W_sx, X_3)
    S_3 = np.where(S_3 > 0, S_3, 0)

    y_new = np.matmul(W_sy, S_3) + W_0
    y_new = np.where(y_new > 0, 1, 0)
    print(y_new)



prob_2()

