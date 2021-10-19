import numpy as np
import kmeans
import common
import naive_em
import em
import matplotlib.pyplot as plt


def project4_2():
    X = np.loadtxt("toy_data.txt")
    k = [1, 2, 3, 4]
    s = [0, 1, 2, 3, 4]
    new_cost = 10000
    cost_k = np.zeros((1, 4))
    param = np.zeros((4, 2))
    for K in k:
        for seed in s:
            mixture, post = common.init(X, K, seed)
            mixture, post, cost = kmeans.run(X, mixture, post)
            if cost_k[0, K - 1] == 0 or cost_k[0, K - 1] > cost:
                cost_k[0, K - 1] = cost
                param[K - 1] = K, seed

    print(cost_k)
    print('param:',param)
    param = param.astype(int)
    for ele in param:
        K = ele[0]
        seed = ele[1]
        mixture, post = common.init(X, K, seed)
        mixture, post, cost = kmeans.run(X, mixture, post)

        common.plot(X, mixture, post, 'K means K={}'.format(K))


project4_2()


# common.plot(X,mixture, post,'7')
# naive_em.run()

def project4_3():
    X = np.loadtxt("toy_data.txt")
    k = [1, 2, 3, 4]
    s = [0, 1, 2, 3, 4]
    new_cost = 10000
    cost_k = np.zeros((1, 4))
    param = np.zeros((4, 2))
    for K in k:
        for seed in s:
            mixture, post = common.init(X, K, seed)
            mixture, post, cost = naive_em.run(X, mixture, post)
            if cost_k[0, K - 1] == 0 or cost_k[0, K - 1] < cost:
                cost_k[0, K - 1] = cost
                param[K - 1] = K, seed

    print(cost_k)
    print(param)




# project4_3()
# print(naive_em.x)

def bic():
    bic = np.zeros(4)
    X = np.loadtxt("toy_data.txt")
    loss = np.array([[-1307.22343176, -1175.71462937, -1138.89089969, -1138.6011757]])
    m = np.array([[1, 0],
                  [2, 2],
                  [3, 0],
                  [4, 4]])

    for i in range(4):
        K = m[i][0]
        seed = m[i][1]
        mixture, post = common.init(X, K, seed)
        mixture, post, cost = naive_em.run(X, mixture, post)
        bic[i] = common.bic(X, mixture, cost)

    print(bic)

# bic()

def plot_compare():
    K = 1
    seed = 2
    X = np.loadtxt("toy_data.txt")
    loss = np.array([[-1307.22343176, -1175.71462937, -1138.89089969, -1138.6011757]])
    m = np.array([[1, 0],
                  [2, 2],
                  [3, 0],
                  [4, 4]])
    for ele in m:
        K = ele[0]
        seed = ele[1]
        mixture, post = common.init(X, K, seed)
        mixture, post, cost = naive_em.run(X, mixture, post)

        common.plot(X, mixture, post, 'em K={}'.format(K))

    # fig, axes = plt.subplots(nrows=1, ncols=4)
    # for ax in axes:
    #     ax.common.plot(X,mixture,post)
    #     ax[1].set_xlabel('x')
    #     ax[2].set_ylabel('y')
    #     ax[3].set_title('title')
    # fig.tight_layout()
    # plt.show()
    #
    # fig, axes = plt.subplots(nrows=1, ncols=1)

    # plt.show()


# plot_compare()
