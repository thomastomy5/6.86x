import numpy as np
import em
import common
import naive_em
import matplotlib.pyplot as plt
from numpy import ma
from scipy.stats import multivariate_normal

def test_plot():
    x = np.linspace(0, 5, 10)
    y = x ** 2
    fig, axes = plt.subplots(nrows=2, ncols=4)
    for ax in axes:
        ax[0,1].plot(x, y, 'r')
        ax[1].set_xlabel('x')
        ax[2].set_ylabel('y')
        ax[3].set_title('title')
    fig.tight_layout()
    plt.show()


# test_plot()

def netflix_Estep_withoutzero():
    X = np.loadtxt("test_incomplete.txt")
    X=np.array([[0.37302982, 0.54976417],
                 [0.01295811, 0.12748927],
                 [0.51204134, 0.97574914],
                 [0.98996544, 0.40565087],
                 [0.95640175, 0.72924638],
                 [0.04531841, 0.68490408],
                 [0.14927128, 0.96242917],
                 [0.52535835, 0.48439769],
                 [0.08470057, 0.97103496],
                 [0.39539121, 0.82500599],
                 [0.64797514, 0.70354985],
                 [0.77468475, 0.25650571],
                 [0.45864211, 0.23492838],
                 [0.96556158, 0.52854308],
                 [0.89370853, 0.64646819],
                 [0.8497968 , 0.43954353],
                 [0.97255007, 0.67433588],
                 [0.57440677, 0.25669299]])
    K = 4
    Mu =  np.array([[0.97255007, 0.67433588],
                    [0.52535835, 0.48439769],
                    [0.98996544,0.40565087],
                    [0.8497968,  0.43954353]])
    Var = np.array([0.17419098, 0.09251203, 0.19242047, 0.13740348])
    P = np.array([0.32549117, 0.24495331, 0.23323009, 0.19632544])
    mixture = common.GaussianMixture(Mu, Var, P)

    print(naive_em.estep(X,mixture))

# netflix_Estep()


def netflix_Estep_zero():

    X = np.array([[0.85794562, 0.84725174],
                 [0.6235637,  0.38438171],
                 [0.29753461, 0.05671298],
                 [0.        , 0.47766512],
                 [0.        , 0.        ],
                 [0.3927848 , 0.        ],
                 [0.        , 0.64817187],
                 [0.36824154, 0.        ],
                 [0.        , 0.87008726],
                 [0.47360805, 0.        ],
                 [0.        , 0.        ],
                 [0.        , 0.        ],
                 [0.53737323, 0.75861562],
                 [0.10590761, 0.        ],
                 [0.18633234, 0.        ]])
    K = 6
    Mu = np.array([[0.6235637,  0.38438171],
                 [0.3927848 , 0.        ],
                 [0.        , 0.        ],
                 [0.        , 0.87008726],
                 [0.36824154, 0.        ],
                 [0.10590761, 0.        ]])
    Var = np.array([0.16865269, 0.14023295, 0.1637321,  0.3077471,  0.13718238, 0.14220473])
    P = np.array([0.1680912,  0.15835331, 0.21384187, 0.14223565, 0.14295074, 0.17452722])

    mixture = common.GaussianMixture(Mu, Var, P)

    print(em.fill_matrix(X,mixture))

    # X_C = ma.masked_values(X,0)
    # print(X_C)
    #
    # n, _ = X.shape
    # filter = X!=0

    # for i in range(n):
    #     f =filter[i]
    #     Cu = X[i][f]
    #     Cmu = mixture.mu[:,f]
    #     d = len(Cu)

    # print(X[filter])
    # print(Cmu)


    # print(naive_em.estep(X,mixture))

# netflix_Estep_zero()


# X=np.array([1,1])
# mu = np.array([0,0])
# var=np.array([[1,1],[0,2]])
#
# t =multivariate_normal.pdf(X, mean=mu, cov=var)
# # print(t)
#
# c = np.array([[1,2],
#               [4,5]])
#
# print(np.sum(c,axis=1))
#
# a = np.array([1,2,3])
# b = np.array([[0,1,1],
#               [1,3,2],
#               [1,2,2]])
# print(a@b)

def newest_test():
    X = np.loadtxt("test_incomplete.txt")
    K = 4
    Mu = np.array([[2., 4., 5., 5., 0.],
                   [3., 5., 0., 4., 3.],
                   [2., 5., 4., 4., 2.],
                   [0., 5., 3., 3., 3.]])
    Var = np.array([5.93, 4.87, 3.99, 4.51])
    P = np.array([0.25, 0.25, 0.25, 0.25])
    mixture = common.GaussianMixture(Mu, Var, P)
    #
    post, cost = em.estep_new(X, mixture)
    print(cost)

# newest_test()

def mstep_netflix():
    X = np.array([[0.97057607 ,0.23965268],
                [0.6719088  ,0.],
                [0.97951078,0.32046905],
                [0.5879251 , 0.],
                [0.,0.95258902],
                [0.        , 0.],
                [0.46120321,0.],
                [0.        , 0.],
                [0.,0.],
                [0.        , 0.41528806],
                [0.54819032,0.]])
    K =5

    post = np.array([[0.11209997, 0.06961755, 0.30173412, 0.16812821, 0.34842014],
                    [0.04450801, 0.12209037, 0.26551814, 0.30134896, 0.26653452],
                    [0.10100147,0.34486255,0.2166208,0.22130398,0.11621119],
                    [0.03745738, 0.2411637,  0.13584704, 0.35995272, 0.22557916],
                    [0.21645964,0.26845908,0.15553528,0.17461553,0.18493047],
                    [0.4832334 , 0.0166697,  0.05740243, 0.41596204, 0.02673244],
                    [0.32610403,0.14211143,0.11174741,0.25784375,0.16219339],
                    [0.291663,   0.22921708, 0.09942144, 0.3070474,  0.07265108],
                    [0.25125741,0.21863372,0.08091467,0.28222333,0.16697086],
                    [0.04046758, 0.18054389, 0.2989046,  0.19070597, 0.28937797],
                    [0.00099463,0.22727602,0.34239789,0.30377468,0.12555678]])

    Mu = np.array([[2., 4., 5., 5., 0.],
                   [3., 5., 0., 4., 3.],
                   [2., 5., 4., 4., 2.],
                   [0., 5., 3., 3., 3.]])
    Var = np.array([5.93, 4.87, 3.99, 4.51])
    P = np.array([0.25, 0.25, 0.25, 0.25])
    mixture = common.GaussianMixture(Mu, Var, P)

    print(em.mstep(X,post,mixture))

# mstep_netflix()

# model = {}
# for i in range(1, 6+1):
#     for j in range(1, 6+1):
#         model[(i, j)] = 1/36
#
# print(model)
#
# for ele in model.keys():
#     print(ele[0])































