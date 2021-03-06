import numpy as np
import em
import common
import naive_em
from scipy.stats import multivariate_normal

from scipy.stats import multivariate_normal

x = np.linspace(0, 5, 10, endpoint=False)
y = multivariate_normal.pdf(x, mean=2.5, cov=0.5)


X_gold = np.loadtxt("test_complete.txt")

# t = np.loadtxt("test_solutions.txt",dtype=float,delimiter='.',usecols=(1,2,3,4), skiprows=2, max_rows=20)

# t = np.loadtxt("test_solutions.txt",skiprows=2, max_rows=20)




K = 4
# n, d = X.shape
seed = 0

def naive_em_test():



 X= np.array([[0.85794562, 0.84725174],
  [0.6235637,  0.38438171],
  [0.29753461, 0.05671298],
  [0.27265629, 0.47766512],
  [0.81216873, 0.47997717],
  [0.3927848,  0.83607876],
  [0.33739616, 0.64817187],
  [0.36824154, 0.95715516],
  [0.14035078, 0.87008726],
  [0.47360805, 0.80091075],
  [0.52047748, 0.67887953],
  [0.72063265, 0.58201979],
  [0.53737323, 0.75861562],
  [0.10590761, 0.47360042],
  [0.18633234, 0.73691818]])

 # X = np.loadtxt("test_solutions.txt",dtype=str)

 K= 6
 #
 # Mu= np.array([[0.6235637,  0.38438171],
 #  [0.3927848,  0.83607876],
 #  [0.81216873, 0.47997717],
 #  [0.14035078, 0.87008726],
 #  [0.36824154, 0.95715516],
 #  [0.10590761, 0.47360042]])
 # Var= np.array([0.10038354, 0.07227467, 0.13240693, 0.12411825, 0.10497521, 0.12220856])
 # P= np.array([0.1680912,  0.15835331, 0.21384187, 0.14223565, 0.14295074, 0.17452722])
 # #

 X = np.loadtxt("test_incomplete.txt")
 K= 4
 Mu=np.array([[2., 4., 5., 5., 0.],
  [3., 5., 0., 4., 3.],
  [2., 5., 4., 4., 2.],
  [0., 5., 3., 3., 3.]])
 Var= np.array([5.93, 4.87, 3.99, 4.51])
 P= np.array([0.25, 0.25, 0.25, 0.25])
 mixture = common.GaussianMixture(Mu,Var,P)
 #
 post, cost = naive_em.estep(X , mixture)
 print(cost)

# naive_em_test()

def testing_pdf():
 x = np.array([[0.2],[-0.9],[-1],[1.2],[1.8]])
 mu = np.array([[-3],[2]])
 var= np.array([[4],[4]])
 P = np.array([0.5,0.5])
 c=np.zeros((5,2))
 for i in range(5):
  for j in range(2):
   c[i][j] = P[j]*multivariate_normal.pdf(x[i,:],mean=mu[j], cov=var[j])
   # print(c)

 s = np.sum(c,axis=1,keepdims=True)
 c = c/s
 print(c)
 inter = np.log(s)
 # inter = np.sum(c,axis=1)
 print('inter=',inter)
 # llh = np.log(inter)
 # print('llh=',llh)
 print('llh=',np.sum(inter))

# testing_pdf()

def naive_em_test_2():
 X = np.array([[0.85794562, 0.84725174],
               [0.6235637, 0.38438171],
               [0.29753461, 0.05671298],
               [0.27265629, 0.47766512],
               [0.81216873, 0.47997717],
               [0.3927848, 0.83607876],
               [0.33739616, 0.64817187],
               [0.36824154, 0.95715516],
               [0.14035078, 0.87008726],
               [0.47360805, 0.80091075],
               [0.52047748, 0.67887953],
               [0.72063265, 0.58201979],
               [0.53737323, 0.75861562],
               [0.10590761, 0.47360042],
               [0.18633234, 0.73691818]])

 Mu = np.array([[0.6235637,  0.38438171],
  [0.3927848,  0.83607876],
  [0.81216873, 0.47997717],
  [0.14035078, 0.87008726],
  [0.36824154, 0.95715516],
  [0.10590761, 0.47360042]])

 Var = np.array([0.10038354, 0.07227467, 0.13240693, 0.12411825, 0.10497521, 0.12220856])

 P =  np.array([0.1680912,  0.15835331, 0.21384187, 0.14223565, 0.14295074, 0.17452722])

 mixture = common.GaussianMixture(Mu, Var, P)
 #
 post, cost = naive_em.estep(X, mixture)
 print(cost)
 print('bic')
 print(common.bic(X,mixture,cost))

# naive_em_test_2()


def m_step():
    post = np.array([[0.17713577, 0.12995693, 0.43161668, 0.26129062],
     [0.08790299, 0.35848927, 0.41566414, 0.13794359],
     [0.15529703, 0.10542632, 0.5030648,  0.23621184],
     [0.23290326,0.10485918, 0.58720619, 0.07503136],
     [0.09060401, 0.41569201, 0.32452345, 0.16918054],
     [0.07639077, 0.08473656, 0.41423836, 0.42463432],
     [0.21838413, 0.20787523, 0.41319756, 0.16054307],
     [0.16534478, 0.04759109, 0.63399833, 0.1530658 ],
     [0.05486073, 0.13290982, 0.37956674, 0.43266271],
     [0.08779356, 0.28748372, 0.37049225, 0.25423047],
     [0.07715067, 0.18612696, 0.50647898, 0.23024339],
     [0.16678427, 0.07789806, 0.45643509, 0.29888258],
     [0.08544132, 0.24851049, 0.53837544, 0.12767275],
     [0.17773171, 0.19578852, 0.41091504, 0.21556473],
     [0.02553529, 0.1258932 , 0.29235844, 0.55621307],
     [0.07604748, 0.19032469, 0.54189543, 0.1917324 ],
     [0.15623582, 0.31418901, 0.41418177, 0.1153934 ],
     [0.19275595, 0.13517877, 0.56734832, 0.10471696],
     [0.33228594, 0.02780214, 0.50397264, 0.13593928],
     [0.12546781, 0.05835499, 0.60962919, 0.20654801]])

    X = np.loadtxt('test_complete.txt')

    print(naive_em.mstep(X,post))

m_step()







