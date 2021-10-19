"""Mixture model for matrix completion"""
from typing import Tuple
import numpy as np
from scipy.special import logsumexp
from common import GaussianMixture
from scipy.special import logsumexp
from scipy.stats import multivariate_normal


def estep(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.ndarray, float]:
    """E-step: Softly assigns each datapoint to a gaussian component

    Args:
        X: (n, d) array holding the data, with incomplete entries (set to 0)
        mixture: the current gaussian mixture

    Returns:
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the assignment

    """
    n, _ = X.shape
    K, _ = mixture.mu.shape
    mu = mixture.mu
    post = np.zeros((n, K))
    var = mixture.var
    # var = np.sqrt(mixture.var)
    P = mixture.p

    for i in range(n):
        for j in range(K):
            post[i][j] = P[j] * multivariate_normal.pdf(X[i, :], mean=mu[j], cov=var[j])

            # post[i][j] = log_func(X[i, :], P[j], mu[j], var[j])

    # log_sum = logsumexp(post, axis=1)
    # post = post.T - log_sum

    s = np.sum(post, axis=1, keepdims=True)
    post = post / s
    cost = np.sum(np.log(s))
    # cost =1
    return post.T, cost


def log_func(X, P, mu, var):
    P = np.log(P + 1e-16)
    return P + multivariate_normal.pdf(X, mean=mu, cov=var)


def mstep(X: np.ndarray, post: np.ndarray, mixture: GaussianMixture,
          min_variance: float = .25) -> GaussianMixture:
    """M-step: Updates the gaussian mixture by maximizing the log-likelihood
    of the weighted dataset

    Args:
        X: (n, d) array holding the data, with incomplete entries (set to 0)
        post: (n, K) array holding the soft counts
            for all components for all examples
        mixture: the current gaussian mixture
        min_variance: the minimum variance for each gaussian

    Returns:
        GaussianMixture: the new gaussian mixture
    """
    n, d = X.shape
    _, K = post.shape
    # post = post[X!= 0]
    # # mu_hat = mixture.mu.copy()
    # filter = X != 0
    # f = np.zeros((1, d))
    #
    # for i in range(n):
    #     f = filter[i]
    #     for j in range(K):
    #         Cmu = mu[j][f]
    #         XC = X[i][f]

    n_hat = post.sum(axis=0)
    p = n_hat / n

    cost = 0
    mu = np.zeros((K, d))
    var = np.zeros(K)

    for j in range(K):
        mu[j, :] = post[:, j] @ X / n_hat[j]
        sse = ((mu[j] - X) ** 2).sum(axis=1) @ post[:, j]
        cost += sse
        var[j] = sse / (d * n_hat[j])
        var[j] = np.where(var[j]<0.25,0.25,var[j])

    return GaussianMixture(mu, var, p)



def run(X: np.ndarray, mixture: GaussianMixture,
        post: np.ndarray) -> Tuple[GaussianMixture, np.ndarray, float]:
    """Runs the mixture model

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the current assignment
    """
    prev_cost = None
    cost = None
    while (prev_cost is None or cost - prev_cost > 1e-6 * np.abs(cost)):
        prev_cost = cost
        post, cost = estep(X, mixture)
        mixture = mstep(X, post)

    return mixture, post, cost


def fill_matrix(X: np.ndarray, mixture: GaussianMixture) -> np.ndarray:
    """Fills an incomplete matrix according to a mixture model

    Args:
        X: (n, d) array of incomplete data (incomplete entries =0)
        mixture: a mixture of gaussians

    Returns
        np.ndarray: a (n, d) array with completed data
    """
    mu_new = mixture.mu.copy()
    X_new = X.copy()
    n, d = X.shape
    K, _ = mixture.mu.shape

    mu = mixture.mu
    post = np.zeros((n, K))
    var = mixture.var
    P=mixture.p

    filter = X!= 0
    f = np.zeros((1,d))

    for i in range(n):
        f = filter[i]
        for j in range(K):
            Cmu = mu[j][f]
            XC = X[i][f]
            if np.all(XC==0):
                post[i][j]=P[j]
            else:
                post[i][j] = P[j]*multivariate_normal.pdf(XC,mean=Cmu, cov=var[j])

    s = np.sum(post,axis=1,keepdims=True)
    post = post / s
    cost = np.sum(np.log(s))

    for a in range(n):
        for b in range(d):
            if X_new[a][b]==0:
                t = post[a]@mu_new
                X_new[a][b] = t[b]

    return X_new


def estep_new(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.ndarray, float]:
    """E-step: Softly assigns each datapoint to a gaussian component

    Args:
        X: (n, d) array holding the data, with incomplete entries (set to 0)
        mixture: the current gaussian mixture

    Returns:
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the assignment"""

    mu_new = mixture.mu.copy()
    X_new = X.copy()
    n, d = X.shape
    K, _ = mixture.mu.shape
    mu = mixture.mu
    post = np.zeros((n, K))
    var = mixture.var
    # var = np.sqrt(mixture.var)
    P=mixture.p

    filter = X!= 0
    f = np.zeros((1,d))

    for i in range(n):
        f = filter[i]
        for j in range(K):
            Cmu = mu[j][f]
            XC = X[i][f]
            if np.all(XC==0):
                post[i][j]=P[j]
            else:
                post[i][j] = P[j]*multivariate_normal.pdf(XC,mean=Cmu, cov=var[j])
            # post[i][j] = P[j]*(1 / ((2 * np.pi * sd[j] * sd[j]) ** (n / 2))) *np.exp(-0.5 * ((np.linalg.norm(X[i] - mean[j]) * 2) / (sd[j] * 2)))

    s = np.sum(post,axis=1,keepdims=True)
    post = post / s
    cost = np.sum(np.log(s))

    for a in range(n):
        for b in range(d):
            if X_new[a][b]==0:
                t = post[a]@mu_new
                X_new[a][b] = t[b]

    return X_new

