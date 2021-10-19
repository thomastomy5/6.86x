"""Mixture model using EM"""
from typing import Tuple
import numpy as np
from common import GaussianMixture
from scipy.stats import multivariate_normal




def estep(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.ndarray, float]:
    """E-step: Softly assigns each datapoint to a gaussian component

    Args:
        X: (n, d) array holding the data
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
    P=mixture.p

    for i in range(n):
        for j in range(K):
            post[i][j] = P[j]*multivariate_normal.pdf(X[i,:],mean=mu[j], cov=var[j])
            # post[i][j] = P[j]*(1 / ((2 * np.pi * sd[j] * sd[j]) ** (n / 2))) *np.exp(-0.5 * ((np.linalg.norm(X[i] - mean[j]) * 2) / (sd[j] * 2)))

    s = np.sum(post,axis=1,keepdims=True)
    post = post / s
    cost = np.sum(np.log(s))

    return post, cost

def mstep(X: np.ndarray, post: np.ndarray) -> GaussianMixture:
    """M-step: Updates the gaussian mixture by maximizing the log-likelihood
    of the weighted dataset

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
    """
    n, d = X.shape
    _, K = post.shape

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
    while (prev_cost is None or prev_cost - cost < 1e-6 * prev_cost):
        prev_cost = cost
        post, cost = estep(X, mixture)
        mixture = mstep(X, post)

    return mixture, post, cost

x= 2
