import numpy as np
from scipy.linalg import sqrtm
from env.env import Env
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.gaussian_process.kernels import Matern
import util

class KernelBanditEnv(Env):
    name = 'kernel'
    config = {
        'N': 10,
        'd': 2,
        'kernel': 'matern',
        'T': 10000,
        'mean_reward_bound': 0.5,
    }

    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        C = self.config

        A = kwargs.get('A')
        A, m = kernel_bandit_generator(
            C['N'], C['d'], C['kernel'], reward_bound=C['mean_reward_bound'], A=A)
        self.A = A
        self.m = m

        self.K = util.kernel_matrix(A, C['kernel'])
        self.Phi = np.linalg.cholesky(self.K)

    def mean_reward(self, t, action):
        return self.m[action]

def kernel_bandit_generator(k, d, config, reward_bound=0.5, A=None):
    if A is None:
        A = np.random.randn(k, d)
        A /= np.sqrt(np.square(A).sum(axis=1))[:, np.newaxis]

    K = util.kernel_matrix(A, config)
    m = np.random.multivariate_normal(np.zeros(k), K)
    m = m / np.max(np.abs(m)) * reward_bound
    return A, m
