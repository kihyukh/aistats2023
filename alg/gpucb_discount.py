import numpy as np
from env.kernel_bandit_env import KernelBanditEnv, kernel_bandit_generator
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.gaussian_process.kernels import Matern
from scipy import linalg
import util
from simulator import Simulator
import matplotlib.pyplot as plt
from tqdm import tqdm
from alg.alg import Alg

class GPUCB(Alg):
    name = 'gpucb_discount'
    config = {
        'tol': 0.0001,
        'ucb': 'chowdury2017',
        'kernel': {
            'name': 'matern',
            'length_scale': 1.0,
            'nu': 2.5,
        },
        'beta_scale': 1.0,
        'lambda': 1,
        'eta': 0.9,
        'v': 0.1,
        'seed': None,
    }

    def __init__(self, env, config):
        self.env = env
        for k, v in config.items():
            if k not in self.config:
                raise Exception(f'Invalid config "{k}"')
            self.config[k] = v
        C = self.config

        self.action_history = np.zeros(env.T + 1, dtype=int) - 1
        self.reward_history = np.zeros(env.T + 1)

        K = util.kernel_matrix(env.A, C['kernel'])
        self.Phi = np.linalg.cholesky(K)

        self.gram = np.zeros((self.env.N, self.env.N))
        self.z = np.zeros(self.env.N)
        self.information_gain = util.InformationGain(self.Phi, env.T, C['lambda'])

    def action(self, t):
        C = self.config
        env_C = self.env.config
        mu = np.zeros(self.env.N)
        sigma = np.zeros(self.env.N)
        w_t = C['eta'] ** (-t)
        gram_inv = np.linalg.inv(self.gram + C['lambda'] * w_t * np.eye(self.env.N))
        if t > 1:
            Phi = self.Phi
            mu = Phi @ gram_inv @ Phi.T @ self.z
            for a in range(self.env.N):
                u = Phi[a, :]
                sigma[a] = C['lambda'] * gram_inv.dot(u).dot(u) * w_t

        beta = env_C['mean_reward_bound'] + env_C['sigma'] * np.sqrt(2 * (self.information_gain.get(t) + np.log(1 / C['tol']))) #/ np.sqrt(C['lambda'])

        score = mu + beta * np.sqrt(sigma) * C['beta_scale']
        return np.argmax(score)

    def update(self, t, a, r):
        C = self.config
        self.action_history[t] = a
        self.reward_history[t] = r

        # row of Phi
        u = self.Phi[[a], :].T

        w_t = C['eta'] ** (-t)
        self.gram += u @ u.T * w_t
        self.z[a] += r * w_t
