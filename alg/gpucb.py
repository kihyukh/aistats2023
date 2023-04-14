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
    name = 'gpucb'
    config = {
        'tol': 0.0001,
        'window': 0,
        'kernel': {
            'name': 'rbf',
            'length_scale': 0.2,
        },
        'lambda': 20,
        'v': 0.01,
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

        self.gram_inv = np.eye(self.env.N) * C['lambda']
        self.z = np.zeros(self.env.N)
        self.action_counts = np.zeros(self.env.N)
        self.information_gain = util.InformationGain(self.Phi, env.T, C['lambda'])

    def action(self, t):
        C = self.config
        mu = np.zeros(self.env.N)
        sigma = np.zeros(self.env.N)
        if t > 1:
            Phi = self.Phi
            mu = Phi @ self.gram_inv @ Phi.T @ self.z
            for a in range(self.env.N):
                u = Phi[a, :]
                sigma[a] = C['lambda'] * self.gram_inv.dot(u).dot(u)

        if C['window']:
            t = min(t, C['window'])
        beta = 1 + np.sqrt(2 * (self.information_gain.get(t) / 2 + 1 + np.log(1 / C['tol'])))
        beta *= C['v']

        score = mu + beta * np.sqrt(sigma)
        return np.argmax(score)

    def update(self, t, a, r):
        C = self.config
        self.action_history[t] = a
        self.reward_history[t] = r

        A = self.gram_inv

        # row of Phi
        u = self.Phi[[a], :].T

        self.gram_inv -= A @ u @ u.T @ A / (1 + u.T @ A @ u)
        self.z[a] += r
        self.action_counts[a] += 1

        arm = self.Phi[:, a]
        if C['window'] and t > C['window']:
            A = self.gram_inv
            past_a = self.action_history[t - int(C['window'])]
            v = self.Phi[[past_a], :].T
            self.gram_inv += A @ v @ v.T @ A / (1 + v.T @ A @ v)
            self.action_counts[past_a] -= 1
            self.z[past_a] -= self.reward_history[t - int(C['window'])]
