import numpy as np
from env.kernel_bandit_env import KernelBanditEnv, kernel_bandit_generator
from scipy import linalg
import util
from simulator import Simulator
from scheduler import Scheduler
import matplotlib.pyplot as plt
from alg.alg import Alg
from alg.opkb import OPKB
from neural_network import NN
import torch

from alg.opkb import kernel_optimal_design

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class OPNN(OPKB):
    name = 'opnn'
    config = {
        'm': 2048,
        'L': 3,
        'tol': 0.0001,
        'gamma': 100,
        'mixture_probability': 0.1,
        'beta_scale': 20,
        'mu_scale': 1 / 50,
        'e_scale': 0.01,
        'g_scale': 0.01,
        'epsilon_scale': 1.5,
        'j_star': 4,
        'change_detection': False,
        'J': 1000,
        'batch_size': 4096,
        'eta': 1e-8,
        'optimizer': 'sgd',
        'lambda': 0.001,
        'seed': None,
    }

    def __init__(self, env, config):
        if 'seed' in config and config['seed'] is not None:
            torch.manual_seed(config['seed'])
            del config['seed']

        self.env = env

        for k, v in config.items():
            if k not in self.config:
                raise Exception(f'Invalid config "{k}"')
            self.config[k] = v
        C = self.config

        N, d = env.A.shape
        self.gamma = C['gamma'] / env.T
        self.model = NN(d, int(C['m']), int(C['L']))
        self.Phi = self.model.feature_map(env.A).cpu().numpy()
        self.Phi0 = self.Phi.copy()
        self.deltas0 = {}
        self.pi = kernel_optimal_design(self.Phi, C['gamma'] / env.T)

        self.information_gain = util.InformationGain(self.Phi, env.T, C['gamma'])
        self.gamma_T = self.information_gain.get_exact(env.T)
        self.E = int(np.ceil(
            4 * self.gamma_T * np.log(8 * env.N * env.T * np.log2(env.T) / C['tol']) * C['e_scale']
        ))
        self.alpha = C['gamma'] / (4 * np.log(8 * env.T * np.log2(env.T) * env.N / C['tol']))
        self._initialize(1)

        self.detection_count = 0

    def _end_of_block_update(self, t, a, r):
        self.block_end_time[self.m] = t
        C = self.config

        self.model.initialize()
        self.model.train(self.env.A, self.history, C)
        self.Phi = self.model.feature_map(self.env.A).cpu().numpy()
        self.information_gain = util.InformationGain(self.Phi, self.env.T, C['gamma'])
        self.pi = kernel_optimal_design(self.Phi, self.gamma)

        self.gram_inv = [
            util.s_inv(util.S(p, self.Phi, self.gamma)) for p in self.ps
        ]
        z = np.zeros((self.m + 1, self.env.N))
        for m in range(self.m + 1):
            history = self.history[:self.block_end_time[m]]
            R_hat, delta = self._compute_delta(history)
            self.reward_estimates[m] = R_hat
            self.deltas[m] = delta.copy()

        R_hat = self.reward_estimates[self.m]
        _, delta = self._compute_delta(history)
        self.beta = self._compute_beta(self.m + 1)
        self.mu = self._compute_mu(self.m + 1)
        x = util.OP(self.Phi, delta, self.beta, self.gamma)
        G = delta <= 2 * self.alpha * self.gamma_T / self.beta * C['beta_scale'] * C['g_scale']
        g_regrets = [self.env.regret(t, aa) for aa in np.arange(self.env.N)[G]]
        if sum(G) == self.env.N:
            pG = self.pi
        elif sum(G) == 1:
            pG = G.astype(float)
        else:
            # TODO: cache
            Phi = self.Phi[G, :][:, G]
            p = kernel_optimal_design(Phi, C['gamma'] / self.env.T)
            pG = np.zeros(self.env.N)
            pG[G] = p

        mixture = C['mixture_probability']
        self.p = (1 - self.mu) * (x * (1 - mixture) + pG * mixture) + self.mu * self.pi
        self.ps.append(self.p)
        self.gram_inv.append(
            util.s_inv(util.S(self.p, self.Phi, self.gamma))
        )

        self.tau = t + 1
        self.m += 1
        self.scheduler = Scheduler(self.m, self.E, j_star=int(C['j_star']))
