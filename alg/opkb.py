import numpy as np
from env.kernel_bandit_env import KernelBanditEnv, kernel_bandit_generator
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.gaussian_process.kernels import Matern
import util
from simulator import Simulator
from scheduler import Scheduler
import matplotlib.pyplot as plt
from tqdm import tqdm
from alg.alg import Alg


class OPKB(Alg):
    name = 'opkb'
    config = {
        'kernel': {
            'name': 'rbf',
            'length_scale': 0.2,
        },
        'tol': 0.0001,
        'gamma': 10,
        'beta_scale': 10,
        'mu_scale': 1 / 10,
        'e_scale': 0.01,
        'mixture_probability': 0.1,
        'epsilon_scale': 1.5,
        'j_star': 4,
        'g_scale': 0.5,
        'change_detection': False,
        'seed': None,
    }

    def __init__(self, env, config):
        self.env = env

        for k, v in config.items():
            if k not in self.config:
                raise Exception(f'Invalid config "{k}"')
            self.config[k] = v
        C = self.config

        K = util.kernel_matrix(env.A, C['kernel'])
        self.Phi = np.linalg.cholesky(K)
        self.pi = kernel_optimal_design(self.Phi, C['gamma'] / env.T)

        self.gamma = C['gamma'] / env.T
        self.T = env.T

        self.information_gain = util.InformationGain(self.Phi, env.T, C['gamma'])
        self.gamma_T = self.information_gain.get_exact(env.T)

        self.E = int(np.ceil(
            4 * self.gamma_T * np.log(8 * env.N * env.T * np.log2(env.T) / C['tol']) * C['e_scale']
        ))
        self.alpha = C['gamma'] / (4 * np.log(8 * env.T * np.log2(env.T) * env.N / C['tol']))
        self._initialize(1)

        self.detection_count = 0

    def _initialize(self, t):
        C = self.config

        self.tau = t
        self.epoch_start_time = t
        self.m = 0
        self.beta = self._compute_beta(0)
        self.mu = self._compute_mu(0)
        self.gram_inv = {}
        self.deltas = {}
        self.reward_estimates = {}
        self.history = []
        self.p = self.pi
        self.ps = [self.pi]
        self.gram_inv[0] = util.s_inv(util.S(self.p, self.Phi, C['gamma'] / self.env.T))
        self.block_end_time = {}

        self.scheduler = Scheduler(0, self.E, j_star=int(C['j_star']))

    def _compute_beta(self, m):
        C = self.config
        t = self.E * (2 ** m)
        gamma_t = self.gamma_T
        mu = 0.5 * np.sqrt(1 / (2 ** m))
        epsilon = (40 + 16 * np.sqrt(self.alpha)) * mu
        return 2 * gamma_t / epsilon * C['beta_scale']

    def _compute_mu(self, m):
        C = self.config
        return 0.5 * np.sqrt(1 / (2 ** m)) * C['mu_scale']

    def _compute_epsilon(self, m):
        C = self.config
        mu = 0.5 * np.sqrt(1 / (2 ** m))
        return (40 + 16 * np.sqrt(self.alpha)) * mu

    def _compute_reward_estimate(self, history, gram_inv=None):
        if gram_inv is None:
            gram_inv = self.gram_inv
        zs = {}
        for policy_index, a, r in history:
            if policy_index not in zs:
                zs[policy_index] = np.zeros(self.env.N)
            zs[policy_index] += self.Phi[a, :] * r
        l = np.zeros(self.env.N)
        for policy_index, z in zs.items():
            l += gram_inv[policy_index].dot(z)
        theta_hat = l / len(history)
        return self.Phi @ theta_hat

    def _compute_delta(self, history, gram_inv=None):
        R_hat = self._compute_reward_estimate(history, gram_inv)
        return R_hat, (np.max(R_hat) - R_hat).ravel()

    def _end_of_block_update(self, t, a, r):
        C = self.config

        R_hat, delta = self._compute_delta(self.history)
        self.reward_estimates[self.m] = R_hat
        self.deltas[self.m] = delta

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
            Phi = self.Phi[G, :][:, G]
            p = kernel_optimal_design(Phi, C['gamma'] / self.env.T)
            pG = np.zeros(self.env.N)
            pG[G] = p

        mixture = C['mixture_probability']
        self.p = (1 - self.mu) * (x * (1 - mixture) + pG * mixture) + self.mu * self.pi
        self.ps.append(self.p)

        self.tau = t + 1
        self.m += 1
        self.gram_inv[self.m] = util.s_inv(util.S(self.p, self.Phi, self.gamma))
        self.scheduler = Scheduler(self.m, self.E, j_star=int(C['j_star']))

    def action(self, t):
        if self.config['change_detection']:
            policy_index = self.scheduler.get_index(t - self.tau)
        else:
            policy_index = self.m
        p = self.ps[policy_index]
        return np.random.choice(p.size, p=p)

    def _end_of_replay_change_detected(self, m, block_t):
        C = self.config
        assert C['change_detection']
        for j, replay_index in self.scheduler.get_intervals_ending_at(block_t):
            start_index = self.tau + self.E * (2 ** j) * replay_index - self.epoch_start_time
            end_index = start_index + self.E * (2 ** j)
            replay_history = self.history[start_index:end_index]
            assert len(replay_history) == self.E * (2 ** j)
            assert all([policy_index <= j for policy_index, _, _ in replay_history])
            R_hat, delta = self._compute_delta(replay_history)
            for k in range(int(C['j_star']), m):
                r = min(k, j)
                epsilon = (self._compute_epsilon(k) + self._compute_epsilon(j)) / 2
                diff = max(
                    np.max(self.deltas[k] - 4 * delta),
                    np.max(delta - 4 * self.deltas[k]),
                ) / (4 * epsilon * C['epsilon_scale'])
                if diff > 1:
                    return True
        return False

    def update(self, t, a, r):
        C = self.config

        block_t = t - self.tau
        if C['change_detection']:
            policy_index = self.scheduler.get_index(block_t)
        else:
            policy_index = self.m
        self.history.append((policy_index, a, r))

        if C['change_detection'] and self._end_of_replay_change_detected(self.m, block_t):
            # Change detected
            self.detection_count += 1
            if not self.detection_count >= 100:
                self._initialize(t + 1)
                return

        # Change not detected
        if t - self.tau + 1 >= (2 ** self.m) * self.E:
            self._end_of_block_update(t, a, r)


def kernel_optimal_design(Phi, gamma):
    k, _ = Phi.shape
    return util.OP(Phi, np.zeros(k), 1, gamma)

