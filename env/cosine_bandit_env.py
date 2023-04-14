import numpy as np
from env.env import Env

class CosineBanditEnv(Env):
    name = 'cosine'
    config = {
        'N': 10,
        'd': 2,
        'frequency': 3,
        'amplitude': 0.5,
        'phase': None,
        'T': 10000,
    }

    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        C = self.config

        A = kwargs.get('A')
        if A is None:
            A = np.random.randn(C['N'], C['d'])
            A /= np.sqrt(np.square(A).sum(axis=1))[:, np.newaxis]
        theta = np.random.randn(C['d'])
        theta /= np.sqrt(np.square(theta).sum())

        self.A = A
        self.theta = theta
        if C['phase'] is None:
            self.phase = np.random.rand() * 2 * np.pi
        else:
            self.phase = C['phase'] * np.pi

    def mean_reward(self, t, action):
        C = self.config
        score = np.sum(self.A[action, :] * self.theta)
        return np.cos(score * C['frequency'] + self.phase) * C['amplitude']

