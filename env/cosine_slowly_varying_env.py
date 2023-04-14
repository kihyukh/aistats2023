import numpy as np
from env.env import Env
from math import pi as pi


class CosineVaryingBanditEnv(Env):
    name = 'cosine_varying'
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

    def phase(self,t):
        x1=1000
        x2=3000
        x3=4000
        x4=6000
        y1=1.0*pi
        y2=2.0*pi
        if t<x1:
            return 0.0
        if t<x2:
            return (t-x1)*y1/(x2-x1)
        if t<x3:
            return y1
        if t<x4:
            return (t-x3)*(y2-y1)/(x4-x3) + y1
        return y2

    def mean_reward(self, t, action):
        C = self.config
        #print("phase: ",self.phase(t))
        score = np.sum(self.A[action, :] * self.theta)
        return np.cos(score * C['frequency'] + self.phase(t)) * C['amplitude']

