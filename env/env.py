import numpy as np
import copy
from abc import ABC, abstractmethod
from scipy.stats import truncnorm

class Env(ABC):
    logger = None
    subclasses = {}
    config = {
        'reward_bound': 1,
        'mean_reward_bound': None,
        'sigma': 0.2,
    }

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        assert hasattr(cls, 'name')
        assert cls.name not in cls.subclasses
        cls.subclasses[cls.name] = cls

    @classmethod
    def create(cls, name, conf, **kwargs):
        assert name in cls.subclasses
        C = copy.deepcopy(conf)
        del C['name']
        return cls.subclasses[name](C, **kwargs)

    def __init__(self, config, **kwargs):
        temp = Env.config.copy()
        temp.update(self.config)
        self.config = temp
        for k, v in config.items():
            if k not in self.config:
                raise Exception(f'Invalid config "{k}"')
            self.config[k] = v
        C = self.config

        # hack: just for convenience
        self.T = C['T']
        self.N = C['N']
        self.sigma = C['sigma']

    @abstractmethod
    def mean_reward(self, t, action):
        pass

    def run(self, t, action):
        C = self.config
        m = self.mean_reward(t, action)
        if C['reward_bound'] == 0:
            return (
                m + np.random.normal() * self.sigma
            )
        assert abs(m) <= C['reward_bound']
        width = (C['reward_bound'] - abs(m)) / C['sigma']
        reward = truncnorm(-width, width, loc=m, scale=C['sigma']).rvs(1).item()
        assert abs(reward) <= C['reward_bound']
        return reward

    def regret(self, t, action):
        max_reward = max([self.mean_reward(t, a) for a in range(self.N)])
        return max_reward - self.mean_reward(t, action)

    def set_logger(self, logger):
        self.logger = logger

    def log(self, k, v):
        if self.logger:
            self.logger.log(k, v)
