from abc import ABC, abstractmethod
from env.env import Env
import copy

class Alg(ABC):
    logger = None
    subclasses = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        assert hasattr(cls, 'name')
        assert cls.name not in cls.subclasses
        cls.subclasses[cls.name] = cls

    @classmethod
    def create(cls, name, env, conf):
        assert name in cls.subclasses
        C = copy.deepcopy(conf)
        del C['name']
        return cls.subclasses[name](env, C)

    def __init__(self, env: Env):
        self.env = env

    @abstractmethod
    def action(self, t):
        pass

    def update(self, t, a, r):
        pass

    def set_logger(self, logger):
        self.logger = logger

    def log(self, k, v):
        if self.logger:
            self.logger.add(k, v)
