import numpy as np
import util
import matplotlib.pyplot as plt
from env.env import Env
from alg.alg import Alg
from logger import Logger


class Simulator(object):
    def __init__(self, env: Env, alg: Alg, T):
        self.env = env
        self.alg = alg
        self.T = T
        self.t = 1
        self.logger = Logger()
        self.env.set_logger(self.logger)
        self.alg.set_logger(self.logger)

    def __iter__(self):
        self.t = 1
        return self

    def __len__(self):
        return self.T

    def __next__(self):
        if self.t > self.T:
            raise StopIteration
        a = self.alg.action(self.t)
        r = self.env.run(self.t, a)
        self.alg.update(self.t, a, r)
        self.t += 1
        return (self.t - 1, a, r, self.logger.flush())
