from env.env import Env
import numpy as np

class SwitchingEnv(Env):
    name = 'switching'
    config = {
        'env': None,
        'switch_time': None,
        'switch_count': None,
        'T': None,
        'N': None,
        'reward_scale': 1,
        'random_reward': False,
        'first_hard_switch': False,
    }

    def __init__(self, config):
        super().__init__(config)

        self.env = Env.create(config['env']['name'], config['env'])
        self.T = config['T']
        self.A = self.env.A

        if type(self.config['switch_time']) == int:
            self.switch_time = [self.config['switch_time']]
        elif type(self.config['switch_time']) == list:
            self.switch_time = self.config['switch_time']
        elif self.config['switch_count'] is not None:
            self.switch_time = np.random.choice(
                np.arange(config['T'] - 2000) + 1000,
                size=int(self.config['switch_count']),
                replace=False
            )
        else:
            raise Exception('Unsupported switch_time type')

        if config['random_reward']:
            self.envs = [self.env] + [
                Env.create(config['env']['name'], config['env'], A=self.A)
                for _ in range(len(self.switch_time))
            ]

    def mean_reward(self, t, action):
        C = self.config
        for i, switch_time in enumerate(self.switch_time):
            if switch_time >= t:
                if C['random_reward'] and not C['first_hard_switch']:
                    return self.envs[i].mean_reward(t, action)
                if C['random_reward'] and C['first_hard_switch'] and i > 1:
                    return self.envs[i].mean_reward(t, action)
                return self.env.mean_reward(t, action) * ((-1) ** i)
        if C['random_reward']:
            return self.envs[len(self.switch_time)].mean_reward(t, action)
        return self.env.mean_reward(t, action) * ((-1) ** len(self.switch_time))

