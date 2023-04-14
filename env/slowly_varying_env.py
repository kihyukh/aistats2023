from env.env import Env

class SlowlyVaryingEnv(Env):
    name = 'slowly_varying'
    config = {
        'switch_time': None,
        'T': None,
        'N': None,
        'env': None,
    }

    def __init__(self, config):
        super().__init__(config)

        self.env = Env.create(config['env']['name'], config['env'])
        self.env2 = Env.create(config['env']['name'], config['env'])

        self.T = config['T']
        self.A = self.env.A
        self.switch_time = config['switch_time']

    def mean_reward(self, t, action):
        r1 = self.env.mean_reward(t, action)
        r2 = self.env2.mean_reward(t, action)
        q2 = self.switch_time
        q1 = int(q2 / 2)
        q3 = int((self.T + q2 / 2))
        if t < q1:
            return r2 * t / q1 + r1 * (q1 - t) / q1
        if t < q2:
            return r2 * (q2 - t) / (q2 - q1) + (-r1) * (t - q1) / (q2 - q1)
        if t < q3:
            return -r1 * (q3 - t) / (q3 - q2) - r2 * (t - q2) / (q3 - q2)
        return r1 * (t - q3) / (self.T - q3) - r2 * (self.T - t) / (self.T - q3)

