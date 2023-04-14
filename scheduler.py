import numpy as np
from collections import defaultdict


class Scheduler:
    def __init__(self, block_index, E, j_star=0, prob_scale=1, block_test=False):
        self.block_index = block_index
        self.E = E
        self.block_test = block_test
        schedule = {}
        for j in range(block_index):
            if j < j_star:
                continue
            p = np.sqrt(2 ** (j - block_index)) * prob_scale
            schedule[j] = np.random.rand(2 ** (block_index - j)) < p
        self.schedule = schedule

    def get_index(self, t):
        if self.block_index == 0:
            return 0
        assert t >= 0 and t < self.E * (2 ** self.block_index)
        for j in range(self.block_index):
            if j not in self.schedule:
                continue
            current_index = int(t / self.E / (2 ** j))
            if self.schedule[j][current_index]:
                return j
        return self.block_index

    def get_intervals_ending_at(self, t):
        ret = []
        for j in range(self.block_index):
            if j not in self.schedule:
                continue
            if (t + 1) % (self.E * (2 ** j)) == 0:
                index = (int((t + 1) / self.E / (2 ** j))) - 1
                if self.schedule[j][index]:
                    ret.append((j, index))
        if self.block_test and t + 1 == self.E * (2 ** self.block_index):
            ret.append((self.block_index, 0))
        return ret


if __name__ == '__main__':
    block_index = 3
    E = 2
    s = Scheduler(block_index, E)
    print(s.schedule)
    for t in range(E * (2 ** block_index)):
        print(t, s.get_index(t), s.get_intervals_ending_at(t))
